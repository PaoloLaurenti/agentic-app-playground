//! The harness: the tools a model can ask for, and the loop that runs them for it.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// Something the model can ask the harness to do, such as searching the knowledge base. The model
/// never runs a tool: it names one in its reply, and the harness calls it and sends back the
/// result.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    /// The input the tool takes, read from the JSON the model wrote. A tool whose schema is only
    /// known at runtime, such as a remote one, takes `serde_json::Value` and overrides
    /// `input_schema`.
    type Input: serde::de::DeserializeOwned + schemars::JsonSchema + Send;

    /// The name the model uses to ask for this tool.
    fn name(&self) -> &str;
    /// What the tool does, written for the model: when to use it, and when not to.
    fn description(&self) -> &str;
    /// The JSON schema of the input, sent to the model with the tool's name and description.
    /// Subschemas are inlined, as for structured output (6.8), so that each field's constraints
    /// sit where the field is.
    fn input_schema(&self) -> serde_json::Value {
        schemars::generate::SchemaSettings::default()
            .with(|settings| settings.inline_subschemas = true)
            .into_generator()
            .into_root_schema_for::<Self::Input>()
            .to_value()
    }
    async fn call(&self, input: Self::Input, ctx: &ToolContext) -> Result<ToolOutput, ToolError>;
}

/// What a tool may need to know about the request it runs for. Empty until a tool needs something.
#[derive(Debug, Clone, Default)]
pub struct ToolContext {}

/// A tool's result, as the text that goes back to the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutput(pub String);

/// Why a tool did not produce a result. The message goes back to the model, so it says what to
/// change.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// The model asked for a tool that does not exist.
    #[error("no tool is named {0}")]
    UnknownTool(String),
    /// The input does not follow the tool's schema.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// A tool as the registry sees it, whatever its input type: callable with the JSON input the
/// model wrote in its reply.
#[async_trait::async_trait]
trait JsonCallable: Send + Sync {
    async fn call_json(
        &self,
        input: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError>;
}

#[async_trait::async_trait]
impl<T: Tool> JsonCallable for T {
    async fn call_json(
        &self,
        input: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        // With the path of the field at fault, so that the model knows what to correct: serde
        // alone says "expected a string" without saying where.
        let input = serde_path_to_error::deserialize(input)
            .map_err(|err| ToolError::InvalidInput(err.to_string()))?;
        self.call(input, ctx).await
    }
}

/// The tools the model may ask for, by name.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn JsonCallable>>,
}

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// The model could not tell the two apart.
    #[error("a tool named {0} is already registered")]
    DuplicateName(String),
}

impl ToolRegistry {
    pub fn register(&mut self, tool: impl Tool + 'static) -> Result<(), RegistryError> {
        match self.tools.entry(tool.name().to_owned()) {
            Entry::Occupied(entry) => Err(RegistryError::DuplicateName(entry.key().clone())),
            Entry::Vacant(entry) => {
                entry.insert(Box::new(tool));
                Ok(())
            }
        }
    }

    /// Runs the tool the model named on the input it wrote.
    pub async fn call(
        &self,
        name: &str,
        input: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::UnknownTool(name.to_owned()))?;
        tool.call_json(input, ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tool that answers with its own name, whatever its input.
    struct Named(&'static str);

    #[async_trait::async_trait]
    impl Tool for Named {
        type Input = serde_json::Value;

        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &str {
            "A tool for tests."
        }
        async fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput(self.0.into()))
        }
    }

    fn registry_of(tools: impl IntoIterator<Item = impl Tool + 'static>) -> ToolRegistry {
        let mut registry = ToolRegistry::default();
        for tool in tools {
            registry.register(tool).unwrap();
        }
        registry
    }

    #[tokio::test]
    async fn a_registered_tool_is_called_by_its_name() {
        let registry = registry_of([Named("get_current_date"), Named("search_knowledge_base")]);

        let output = registry
            .call(
                "search_knowledge_base",
                serde_json::json!({}),
                &ToolContext::default(),
            )
            .await
            .unwrap();

        assert_eq!(output, ToolOutput("search_knowledge_base".into()));
    }

    #[tokio::test]
    async fn a_name_no_tool_has_is_an_error_that_names_it() {
        let registry = registry_of([Named("get_current_date")]);

        let result = registry
            .call(
                "get_weather",
                serde_json::json!({}),
                &ToolContext::default(),
            )
            .await;

        assert!(matches!(result, Err(ToolError::UnknownTool(name)) if name == "get_weather"));
    }

    #[test]
    fn two_tools_with_the_same_name_cannot_both_be_registered() {
        let mut registry = registry_of([Named("get_current_date")]);

        let result = registry.register(Named("get_current_date"));

        assert!(
            matches!(result, Err(RegistryError::DuplicateName(name)) if name == "get_current_date")
        );
    }

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct GreetInput {
        name: String,
        times: u32,
    }

    /// A tool written over a typed input.
    struct Greet;

    #[async_trait::async_trait]
    impl Tool for Greet {
        type Input = GreetInput;

        fn name(&self) -> &str {
            "greet"
        }
        fn description(&self) -> &str {
            "Greets someone by name."
        }
        async fn call(
            &self,
            input: GreetInput,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput(
                vec![format!("Hello, {}!", input.name); input.times as usize].join(" "),
            ))
        }
    }

    async fn greet_with(input: serde_json::Value) -> Result<ToolOutput, ToolError> {
        registry_of([Greet])
            .call("greet", input, &ToolContext::default())
            .await
    }

    #[tokio::test]
    async fn a_tool_receives_the_model_s_input_as_its_own_type() {
        let output = greet_with(serde_json::json!({ "name": "Giulia", "times": 2 }))
            .await
            .unwrap();

        assert_eq!(output, ToolOutput("Hello, Giulia! Hello, Giulia!".into()));
    }

    #[tokio::test]
    async fn a_missing_field_is_invalid_input_that_names_the_field() {
        let err = greet_with(serde_json::json!({ "name": "Giulia" }))
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::InvalidInput(_)), "{err:?}");
        assert_eq!(err.to_string(), "invalid input: missing field `times`");
    }

    #[tokio::test]
    async fn a_field_of_the_wrong_type_is_invalid_input_that_names_the_field() {
        let err = greet_with(serde_json::json!({ "name": 3, "times": 2 }))
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::InvalidInput(_)), "{err:?}");
        assert!(err.to_string().contains("name"), "{err}");
    }
}
