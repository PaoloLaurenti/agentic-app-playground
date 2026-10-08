//! `app prompts-push`: copies to Langfuse every prompt file whose text is not yet a version there
//! (ADR 0009). The file stays the source; Langfuse only numbers the copy.

use std::path::PathBuf;

use anyhow::{Context, Result};
use observability::{LangfuseConfig, PromptClient, PromptRegistry};
use prompts::Prompt;
use toml_edit::{DocumentMut, value};

/// What pushing one prompt file did.
#[derive(Debug, PartialEq, Eq)]
pub enum Pushed {
    /// The version the file names already holds its text.
    Unchanged,
    /// A new version was created, and the file now names it.
    Created(u32),
}

/// The repository's prompt files, found from the crate rather than from the current directory.
const PROMPTS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../prompts");

/// Pushes every prompt file in the repository, and prints what happened to each.
pub async fn push_all(langfuse: Option<LangfuseConfig>) -> Result<()> {
    let langfuse = langfuse.context("prompts-push needs the Langfuse keys in .env")?;
    let registry = PromptClient::new(&langfuse);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(PROMPTS_DIR)
        .with_context(|| format!("cannot read {PROMPTS_DIR}"))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<_, _>>()?;
    paths.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "toml")
    });
    paths.sort();
    for path in paths {
        let text = std::fs::read_to_string(&path)?;
        let (pushed, new_text) = push_file(&text, &registry)
            .await
            .with_context(|| format!("cannot push {}", path.display()))?;
        let name = path.file_stem().unwrap_or_default().to_string_lossy();
        match pushed {
            Pushed::Unchanged => println!("{name}: unchanged"),
            Pushed::Created(version) => {
                std::fs::write(&path, new_text)?;
                println!("{name}: created version {version}");
            }
        }
    }
    Ok(())
}

/// The version a new prompt file starts from, before its first push.
const NEVER_PUSHED: u32 = 0;

/// Pushes one prompt file, and returns what happened along with the file's new text.
pub async fn push_file(text: &str, registry: &dyn PromptRegistry) -> Result<(Pushed, String)> {
    let file = Prompt::from_toml(text)?;
    // Version 0 marks a prompt that was never pushed: there is nothing to compare it with.
    let stored = match file.version {
        NEVER_PUSHED => None,
        version => registry.version(&file.name, version).await?,
    };
    if stored.as_ref() == Some(&file) {
        return Ok((Pushed::Unchanged, text.to_owned()));
    }
    let version = registry.create(&file).await?;
    // Edited as a document rather than rewritten, so that comments and layout stay as they are.
    let mut document: DocumentMut = text.parse()?;
    document["version"] = value(i64::from(version));
    Ok((Pushed::Created(version), document.to_string()))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use observability::Error;

    use super::*;

    /// Keeps versions in memory, numbered from 1 as Langfuse numbers them.
    #[derive(Default)]
    struct FakeRegistry {
        versions: Mutex<Vec<Prompt>>,
        lookups: Mutex<u32>,
    }

    impl FakeRegistry {
        fn holding(prompt: Prompt) -> Self {
            Self {
                versions: Mutex::new(vec![prompt]),
                ..Self::default()
            }
        }

        fn created(&self) -> usize {
            self.versions.lock().unwrap().len()
        }
    }

    #[async_trait::async_trait]
    impl PromptRegistry for FakeRegistry {
        async fn version(&self, name: &str, version: u32) -> Result<Option<Prompt>, Error> {
            *self.lookups.lock().unwrap() += 1;
            let versions = self.versions.lock().unwrap();
            Ok(versions
                .iter()
                .find(|prompt| prompt.name == name && prompt.version == version)
                .cloned())
        }

        async fn create(&self, prompt: &Prompt) -> Result<u32, Error> {
            let mut versions = self.versions.lock().unwrap();
            let version = versions.iter().filter(|p| p.name == prompt.name).count() as u32 + 1;
            versions.push(Prompt {
                version,
                ..prompt.clone()
            });
            Ok(version)
        }
    }

    const GREETER_V1: &str = r#"name = "greeter"
# Written by `just prompts-push`.
version = 1

[config]
max_tokens = 200

[[messages]]
role = "user"
content = "My name is {{name}}."
"#;

    #[tokio::test]
    async fn a_file_whose_version_holds_its_text_is_not_pushed() {
        let registry = FakeRegistry::holding(Prompt::from_toml(GREETER_V1).unwrap());

        let (pushed, text) = push_file(GREETER_V1, &registry).await.unwrap();

        assert_eq!(pushed, Pushed::Unchanged);
        assert_eq!(text, GREETER_V1);
        assert_eq!(registry.created(), 1);
    }

    #[tokio::test]
    async fn an_edited_file_becomes_a_new_version_and_names_it() {
        let registry = FakeRegistry::holding(Prompt::from_toml(GREETER_V1).unwrap());
        let edited = GREETER_V1.replace("My name is", "Call me");

        let (pushed, text) = push_file(&edited, &registry).await.unwrap();

        assert_eq!(pushed, Pushed::Created(2));
        assert_eq!(
            text,
            r#"name = "greeter"
# Written by `just prompts-push`.
version = 2

[config]
max_tokens = 200

[[messages]]
role = "user"
content = "Call me {{name}}."
"#
        );
        let stored = registry.version("greeter", 2).await.unwrap().unwrap();
        assert_eq!(stored.messages[0].content, "Call me {{name}}.");
    }

    #[tokio::test]
    async fn a_prompt_never_pushed_becomes_version_1_without_a_lookup() {
        let registry = FakeRegistry::default();
        let new = GREETER_V1.replace("version = 1", "version = 0");

        let (pushed, text) = push_file(&new, &registry).await.unwrap();

        assert_eq!(pushed, Pushed::Created(1));
        assert_eq!(text, GREETER_V1);
        assert_eq!(*registry.lookups.lock().unwrap(), 0);
    }
}
