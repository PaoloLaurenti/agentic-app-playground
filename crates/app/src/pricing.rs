//! Estimated cost of a call, from a price table written by hand. The model registry of Module 10
//! replaces it.

use llm_core::Usage;

/// Prices in USD per million tokens, before tax.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Price {
    pub input: f64,
    pub output: f64,
    pub cache_write: f64,
    pub cache_read: f64,
}

/// Bedrock prices in `eu-west-1` for `eu.*` inference profiles, from the AWS Price List API
/// (step 3.11). The cache write price is the five-minute one.
const PRICES: &[(&str, Price)] = &[
    (
        "eu.anthropic.claude-haiku-4-5",
        Price {
            input: 1.10,
            output: 5.50,
            cache_write: 1.375,
            cache_read: 0.11,
        },
    ),
    (
        "eu.anthropic.claude-sonnet-4-6",
        Price {
            input: 3.30,
            output: 16.50,
            cache_write: 4.125,
            cache_read: 0.33,
        },
    ),
    (
        "eu.anthropic.claude-sonnet-5",
        Price {
            input: 2.20,
            output: 11.00,
            cache_write: 2.75,
            cache_read: 0.22,
        },
    ),
];

/// The price of a model, matched on the start of its id so that version suffixes do not matter.
pub fn price_of(model_id: &str) -> Option<Price> {
    PRICES
        .iter()
        .find(|(prefix, _)| model_id.starts_with(prefix))
        .map(|(_, price)| *price)
}

/// The cost in USD of one call.
pub fn cost(usage: &Usage, price: &Price) -> f64 {
    let per_token = |tokens: u32, per_million: f64| f64::from(tokens) * per_million / 1_000_000.0;
    per_token(usage.input_tokens, price.input)
        + per_token(usage.output_tokens, price.output)
        + per_token(usage.cache_write_tokens, price.cache_write)
        + per_token(usage.cache_read_tokens, price.cache_read)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_id_with_a_version_suffix_finds_its_price() {
        let price = price_of("eu.anthropic.claude-haiku-4-5-20251001-v1:0").unwrap();

        assert_eq!(price.input, 1.10);
    }

    #[test]
    fn a_model_without_a_price_has_no_estimate() {
        assert_eq!(price_of("eu.amazon.nova-lite-v1:0"), None);
    }

    #[test]
    fn the_cost_adds_every_kind_of_token() {
        let usage = Usage {
            input_tokens: 15,
            output_tokens: 5,
            cache_read_tokens: 1_000_000,
            cache_write_tokens: 0,
        };
        let price = price_of("eu.anthropic.claude-haiku-4-5").unwrap();

        // 15 × 1.10 + 5 × 5.50 per million, plus one million cache reads at 0.11.
        let expected = 0.000_044 + 0.11;
        assert!((cost(&usage, &price) - expected).abs() < 1e-12);
    }
}
