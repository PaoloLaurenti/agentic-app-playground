fn main() {
    println!("{}", greeting("world"));
}

fn greeting(name: &str) -> String {
    format!("Hello, {name}!")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greets_the_given_name() {
        assert_eq!(greeting("Bedrock"), "Hello, Bedrock!");
    }
}
