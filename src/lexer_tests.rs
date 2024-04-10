#[cfg(test)]
mod lexer_test {
    use crate::{diagnostics::TextSpan, SyntaxKind, SyntaxToken, SyntaxTree, Value};

    #[test]
    fn lexer() {
        let tokens = SyntaxTree::parse_tokens("a = 10".to_string());

        let tokens_expected = Vec::from([
            SyntaxToken {
                kind: SyntaxKind::IdentifierToken,
                position: 0,
                text: "a".into(),
                value: None,
                span: TextSpan {
                    start: 0,
                    length: 1,
                },
            },
            SyntaxToken {
                kind: SyntaxKind::WhiteSpace,
                position: 1,
                text: " ".into(),
                value: None,
                span: TextSpan {
                    start: 1,
                    length: 1,
                },
            },
            SyntaxToken {
                kind: SyntaxKind::Equals,
                position: 3,
                text: "=".into(),
                value: None,
                span: TextSpan {
                    start: 3,
                    length: 1,
                },
            },
            SyntaxToken {
                kind: SyntaxKind::WhiteSpace,
                position: 3,
                text: " ".into(),
                value: None,
                span: TextSpan {
                    start: 3,
                    length: 1,
                },
            },
            SyntaxToken {
                kind: SyntaxKind::Number,
                position: 4,
                text: "10".into(),
                value: Some(Value {
                    value: Some(10),
                    bool: None,
                }),
                span: TextSpan {
                    start: 4,
                    length: 2,
                },
            },
        ]);

        assert_eq!(tokens, tokens_expected);
    }

    fn requieres_seperator(kind: SyntaxKind, kind2: SyntaxKind)  -> bool {
        if kind == SyntaxKind::IdentifierToken && kind2 == SyntaxKind::IdentifierToken {
            return true
        }

        if kind == SyntaxKind::IdentifierToken && kind2 == SyntaxKind::TrueKeyword {
            return true
        }

        if kind == SyntaxKind::FalseKeyword && kind2 == SyntaxKind::IdentifierToken {
            return true
        }

        false
    }
}
