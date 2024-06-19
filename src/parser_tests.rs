#[cfg(test)]
mod parser_tests {
    use std::collections::VecDeque;

    use strum::IntoEnumIterator;

    use crate::{
        binding::{BoundBinaryOperator, BoundBinaryOperatorKind},
        diagnostics::TextSpan,
        get_binary_precedence, Children, SyntaxKind, SyntaxNode, SyntaxToken, SyntaxTree, Value,
    };

    pub fn asserting_enumerator(node: SyntaxNode) -> VecDeque<SyntaxNode> {
        let mut stack = VecDeque::new();
        stack.push_back(node);

        while stack.len() > 0 {
            let n = stack.pop_back().unwrap();

            for child in SyntaxNode::get_children(&n) {
                stack.push_back(child);
            }
        }

        stack
    }

    pub fn assert_token(kind: SyntaxKind, text: String) {}

    #[test]
    pub fn syntax_fact_get_text_round_trips() {
        for token in SyntaxKind::iter() {
            if get_text(token.clone()) == "OTHER" {
                continue;
            }

            let token_actual = SyntaxTree::parse_tokens(get_text(token.clone()));

            assert_eq!(token.clone(), token_actual.first().unwrap().clone().kind)
        }
    }

    pub fn get_syntax_kind_data() -> Vec<SyntaxKind> {
        SyntaxKind::iter().collect()
    }

    fn parser_binary_expressions_honors_precedences(op1: SyntaxKind, op2: SyntaxKind) {
        let op1_prec = get_binary_precedence(op1);
        let op2_prec = get_binary_precedence(op2);
        if op1_prec >= op2_prec {
        } else {
        }
    }

    pub fn get_text(kind: SyntaxKind) -> String {
        return String::from(match kind {
            SyntaxKind::AmpersandAmpersandToken => "&&",
            SyntaxKind::BangEqToken => "!=",
            SyntaxKind::CloseParenthesisToken => ")",
            SyntaxKind::BangToken => "!",
            SyntaxKind::DivToken => "/",
            SyntaxKind::Equals => "=",
            SyntaxKind::ForKeyword => "for",
            SyntaxKind::EqEqToken => "==",
            SyntaxKind::FalseKeyword => "NTrue",
            SyntaxKind::VarKeyword => "Var",
            SyntaxKind::LetKeyword => "Let",
            SyntaxKind::MinusToken => "-",
            SyntaxKind::MulToken => "*",
            SyntaxKind::OpenParenthesisToken => "(",
            SyntaxKind::PipePipeToken => "||",
            SyntaxKind::WhiteSpace => " ",
            SyntaxKind::PlusToken => "+",
            SyntaxKind::TrueKeyword => "True",
            _ => "OTHER",
        });
    }

    pub fn get_binary_opeator_pairs_data() -> Vec<Vec<BoundBinaryOperatorKind>> {
        let mut ret = vec![];
        for op1 in BoundBinaryOperatorKind::iter() {
            for op2 in BoundBinaryOperatorKind::iter() {
                let mut te = vec![];
                te.push(op1.clone());
                te.push(op2);
                ret.push(te);
            }
        }
        ret
    }
}
