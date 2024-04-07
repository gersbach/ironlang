use core::panic;

use crate::{diagnostics::DiagnosticBag, BinaryExpressionSyntax, SyntaxKind, SyntaxNode, UnaryExpressionSyntax, Value};


#[derive(Clone)]
pub enum BoundNodeKind {
    BoundUnaryExpressionNode(BoundUnaryExpressionNode),
    BoundUnaryOperatorKind(BoundUnaryOperatorKind),
    BoundLiteralExpression(BoundLiteralExpression),
    BoundBinaryExpression(BoundBinaryExpression),
    Unkown
}

impl BoundNodeKind {
    pub fn get_type(&self) -> Type {
        if let BoundNodeKind::BoundLiteralExpression(lit) = self {
            if lit.is_bool() {
                return Type::Bool
            } else {
                return Type::Number
            }
        } else if let BoundNodeKind::BoundBinaryExpression(bin) = self {
            return bin.bound_binary_operator_kind.result_type.clone()
        } else if let BoundNodeKind::BoundUnaryExpressionNode(un) = self {
            return un.operator_kind.result_type.clone()
        }
        // panic!()
        Type::Unkown
    }
}

#[derive(Clone)]
pub struct BoundLiteralExpression {
    pub value_num: Option<i32>,
    pub value_bool: Option<bool>
}

impl BoundLiteralExpression {
    pub fn is_bool(&self) -> bool {
        self.value_bool.is_some()
    }

    pub fn is_num(&self) -> bool {
        self.value_num.is_some()
    }
}

impl BoundLiteralExpression {
    pub fn into_value(&self) -> Value {
        Value {
            value: self.value_num,
            bool: self.value_bool
        }
    }
}

#[derive(PartialEq, Clone)]
pub enum BoundUnaryOperatorKind {
    Identity,
    Negation,
    LogicalNegation
}

#[derive(Clone)]
pub enum BoundBinaryOperatorKind {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    LogicalAnd,
    LogicalOr,
    Equals,
    DoesNotEqual,
}

// enum BoundBinaryOperatorKind {
//     Plus,
//     Minus,
//     Times,
//     Division,
// }


#[derive(PartialEq, Clone)]
pub enum Type {
    Number,
    Bool,
    Any,
    Unkown
}

#[derive(Clone)]
pub struct BoundUnaryExpressionNode {
    pub operator_kind: Box<BoundUnaryOperator>,
    pub operand: Box<BoundNodeKind>
}

#[derive(Clone)]
pub struct BoundBinaryExpression {
    pub left: Box<BoundNodeKind>,
    pub bound_binary_operator_kind: Box<BoundBinaryOperator>,
    pub right: Box<BoundNodeKind>
}

#[derive(Clone)]
pub struct BoundUnaryOperator {
    pub syntax_kind: SyntaxKind,
    pub operator_kind: BoundUnaryOperatorKind,
    pub operand_type: Type,
    pub result_type: Type,
    
}


impl BoundUnaryOperator {
    pub fn new(syntax_kind: SyntaxKind, operator_kind: BoundUnaryOperatorKind, operand_type: Type, result_type: Type) -> Self {
        Self {
            syntax_kind,
            operator_kind,
            operand_type,
            result_type
        }
    }

    pub fn new_same_type(syntax_kind: SyntaxKind, operator_kind: BoundUnaryOperatorKind, operand_type: Type) -> Self {
        Self {
            syntax_kind,
            operator_kind,
            operand_type: operand_type.clone(),
            result_type: operand_type
        }
    }

    pub fn bind(syntax_kind: SyntaxKind, operand_type: Type) -> BoundUnaryOperator {

        let thing = vec![Self::new_same_type(SyntaxKind::PlusToken, BoundUnaryOperatorKind::Identity, Type::Number),
        Self::new_same_type(SyntaxKind::MinusToken, BoundUnaryOperatorKind::Negation, Type::Number), 
        Self::new_same_type(SyntaxKind::BangToken, BoundUnaryOperatorKind::LogicalNegation, Type::Number)];

        for op in thing {
            if (op.syntax_kind == syntax_kind && op.operand_type == operand_type) {
                return op
            }
        } 

        panic!()
    }
}

#[derive(Clone)]
pub struct BoundBinaryOperator {
    pub syntax_kind: SyntaxKind,
    pub operator_kind: BoundBinaryOperatorKind,
    pub left_type: Type,
    pub right_type: Type,
    pub result_type: Type,
    
}


impl BoundBinaryOperator {
    pub fn new(syntax_kind: SyntaxKind, operator_kind: BoundBinaryOperatorKind, left_type: Type, right_type: Type, result_type: Type) -> Self {
        Self {
            syntax_kind,
            operator_kind,
            left_type,
            right_type,
            result_type
        }
    }

    pub fn new_same_type(syntax_kind: SyntaxKind, operator_kind: BoundBinaryOperatorKind, operand_type: Type) -> Self {
        Self {
            syntax_kind,
            operator_kind,
            left_type: operand_type.clone(),
            right_type: operand_type.clone(),
            result_type: operand_type.clone(),
        }
    }

    pub fn bind(syntax_kind: SyntaxKind, left_type: Type, right_type: Type) -> BoundBinaryOperator {

        if syntax_kind == SyntaxKind::EqEqToken {
            return Self::new(SyntaxKind::EqEqToken, BoundBinaryOperatorKind::Equals, left_type, right_type, Type::Bool)
        } else if syntax_kind == SyntaxKind::BangEqToken {
            return Self::new(SyntaxKind::BangEqToken, BoundBinaryOperatorKind::DoesNotEqual, left_type, right_type, Type::Bool)
        }

        let thing = vec![
            Self::new_same_type(SyntaxKind::PlusToken, BoundBinaryOperatorKind::Addition, Type::Number),
        Self::new_same_type(SyntaxKind::MinusToken, BoundBinaryOperatorKind::Subtraction, Type::Number),
        Self::new_same_type(SyntaxKind::MulToken, BoundBinaryOperatorKind::Multiplication, Type::Number),
        Self::new_same_type(SyntaxKind::DivToken, BoundBinaryOperatorKind::Division, Type::Number),
        Self::new_same_type(SyntaxKind::PipePipeToken, BoundBinaryOperatorKind::LogicalOr, Type::Bool),
        Self::new_same_type(SyntaxKind::AmpersandAmpersandToken, BoundBinaryOperatorKind::LogicalAnd, Type::Bool)];

        for op in thing {
            if op.syntax_kind == syntax_kind && op.left_type == left_type && op.right_type == right_type {
                return op
            }
        } 

        panic!()
        
    }
}

pub struct Binder {
    pub diagnostics: DiagnosticBag,
}

impl Binder {
    pub fn new() -> Self {
        Self { diagnostics: DiagnosticBag::new() }
    }

    pub fn get_diagnostics(&self) -> DiagnosticBag {
        return self.diagnostics.clone()
    }

    pub fn bind_expression(&self, expression_kind: SyntaxNode) -> BoundNodeKind {
        return match expression_kind {
            SyntaxNode::ParenthesizedExpression(par) => self.bind_expression(*par.sub_expression),
            SyntaxNode::NumberNode(number_node) => self.bind_literal_expression(number_node),
            SyntaxNode::BoolNode(bool_node) => self.bind_literal_expression(bool_node),
            SyntaxNode::UnaryExpressionSyntax(unary_node) => BoundNodeKind::BoundUnaryExpressionNode(self.bind_unary_expresssion(unary_node)),
            SyntaxNode::BinaryExpressionSyntax(binary) => BoundNodeKind::BoundBinaryExpression(self.bind_binary_expression(binary)),
            _ => BoundNodeKind::Unkown,
        }
    }

    pub fn bind_literal_expression(&self, syntax: Value ) -> BoundNodeKind {
        if let Some(value) = syntax.value {
            return BoundNodeKind::BoundLiteralExpression(BoundLiteralExpression { value_num: Some(value), value_bool: None })
        } else if let Some(bool) = syntax.bool {
            return BoundNodeKind::BoundLiteralExpression(BoundLiteralExpression { value_num: None, value_bool: Some(bool) })
        }
        return BoundNodeKind::BoundLiteralExpression(BoundLiteralExpression {value_bool: None, value_num: None})
    }

    pub fn bind_unary_expresssion(&self, bound_unary_expression: UnaryExpressionSyntax) -> BoundUnaryExpressionNode {
        let bound_operand = self.bind_expression(*bound_unary_expression.operand.clone());
        let bound_operator_kind = BoundUnaryOperator::bind(bound_unary_expression.clone().operator_token.try_into_syntax_kind(), bound_operand.get_type());
        return BoundUnaryExpressionNode { operator_kind: Box::from(bound_operator_kind), operand: Box::from(bound_operand) }
    }

    pub fn bind_binary_expression(&self, bound_binary_expression: BinaryExpressionSyntax) -> BoundBinaryExpression {
        let left = self.bind_expression(*bound_binary_expression.left);
        let right = self.bind_expression(*bound_binary_expression.right);
        let bound_operand = BoundBinaryOperator::bind(bound_binary_expression.operator_token.try_into_syntax_kind(), left.get_type(), right.get_type());

        return BoundBinaryExpression { left: Box::from(left), bound_binary_operator_kind: Box::from(bound_operand), right: Box::from(right)}
    }

    // pub fn bind_unary_operator_kind(&self, kind: SyntaxNode) -> BoundUnaryOperatorKind {
    //     match kind {
    //         SyntaxNode::OperatorNode(SyntaxKind::PlusToken) => BoundUnaryOperatorKind::Identity,
    //         SyntaxNode::OperatorNode(SyntaxKind::MinusToken) => BoundUnaryOperatorKind::Negation,
    //         SyntaxNode::OperatorNode(SyntaxKind::BangToken) => BoundUnaryOperatorKind::LogicalNegation,
    //         _ => panic!()
    //     }
    // }

    // pub fn bind_binary_operator_kind(&self, kind: SyntaxNode) -> BoundBinaryOperatorKind {
            
    //     match kind {
    //         SyntaxNode::OperatorNode(SyntaxKind::PlusToken) => BoundBinaryOperatorKind::Addition,
    //         SyntaxNode::OperatorNode(SyntaxKind::MinusToken) => BoundBinaryOperatorKind::Subtraction,
    //         SyntaxNode::OperatorNode(SyntaxKind::DivToken) => BoundBinaryOperatorKind::Division,
    //         SyntaxNode::OperatorNode(SyntaxKind::MulToken) => BoundBinaryOperatorKind::Multiplication,
    //         SyntaxNode::OperatorNode(SyntaxKind::AmpersandAmpersandToken) => BoundBinaryOperatorKind::LogicalAnd,
    //         SyntaxNode::OperatorNode(SyntaxKind::PipePipeToken) => BoundBinaryOperatorKind::LogicalOr,
    //         _ => panic!()
    //     }
    // }


}