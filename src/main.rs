mod binding;
mod diagnostics;
mod lexer_tests;
mod parser_tests;
use strum::IntoEnumIterator; // 0.17.1
use strum_macros::EnumIter; // 0.17.1


use binding::{BoundBinaryOperatorKind, BoundNodeKind, BoundUnaryOperatorKind, Type};
use diagnostics::{Diagnostic, DiagnosticBag, TextSpan};
use inline_colorization::*;
use std::{
    collections::{binary_heap::Iter, HashMap},
    fmt,
    hash::Hash,
    io::{stdin, stdout, Write},
    ops::Bound,
    vec,
};

use crate::binding::Binder;

fn main() {
    let mut variables = HashMap::new();

    while true {
        let mut line = String::new();
        print!("> ");
        let _ = stdout().flush();
        stdin()
            .read_line(&mut line)
            .expect("Did not enter a correct string");
        if line == "\n" {
            return;
        }

        let mut terminal = term::stdout().unwrap();

        let mut parser = Parser::new(line.clone());

        // if you write an ide, you care about whitespace
        // ASTs abstract syntax trees do not have whitespace or parenthesis
        // compiilers generally don't care
        // in the production enviornment, you expect a immutable represetnation
        // if syntax trees get modifieid and there are flags there is a lot of pain
        // ASTs have a slot for the type and the type checker fills that in
        let token = parser.parse();

        let compilation = Compilation::new(token.clone());
        let result = compilation.evaluate(&mut variables);

        parser.pretty_print(&token.expression.clone(), "".to_string(), true);

        // bound tree / ir / annotated representation contains the types

        if parser.diagnostics.len() > 0 {
            for diagnostic in parser.diagnostics.diagnostics {
                println!("{color_red}{diagnostic}{color_reset}");
            }
        } else {
            // let evaluateor = Evaluator::new(token.expression);
            // let value = evaluateor.evaluate();
            println!("RET {:?}", result);
        }

        // if line == "1 + 2 + 3\n" {
        //     println!("7");
        // } else {
        //     println!("invalid expression {line:?}")
        // }
    }
}

#[derive(Debug, PartialEq, Clone, Eq, EnumIter)]
enum SyntaxKind {
    Number,
    WhiteSpace,
    PlusToken,
    MinusToken,
    DivToken,
    MulToken,
    BadToken,
    EndOfFileToekn,
    OpenParenthesisToken,
    CloseParenthesisToken,
    BinaryExpression,
    TrueKeyword,
    FalseKeyword,
    AmpersandAmpersandToken,
    PipePipeToken,
    BangToken,
    EqEqToken,
    BangEqToken,
    IdentifierToken,
    Equals,
}

#[derive(Debug)]
struct Lexer {
    text: String,
    position: i32,
    diagnostics: DiagnosticBag,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxToken {
    kind: SyntaxKind,
    position: i32,
    text: String,
    value: Option<Value>,
    span: TextSpan,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct Value {
    value: Option<i32>,
    bool: Option<bool>,
}

impl Value {
    pub fn from_num(num: i32) -> Self {
        Self {
            value: Some(num),
            bool: None,
        }
    }

    pub fn from_bool(bool: bool) -> Self {
        Self {
            value: None,
            bool: Some(bool),
        }
    }

    pub fn is_bool(&self) -> bool {
        self.bool.is_some()
    }

    pub fn is_num(&self) -> bool {
        self.value.is_some()
    }

    pub fn get_type(&self) -> Type {
        if self.is_bool() {
            Type::Bool
        } else if self.is_num() {
            Type::Number
        } else {
            Type::Unkown
        }
    }

    pub fn get_bool(&self) -> bool {
        if let Some(value) = self.bool {
            return value;
        }

        panic!("SHOULD BE BOOL")
    }

    pub fn get_num(&self) -> i32 {
        if let Some(value) = self.value {
            return value;
        }

        panic!("SHOULD BE NUM")
    }
}

impl SyntaxToken {
    pub fn new(kind: SyntaxKind, position: i32, text: String, value: Option<Value>) -> Self {
        SyntaxToken {
            kind,
            position,
            text: text.clone(),
            value,
            span: TextSpan::new(position, text.clone().len() as i32),
        }
    }
}

impl Lexer {
    pub fn new(text: String) -> Self {
        Lexer {
            text,
            position: 0,
            diagnostics: DiagnosticBag::new(),
        }
    }

    fn current(&self) -> char {
        if self.position >= self.text.len() as i32 {
            return '\0';
        };
        return self
            .text
            .chars()
            .nth(self.position as usize)
            .unwrap_or('\0');
    }

    fn look_ahead(&self) -> String {
        self.peek(1)
    }

    fn peek(&self, offset: i32) -> String {
        let index = self.position + offset;

        return self
            .text
            .chars()
            .skip(index as usize)
            .take(offset as usize)
            .collect();
    }

    fn next(&mut self) {
        self.position += 1;
    }

    pub fn get_keyword_kind(&self, text: &str) -> SyntaxKind {
        match text {
            "True" => SyntaxKind::TrueKeyword,
            "NTrue" => SyntaxKind::FalseKeyword,
            _ => SyntaxKind::IdentifierToken,
        }
    }

    pub fn next_token(&mut self) -> SyntaxToken {
        // <numbers>
        // + - * /
        // <whitespace>

        if self.position as usize >= self.text.len() {
            return SyntaxToken::new(
                SyntaxKind::EndOfFileToekn,
                self.position,
                "\0".to_string(),
                None,
            );
        }

        if self.current().is_numeric() {
            let mut start = self.position;
            while self.current().is_numeric() {
                self.next()
            }
            let length = self.position - start;
            let text: String = self
                .text
                .chars()
                .skip(start as usize)
                .take(length as usize)
                .collect();
            let value = text.parse();
            let value = match value {
                Ok(value) => Some(Value {
                    value: Some(value),
                    bool: None,
                }),
                Err(_) => {
                    self.diagnostics
                        .report_invalid_number(TextSpan::new(start, length), Type::Number);
                    None
                }
            };
            return SyntaxToken::new(SyntaxKind::Number, start, text, value);
        } else if self.current().is_whitespace() {
            let mut start = self.position;
            while self.current().is_whitespace() {
                self.next()
            }
            let length = self.position - start;
            let text: String = self
                .text
                .chars()
                .skip(start as usize)
                .take(length as usize)
                .collect();
            return SyntaxToken::new(SyntaxKind::WhiteSpace, start, text, None);
        } else if self.current().is_alphabetic() {
            let start = self.position;
            while self.current().is_alphabetic() {
                self.next()
            }
            let length = self.position - start;
            let text: String = self
                .text
                .chars()
                .skip(start as usize)
                .take(length as usize)
                .collect();
            let kind = self.get_keyword_kind(&text);
            return SyntaxToken::new(kind, start, text, None);
        } else if self.current() == '+' {
            self.position += 1;
            return SyntaxToken::new(
                SyntaxKind::PlusToken,
                self.position,
                String::from("+"),
                None,
            );
        } else if self.current() == '-' {
            self.position += 1;
            return SyntaxToken::new(
                SyntaxKind::MinusToken,
                self.position,
                String::from("-"),
                None,
            );
        } else if self.current() == '*' {
            self.position += 1;
            return SyntaxToken::new(SyntaxKind::MulToken, self.position, String::from("*"), None);
        } else if self.current() == '/' {
            self.position += 1;
            return SyntaxToken::new(SyntaxKind::DivToken, self.position, String::from("/"), None);
        } else if self.current() == '&' && self.look_ahead() == "&" {
            self.position += 2;
            println!("here ..");
            // suspect taht we may need to increment her ...
            return SyntaxToken::new(
                SyntaxKind::AmpersandAmpersandToken,
                self.position,
                String::from("&&"),
                None,
            );
        } else if self.current() == '|' && self.look_ahead() == "|" {
            self.position += 2;
            return SyntaxToken::new(
                SyntaxKind::PipePipeToken,
                self.position,
                String::from("||"),
                None,
            );
        } else if self.current() == '=' && self.look_ahead() == "=" {
            self.position += 2;
            return SyntaxToken::new(
                SyntaxKind::EqEqToken,
                self.position,
                String::from("=="),
                None,
            );
        } else if self.current() == '!' && self.look_ahead() == "=" {
            self.position += 2;
            println!("here .......");
            return SyntaxToken::new(
                SyntaxKind::BangEqToken,
                self.position,
                String::from("!="),
                None,
            );
        } else if self.current() == '(' {
            self.position += 1;
            return SyntaxToken::new(
                SyntaxKind::OpenParenthesisToken,
                self.position,
                String::from("("),
                None,
            );
        } else if self.current() == ')' {
            self.position += 1;
            return SyntaxToken::new(
                SyntaxKind::CloseParenthesisToken,
                self.position,
                String::from(")"),
                None,
            );
        } else if self.current() == '!' {
            self.position += 1;
            return SyntaxToken::new(
                SyntaxKind::BangToken,
                self.position,
                String::from("!"),
                None,
            );
        } else if self.current() == '=' {
            self.position += 1;
            return SyntaxToken::new(SyntaxKind::Equals, self.position, String::from("="), None);
        }

        self.diagnostics
            .report_bad_character(self.position, self.current());

        return SyntaxToken::new(
            SyntaxKind::BadToken,
            self.position,
            "BAD TOKEN".to_string(),
            None,
        );
    }
}

struct Parser {
    tokens: Vec<SyntaxToken>,
    position: usize,
    pub diagnostics: DiagnosticBag,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct BinaryExpressionSyntax {
    pub left: Box<SyntaxNode>,
    operator_token: Box<SyntaxNode>,
    right: Box<SyntaxNode>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct UnaryExpressionSyntax {
    operator_token: Box<SyntaxNode>,
    operand: Box<SyntaxNode>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ParenthesizedExpression {
    pub sub_expression: Box<SyntaxNode>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum SyntaxNode {
    NameExpressionSyntax(NameExpressionSyntax),
    AssignmentExpressionSyntax(AssignmentExpressionSyntax),
    BinaryExpressionSyntax(BinaryExpressionSyntax),
    UnaryExpressionSyntax(UnaryExpressionSyntax),
    OperatorNode(SyntaxKind),
    NumberNode(Value),
    BoolNode(Value),
    ParenthesizedExpression(ParenthesizedExpression),
    None,
}

impl SyntaxNode {
    pub fn try_into_syntax_kind(&self) -> SyntaxKind {
        if let SyntaxNode::OperatorNode(op) = self {
            return op.clone();
        }

        panic!()
    }
}

pub trait Children {
    fn get_children(&self) -> Vec<SyntaxNode>;
}

impl Children for BinaryExpressionSyntax {
    fn get_children(&self) -> Vec<SyntaxNode> {
        vec![
            (*self.left).clone(),
            (*self.operator_token).clone(),
            (*self.right).clone(),
        ]
    }
}

impl Children for SyntaxKind {
    fn get_children(&self) -> Vec<SyntaxNode> {
        vec![]
    }
}

impl Children for SyntaxNode {
    fn get_children(&self) -> Vec<SyntaxNode> {
        match self {
            SyntaxNode::BinaryExpressionSyntax(bin) => vec![
                (*bin.left).clone(),
                (*bin.operator_token).clone(),
                (*bin.right).clone(),
            ],
            SyntaxNode::UnaryExpressionSyntax(un) => {
                vec![(*un.operator_token).clone(), (*un.operand).clone()]
            }
            SyntaxNode::AssignmentExpressionSyntax(assign) => vec![(*assign.expression).clone()],
            SyntaxNode::NameExpressionSyntax(name) => vec![],
            SyntaxNode::BoolNode(_) => vec![],
            SyntaxNode::NumberNode(num) => vec![],
            SyntaxNode::OperatorNode(op) => vec![],
            SyntaxNode::None => vec![],
            SyntaxNode::ParenthesizedExpression(par) => vec![*par.sub_expression.clone()],
        }
    }
}

impl fmt::Display for SyntaxNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SyntaxNode::BinaryExpressionSyntax(bin) => write!(f, "binary expression "),
            SyntaxNode::NumberNode(num) => write!(f, "number "),
            SyntaxNode::AssignmentExpressionSyntax(_) => write!(f, "assign "),
            SyntaxNode::NameExpressionSyntax(_) => write!(f, "name "),
            &SyntaxNode::BoolNode(bool) => write!(f, "bool "),
            SyntaxNode::OperatorNode(op) => write!(f, "operator "),
            SyntaxNode::UnaryExpressionSyntax(u) => write!(f, "unary expression "),
            SyntaxNode::None => write!(f, "unknown "),
            SyntaxNode::ParenthesizedExpression(par) => write!(f, "parenthesis "),
        }
    }
}

#[derive(Clone)]
struct SyntaxTree {
    diagnostics: DiagnosticBag,
    expression: Box<SyntaxNode>,
    end_of_file_token: SyntaxToken,
}

impl SyntaxTree {
    pub fn parse_tokens(text: String) -> Vec<SyntaxToken> {
        let mut lexer = Lexer::new(text);
        let mut tokens = vec![];
        while true {
            let token = lexer.next_token();
            if (token.kind == SyntaxKind::EndOfFileToekn) {
                break;
            }
            tokens.push(token)
        }
        tokens
    }
}

impl Parser {
    pub fn new(text: String) -> Self {
        let mut lexer = Lexer::new(text);

        let mut var_tokens = vec![];
        let mut diagnostics = lexer.diagnostics.clone();

        let mut token = lexer.next_token();

        while true {
            if token.kind == SyntaxKind::EndOfFileToekn {
                var_tokens.push(token);
                break;
            } else if token.kind != SyntaxKind::BadToken && token.kind != SyntaxKind::WhiteSpace {
                var_tokens.push(token);
            }
            token = lexer.next_token()
        }

        Parser {
            tokens: var_tokens,
            position: 0,
            diagnostics,
        }
    }

    pub fn pretty_print(&mut self, node: &SyntaxNode, mut indent: String, is_last: bool) {
        // ├──
        // └──
        // │

        let marker = if is_last { "└──" } else { "├──" };

        print!("{indent}");
        print!("{marker}");
        print!("{node}");
        println!("");

        indent += if is_last { "   " } else { "│  " };

        for (i, child) in node.get_children().into_iter().enumerate() {
            self.pretty_print(&child, indent.clone(), i == node.get_children().len() - 1);
        }
    }

    pub fn parse_expression(&mut self) -> SyntaxNode {
        return self.parse_assignment_expression();
    }

    pub fn parse_assignment_expression(&mut self) -> SyntaxNode {
        if self.peek(0).kind == SyntaxKind::IdentifierToken
            && self.peek(1).kind == SyntaxKind::Equals
        {
            let id = self.current();
            let identifier_token = self.next_token();
            println!("indentifier token {:?} {:?}", identifier_token, id);
            let opeator_token = self.next_token();
            let right = self.parse_assignment_expression();
            return SyntaxNode::AssignmentExpressionSyntax(AssignmentExpressionSyntax {
                identifier_token: id,
                equals_token: identifier_token,
                expression: Box::from(right),
            });
        }

        return self.parse_binary_expression(0);
    }

    pub fn parse(&mut self) -> SyntaxTree {
        let expression = self.parse_expression();
        println!("current {:?}", self.peek(1));
        let thing = self.peek(0).kind;
        if thing != SyntaxKind::EndOfFileToekn {
            println!("thing {thing:?}");
            self.diagnostics.report_unexpected_token(
                TextSpan::new(self.position as i32, 1),
                thing,
                SyntaxKind::EndOfFileToekn,
            );
        }

        return SyntaxTree {
            diagnostics: self.diagnostics.clone(),
            expression: Box::new(expression),
            end_of_file_token: self.current(),
        };
    }

    // pub fn parse_term(&mut self) -> SyntaxNode {
    //     let mut left = self.parse_factor();

    //     while self.current().kind == SyntaxKind::PlusToken
    //         || self.current().kind == SyntaxKind::MinusToken
    //     {
    //         let curr_token = self.current();

    //         let operator_token = self.next_token();
    //         let mut right = self.parse_factor();
    //         left = SyntaxNode::BinaryExpressionSyntax(BinaryExpressionSyntax {
    //             left: Box::new(left),
    //             operator_token: Box::new(SyntaxNode::OperatorNode(curr_token.kind)),
    //             right: Box::new(right),
    //         });
    //     }

    //     return left;
    // }

    pub fn parse_binary_expression(&mut self, parent_precedence: i32) -> SyntaxNode {
        let mut left;
        let unary_operator_precedence = get_unary_precedence(self.current().kind);
        if unary_operator_precedence != 0 && unary_operator_precedence >= parent_precedence {
            let op = self.current().kind;
            let operator_token = self.next_token();
            let operand = self.parse_binary_expression(unary_operator_precedence);
            left = SyntaxNode::UnaryExpressionSyntax(UnaryExpressionSyntax {
                operator_token: Box::new(SyntaxNode::OperatorNode(op)),
                operand: Box::new(operand),
            });
        } else {
            left = self.parse_primary_expression();
        }

        while true {
            let precedence = get_binary_precedence(self.current().kind);
            if precedence == 0 || precedence <= parent_precedence {
                break;
            }

            let op = self.current().kind;
            let operator_token = self.next_token();
            let right = self.parse_binary_expression(precedence);
            left = SyntaxNode::BinaryExpressionSyntax(BinaryExpressionSyntax {
                left: Box::new(left.clone()),
                operator_token: Box::new(SyntaxNode::OperatorNode(op)),
                right: Box::new(right),
            });
        }

        return left;
    }

    // recursive descent parser
    // pub fn parse_factor(&mut self) -> SyntaxNode {
    //     let mut left = self.parse_primary_expression();

    //     while self.current().kind == SyntaxKind::DivToken
    //         || self.current().kind == SyntaxKind::MulToken
    //     {
    //         let curr_token = self.current();

    //         let operator_token = self.next_token();
    //         let mut right = self.parse_primary_expression();
    //         left = SyntaxNode::BinaryExpressionSyntax(BinaryExpressionSyntax {
    //             left: Box::new(left),
    //             operator_token: Box::new(SyntaxNode::OperatorNode(curr_token.kind)),
    //             right: Box::new(right),
    //         });
    //     }

    //     return left;
    // }

    fn parse_primary_expression(&mut self) -> SyntaxNode {
        // let number_token = self.match_kind(SyntaxKind::Number);

        let node;

        if self.current().kind == SyntaxKind::OpenParenthesisToken {
            let left = self.next_token();

            let expression = self.parse_binary_expression(0);
            node = if self.current().kind == SyntaxKind::CloseParenthesisToken {
                SyntaxNode::ParenthesizedExpression(ParenthesizedExpression {
                    sub_expression: Box::new(expression),
                })
            } else {
                SyntaxNode::None
            };
        } else if self.current().kind == SyntaxKind::FalseKeyword
            || self.current().kind == SyntaxKind::TrueKeyword
        {
            let value = self.current().kind == SyntaxKind::TrueKeyword;
            // self.next_token();
            node = SyntaxNode::BoolNode(Value {
                bool: Some(value),
                value: None,
            });
        } else if self.current().kind == SyntaxKind::IdentifierToken {
            let id = self.current();
            let identifier_token = self.next_token();
            println!("here xxxx");
            return SyntaxNode::NameExpressionSyntax(NameExpressionSyntax::new(id));
        } else if self.current().kind == SyntaxKind::Number {
            node = SyntaxNode::NumberNode(self.current().value.unwrap())
        } else {
            node = SyntaxNode::None
        }
        self.position += 1;
        return node;
    }

    fn next_token(&mut self) -> SyntaxToken {
        self.position += 1;
        return self.current();
    }

    fn peek(&self, offset: usize) -> SyntaxToken {
        let index = self.position + offset;
        if index >= self.tokens.len() {
            return self.tokens[self.tokens.len() - 1].clone();
        }
        return self.tokens[index].clone();
    }

    fn current(&self) -> SyntaxToken {
        self.peek(0)
    }
}

pub struct Compilation {
    syntax_tree: SyntaxTree,
}

pub struct EvaluationResult {
    diagnostics: Vec<Diagnostic>,
    value: Value,
}

impl Compilation {
    pub fn new(syntax_tree: SyntaxTree) -> Self {
        Self { syntax_tree }
    }

    pub fn evaluate<'a>(&self, variables: &'a mut HashMap<String, Value>) -> Value {
        let mut binder = Binder::new(variables);
        let boundExpression = binder.bind_expression(*self.syntax_tree.expression.clone());

        let mut evaluator = Evaluator::new(boundExpression, variables);
        evaluator.evaluate()
    }
}

impl EvaluationResult {
    pub fn new(diagnostics: Vec<Diagnostic>, value: Value) -> Self {
        Self { diagnostics, value }
    }
}

struct Evaluator<'a> {
    root: BoundNodeKind,
    variables: &'a mut HashMap<String, Value>,
}

impl<'a> Evaluator<'a> {
    pub fn new(node: BoundNodeKind, variables: &'a mut HashMap<String, Value>) -> Self {
        return Evaluator {
            root: node,
            variables,
        };
    }

    pub fn evaluate(&mut self) -> Value {
        return self.evaluate_expression(self.root.clone());
    }

    pub fn evaluate_expression(&mut self, root: BoundNodeKind) -> Value {
        // binary expression
        // number epxresion

        match root {
            BoundNodeKind::BoundVariableExpression(bound_var) => {
                println!("here xxx ... {bound_var:?} {}", bound_var.name);
                return self.variables.get(&bound_var.name).unwrap().clone();
            }
            BoundNodeKind::BoundAssignmentExpression(assing_var) => {
                let value = self.evaluate_expression(*assing_var.bound_expression);
                println!("value {value:?} {}", assing_var.name);
                self.variables.insert(assing_var.name, value);
                println!("\n\n vairables {:?} \n", self.variables);
                return value;
            }
            BoundNodeKind::BoundLiteralExpression(number_node) => return number_node.into_value(),
            // BoundNodeKind::ParenthesizedExpression(p) => return self.evaluate_expression(*p.sub_expression.clone()),
            BoundNodeKind::BoundUnaryExpressionNode(u) => {
                let operand = self.evaluate_expression(*u.operand);
                if operand.is_num() {
                    if BoundUnaryOperatorKind::Identity == u.operator_kind.operator_kind {
                        return Value::from_num(operand.get_num());
                    } else if BoundUnaryOperatorKind::Negation == u.operator_kind.operator_kind {
                        return Value::from_num(operand.get_num() * -1);
                    } else {
                        Value::from_num(-101)
                    }
                } else if operand.is_bool() {
                    if BoundUnaryOperatorKind::LogicalNegation == u.operator_kind.operator_kind {
                        return Value::from_bool(!operand.get_bool());
                    } else {
                        Value::from_num(-102)
                    }
                } else {
                    Value::from_num(-103)
                }
            }
            BoundNodeKind::BoundBinaryExpression(b) => {
                let left = self.evaluate_expression(*b.left);
                let right = self.evaluate_expression(*b.right);
                match b.bound_binary_operator_kind.operator_kind {
                    BoundBinaryOperatorKind::Equals => return Value::from_bool(left == right),
                    BoundBinaryOperatorKind::DoesNotEqual => {
                        return Value::from_bool(left != right)
                    }
                    _ => {}
                }
                if left.is_num() && right.is_num() {
                    return match b.bound_binary_operator_kind.operator_kind {
                        BoundBinaryOperatorKind::Subtraction => {
                            Value::from_num(left.get_num() - right.get_num())
                        }
                        BoundBinaryOperatorKind::Addition => {
                            Value::from_num(left.get_num() + right.get_num())
                        }
                        BoundBinaryOperatorKind::Division => {
                            Value::from_num(left.get_num() / right.get_num())
                        }
                        BoundBinaryOperatorKind::Multiplication => {
                            Value::from_num(left.get_num() * right.get_num())
                        }
                        _ => Value::from_num(-303),
                    };
                } else if left.is_bool() && right.is_bool() {
                    return match b.bound_binary_operator_kind.operator_kind {
                        BoundBinaryOperatorKind::LogicalAnd => {
                            Value::from_bool(left.get_bool() && right.get_bool())
                        }
                        BoundBinaryOperatorKind::LogicalOr => {
                            Value::from_bool(left.get_bool() || right.get_bool())
                        }
                        _ => Value::from_num(-305),
                    };
                } else {
                    return match b.bound_binary_operator_kind.operator_kind {
                        BoundBinaryOperatorKind::Equals => Value::from_bool(left == right),
                        BoundBinaryOperatorKind::DoesNotEqual => Value::from_bool(left != right),
                        _ => Value::from_num(-308),
                    };
                }
            }
            _ => Value::from_num(-202),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameExpressionSyntax {
    pub token: SyntaxToken,
}

impl NameExpressionSyntax {
    pub fn new(identifier_token: SyntaxToken) -> Self {
        Self {
            token: identifier_token,
        }
    }

    pub fn get_children(&self) -> Vec<SyntaxNode> {
        vec![]
    }
}

/// You want expression, these expression bind to the left
/// assotiativity of the right
///
///   =
///  / \
/// a  =
///   / \
///  b   5

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssignmentExpressionSyntax {
    pub identifier_token: SyntaxToken,
    pub equals_token: SyntaxToken,
    pub expression: Box<SyntaxNode>,
}

impl AssignmentExpressionSyntax {
    pub fn new(
        identifier_token: SyntaxToken,
        equals_token: SyntaxToken,
        expression: SyntaxNode,
    ) -> Self {
        Self {
            identifier_token,
            equals_token,
            expression: Box::from(expression),
        }
    }

    pub fn get_children(&self) -> Vec<SyntaxNode> {
        vec![*self.expression.clone()]
    }
}

pub fn get_binary_precedence(kind: SyntaxKind) -> i32 {
    match kind {
        SyntaxKind::MulToken | SyntaxKind::DivToken => 5,
        SyntaxKind::PlusToken | SyntaxKind::MinusToken => 4,
        SyntaxKind::EqEqToken | SyntaxKind::BangEqToken => 3,
        SyntaxKind::AmpersandAmpersandToken => 2,
        SyntaxKind::PipePipeToken => 1,
        _ => 0,
    }
}

pub fn get_unary_precedence(kind: SyntaxKind) -> i32 {
    match kind {
        SyntaxKind::PlusToken | SyntaxKind::MinusToken => 3,
        SyntaxKind::BangToken => 1,
        _ => 0,
    }
}