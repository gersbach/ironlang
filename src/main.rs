mod binding;
mod compilation_unit;
mod diagnostics;
mod lexer_tests;
mod parser_tests;
mod source_text;

use compilation_unit::CompilationUnit;
use strum::IntoEnumIterator; // 0.17.1
use strum_macros::EnumIter; // 0.17.1
use rand::prelude::*;

use binding::{
    BoundBinaryOperatorKind, BoundBlockStatement, BoundCallExpressionNode, BoundForStatement, BoundGlobalScope, BoundIfStatement, BoundNodeKind, BoundScope, BoundUnaryOperatorKind, BoundVariableDeclarion, BoundWhileStatment, BuiltInFunctions, LabelSymbol, Lowerer, Type, VariableSymbol
};
use diagnostics::{Diagnostic, DiagnosticBag, TextSpan};
use inline_colorization::*;
use std::{
    collections::{binary_heap::Iter, HashMap}, f32::consts::E, fmt::{self, write}, hash::Hash, io::{stdin, stdout, Write}, ops::Bound, os::macos::raw::stat, sync::Condvar, vec
};

use crate::{binding::Binder, source_text::SourceText};

fn main() {
    let mut variables = HashMap::new();

    let mut compilation: Option<Compilation> = None;

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

        // let mut terminal = term::stdout().unwrap();

        // print!("> ");
        // let lines = stdin().lines();

        // if you write an ide, you care about whitespace
        // ASTs abstract syntax trees do not have whitespace or parenthesis
        // compiilers generally don't care
        // in the production enviornment, you expect a immutable represetnation
        // if syntax trees get modifieid and there are flags there is a lot of pain
        // ASTs have a slot for the type and the type checker fills that in

        let source_text = SourceText::new(line.to_string().clone());
        let mut parser = Parser::new(source_text.clone());

        let token = SyntaxTree::new(source_text);

        if let Some(comp) = compilation.clone() {
            compilation = Some(comp.clone().continue_with(token.clone()));
        } else {
            compilation = Some(Compilation::new(token.clone()))
        }

        let result = compilation.clone().unwrap().evaluate(&mut variables); // this funciton has the wrong one....

        parser.pretty_print(&&&token.root.root.clone(), "".to_string(), true);

        // bound tree / ir / annotated representation contains the types

        if parser.diagnostics.len() > 0 {
            for diagnostic in parser.diagnostics.diagnostics {

                // let line_index = compilation..get_line_index(diagnostic.span.start);
                // let line =parser.diagnostics.lines.get(line_index).unwrap();
                // let line_number = line_index + 1;
                // let character = diagnostic.span.start - line.start + 1;

                // println!("{color_red}{}{color_reset}\n Error at: {}", diagnostic.message, line_number);
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
    Colon,
    Tilda,
    Pipe,
    Comma,
    Carrot,
    AmpersandToken,
    StringToken,
    MinusToken,
    DivToken,
    MulToken,
    ForKeyword,
    WhileKeyword,
    ToKeyword,
    BadToken,
    EndOfFileToekn,
    OpenParenthesisToken,
    CloseParenthesisToken,
    IfKeyword,
    ElseKeyword,
    OpenBraceToken,
    CloseBraceToken,
    LetKeyword,
    VarKeyword,
    BinaryExpression,
    TrueKeyword,
    FalseKeyword,
    AmpersandAmpersandToken,
    PipePipeToken,
    BangToken,
    LessThan,
    LessThanEqualTo,
    GreaterThan,
    GreaterThanEqualTo,
    EqEqToken,
    BangEqToken,
    IdentifierToken,
    Equals,
    Other,
}

#[derive(Debug)]
struct Lexer {
    text: String,
    position: i32,
    kind: SyntaxKind,
    start: i32,
    value: Value,
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

#[derive(Debug, Clone, Eq, PartialEq)]
struct Value {
    value: Option<i32>,
    bool: Option<bool>,
    string: Option<String>,
    _type: Type,
}

impl Value {
    pub fn from_num(num: i32) -> Self {
        Self {
            value: Some(num),
            bool: None,
            string: None,
            _type: Type::Number
        }
    }

    pub fn from_string(string: String) -> Self {
        Self {
            value: None,
            bool: None,
            string: Some(string),
            _type: Type::String
        }
    }

    pub fn default() -> Self {
        Self {
            value: None,
            bool: None,
            string: None,
            _type: Type::Any
        }
    }

    pub fn from_bool(bool: bool) -> Self {
        Self {
            value: None,
            bool: Some(bool),
            string: None,
            _type: Type::Bool
        }
    }

    pub fn is_bool(&self) -> bool {
        self.bool.is_some()
    }

    pub fn is_string(&self) -> bool {
        self.string.is_some()
    }

    pub fn is_num(&self) -> bool {
        self.value.is_some()
    }

    pub fn get_type(&self) -> Type {
        if self.is_bool() {
            Type::Bool
        } else if self.is_num() {
            Type::Number
        } else if self.is_string() {
            Type::String
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

    pub fn get_string(&self) -> String {
        if let Some(value) = self.string.clone() {
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
    pub fn new(source_text: SourceText) -> Self {
        Lexer {
            text: source_text.text,
            position: 0,
            start: 0,
            kind: SyntaxKind::Other,
            value: Value::default(),
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
            "to" => SyntaxKind::ToKeyword,
            "if" => SyntaxKind::IfKeyword,
            "for" => SyntaxKind::ForKeyword,
            "else" => SyntaxKind::ElseKeyword,
            "Let" => SyntaxKind::LetKeyword,
            "Var" => SyntaxKind::VarKeyword,
            "while" => SyntaxKind::WhileKeyword,
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
                Ok(value) => Some(Value::from_num(value)),
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
        } else if self.current() == ':' {
            return SyntaxToken::new(SyntaxKind::Colon, self.position, String::from(":"), None);
        } else if self.current() == '*' {
            self.position += 1;
            return SyntaxToken::new(SyntaxKind::MulToken, self.position, String::from("*"), None);
        } else if self.current() == '/' {
            self.position += 1;
            return SyntaxToken::new(SyntaxKind::DivToken, self.position, String::from("/"), None);
        } else if self.current() == '&' && self.look_ahead() == "&" {
            self.position += 2;
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
            return SyntaxToken::new(
                SyntaxKind::BangEqToken,
                self.position,
                String::from("!="),
                None,
            );
        } else if self.current() == '~' {
            self.position += 1;
            return SyntaxToken::new(SyntaxKind::Tilda, self.position, String::from("~"), None);
        } else if self.current() == '|' {
            self.position += 1;
            return SyntaxToken::new(SyntaxKind::Pipe, self.position, String::from("|"), None);
        } else if self.current() == '^' {
            self.position += 1;
            return SyntaxToken::new(SyntaxKind::Carrot, self.position, String::from("^"), None);
        } else if self.current() == '&' {
            self.position += 1;
            return SyntaxToken::new(
                SyntaxKind::AmpersandToken,
                self.position,
                String::from("&"),
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
        } else if self.current() == '{' {
            self.position += 1;
            return SyntaxToken::new(
                SyntaxKind::OpenBraceToken,
                self.position,
                String::from("{"),
                None,
            );
        } else if self.current() == '}' {
            self.position += 1;
            return SyntaxToken::new(
                SyntaxKind::CloseBraceToken,
                self.position,
                String::from("}"),
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
        } else if self.current() == '>' && self.look_ahead() == "=" {
            self.position += 2;
            return SyntaxToken::new(
                SyntaxKind::GreaterThanEqualTo,
                self.position,
                String::from(">="),
                None,
            );
        } else if self.current() == '<' && self.look_ahead() == "=" {
            self.position += 2;
            return SyntaxToken::new(
                SyntaxKind::LessThanEqualTo,
                self.position,
                String::from("<="),
                None,
            );
        } else if self.current() == '>' {
            self.position += 1;
            return SyntaxToken::new(
                SyntaxKind::GreaterThan,
                self.position,
                String::from(">"),
                None,
            );
        } else if self.current() == '<' {
            self.position += 1;
            return SyntaxToken::new(SyntaxKind::LessThan, self.position, String::from("<"), None);
        } else if self.current() == ',' {
            self.position += 1;
            return SyntaxToken::new(SyntaxKind::Comma, self.position, String::from(","), None);
        }  else if self.current() == '"' {
            return self.read_string()
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

    pub fn read_string(&mut self) -> SyntaxToken {
        self.position += 1;
        let mut string = String::new();
        let mut done = false;
        while !done {
            match self.current() {
                '"' => { 
                    if self.look_ahead() == "\"" {
                        string.push(self.current());
                        self.position += 2;
                    } else {
                        self.position += 1;
                        done = true;
                    }
                     }
                '\0' | '\n' => {
                    self.diagnostics.report(TextSpan { start: 0, length: 0 }, String::from("nothing"));
                    done = true;
                    break;
                }
                _ => {
                    string.push(  self.current());
                    self.position += 1;}
            }
        }

        SyntaxToken::new(SyntaxKind::StringToken, self.position, string.clone(), Some(Value::from_string(string.clone())))
         
    }
}

struct Parser {
    tokens: Vec<SyntaxToken>,
    position: usize,
    source_text: SourceText,
    pub diagnostics: DiagnosticBag,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct BinaryExpressionSyntax {
    pub left: Box<SyntaxNode>,
    operator_token: Box<SyntaxNode>,
    right: Box<SyntaxNode>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct CallExpressionSyntax {
    identifier: SyntaxToken, 
    arguments: SeperatedSyntaxList
}

impl CallExpressionSyntax {
    pub fn new(identifier: SyntaxToken, arguments: SeperatedSyntaxList) -> Self {
        Self {
            identifier,
            arguments
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct SeperatedSyntaxList {
    seperators_and_nodes: Vec<Box<SyntaxNode>>
}

impl SeperatedSyntaxList {
    pub fn new(seperators_and_nodes: Vec<SyntaxNode>) -> Self {
        Self { seperators_and_nodes: seperators_and_nodes.iter().map(|node| Box::from(node.clone())).collect() }
    }

    pub fn get_seperator(&self, index: i32) -> SyntaxNode {
        *self.seperators_and_nodes.get((index * 2 + 1) as usize).unwrap().clone()
    }

    pub fn get_count(&self) -> i32 {
        (self.seperators_and_nodes.len() + 1 / 2 ) as i32
    }



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
    Comma,
    CallExpressionSyntax(CallExpressionSyntax),
    SeperatedSyntaxList(SeperatedSyntaxList),
    NameExpressionSyntax(NameExpressionSyntax),
    ForStatementSytnax(ForStatementSyntax),
    IfStatementSyntax(IfStatementSyntax),
    AssignmentExpressionSyntax(AssignmentExpressionSyntax),
    BinaryExpressionSyntax(BinaryExpressionSyntax),
    UnaryExpressionSyntax(UnaryExpressionSyntax),
    OperatorNode(SyntaxKind),
    ElseClause(ElseClause),
    TypeClause(TypeClause),
    LiteralNode(Value),
    VariableDeclaration(VariableDeclaration),
    StatementSyntax(StatementSyntax),
    ExpressionSyntax(ExpressionSyntax),
    BlockStatmentSyntax(BlockStatmentSyntax),
    ParenthesizedExpression(ParenthesizedExpression),
    WhileStatmentSyntax(WhileStatmentSyntax),
    ExpressionSyntaxStatement(ExpressionSyntaxStatement),
    CompilationUnitSyntax(CompilationUnitSyntax), // question does there need to be a compilation unit and a compilation unit syntax ????,
    MemberSyntax(MemberSyntax),
    GlobalStatementSyntax(GlobalStatementSyntax),
    FunctionDeclarationSyntax(FunctionDeclarationSyntax),
    None,
}

pub struct MemberSyntax {

}

pub struct GlobalStatementSyntax {

}

impl GlobalStatementSyntax {
    pub fn new() -> Self {
        Self { 

        }
    }
}

pub struct FunctionDeclarationSyntax {

}

impl FunctionDeclarationSyntax {
    pub fn new(function_keyword: SyntaxToken, 
            identifier: SyntaxToken,
            parenthesis_open_token: SyntaxToken,
            seoerated_syntax_list: SeperatedSyntaxList,
            parenthesis_closed_token: SyntaxToken,
            type_clasue: TypeClause,
        ) -> Self {
        Self { 
            
        }
    }
}


#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TypeClause {
    colon: SyntaxToken, 
    identifier: SyntaxToken
}

impl TypeClause {
    pub fn new(colon: SyntaxToken, identifier: SyntaxToken) -> Self {
        Self {
            colon,
            identifier
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ForStatementSyntax {
    keyword: SyntaxToken,
    identifier: SyntaxToken,
    equals: SyntaxToken,
    lower_bound: Box<SyntaxNode>,
    to_keyword: SyntaxToken,
    upper_bound: Box<SyntaxNode>,
    body: Box<SyntaxNode>,
}

impl ForStatementSyntax {
    pub fn new(
        keyword: SyntaxToken,
        identifier: SyntaxToken,
        equals: SyntaxToken,
        lower_bound: SyntaxNode,
        to_keyword: SyntaxToken,
        upper_bound: SyntaxNode,
        body: SyntaxNode,
    ) -> Self {
        Self {
            keyword,
            identifier,
            equals,
            lower_bound: Box::from(lower_bound),
            to_keyword,
            upper_bound: Box::from(upper_bound),
            body: Box::from(body),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WhileStatmentSyntax {
    while_keyword: SyntaxToken,
    condiiton: Box<SyntaxNode>,
    body: Box<SyntaxNode>,
}

impl WhileStatmentSyntax {
    pub fn new(while_keyword: SyntaxToken, condiiton: SyntaxNode, body: SyntaxNode) -> Self {
        Self {
            while_keyword,
            condiiton: Box::from(condiiton),
            body: Box::from(body),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct IfStatementSyntax {
    pub if_keyword: SyntaxToken,
    pub condition: Box<SyntaxNode>,
    pub then_statement: Box<SyntaxNode>,
    pub else_clause: Option<Box<SyntaxNode>>,
}

impl IfStatementSyntax {
    pub fn new(
        if_keyword: SyntaxToken,
        condition: SyntaxNode,
        then_statement: SyntaxNode,
        else_clause: Option<SyntaxNode>,
    ) -> Self {
        Self {
            if_keyword,
            condition: Box::from(condition),
            then_statement: Box::from(then_statement),
            else_clause: if else_clause.is_some() {
                Some(Box::from(else_clause.unwrap()))
            } else {
                None
            },
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ElseClause {
    else_keyword: SyntaxToken,
    else_statement: Box<SyntaxNode>,
}

impl ElseClause {
    pub fn new(else_keyword: SyntaxToken, else_statement: SyntaxNode) -> Self {
        Self {
            else_keyword,
            else_statement: Box::from(else_statement),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct VariableDeclaration {
    keyword: SyntaxToken,
    identifier: SyntaxToken,
    type_clause: Option<Box<SyntaxNode>>,
    equals: SyntaxToken,
    pub initializer: Box<SyntaxNode>,
}

impl VariableDeclaration {
    pub fn new(
        keyword: SyntaxToken,
        identifier: SyntaxToken,
        type_clause: Option<SyntaxNode>,
        equals: SyntaxToken,
        initializer: SyntaxNode,
    ) -> Self {
        VariableDeclaration {
            keyword,
            identifier,
            type_clause: if let Some(type_clause) = type_clause { Some(Box::from(type_clause)) } else { None },
            equals,
            initializer: Box::from(initializer),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CompilationUnitSyntax {
    expression: StatementSyntax,
    end_of_file_token: SyntaxToken,
}

impl CompilationUnitSyntax {
    pub fn new(expression: StatementSyntax, end_of_file_token: SyntaxToken) -> Self {
        Self {
            expression,
            end_of_file_token,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpressionSyntax {}

impl ExpressionSyntax {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StatementSyntax {}

impl StatementSyntax {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpressionSyntaxStatement {
    // a = 10
    // a + 1
    // a++
    // functionCall()
    pub expression: Box<SyntaxNode>,
}

impl ExpressionSyntaxStatement {
    pub fn new(expression: ExpressionSyntax) -> Self {
        Self {
            expression: Box::from(SyntaxNode::ExpressionSyntax(expression)),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BlockStatmentSyntax {
    open_brace: SyntaxToken,
    statements: Vec<Box<SyntaxNode>>,
    close_brace: SyntaxToken,
}

impl BlockStatmentSyntax {
    pub fn new(
        open_brace: SyntaxToken,
        statements: Vec<Box<SyntaxNode>>,
        close_brace: SyntaxToken,
    ) -> Self {
        Self {
            open_brace,
            statements,
            close_brace,
        }
    }
}

impl SyntaxNode {
    pub fn try_into_syntax_kind(&self) -> SyntaxKind {
        if let SyntaxNode::OperatorNode(op) = self {
            return op.clone();
        }

        panic!()
    }

    // this doesn't really work since children that are tokens are not being returned from the get_children function .... :(. Will need to think og a way to refactor
    pub fn get_span(&self) -> TextSpan {
        let first = self.get_children().first().unwrap().get_span();
        let last = self.get_children().last().unwrap().get_span();

        return TextSpan::new(first.length, first.length - last.length);
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
            SyntaxNode::TypeClause(_) => vec![],
            SyntaxNode::Comma  => vec![],
            SyntaxNode::CallExpressionSyntax(_) => vec![],
            SyntaxNode::SeperatedSyntaxList(_) => vec![],
            SyntaxNode::ForStatementSytnax(_) => vec![],
            SyntaxNode::WhileStatmentSyntax(_) => vec![],
            SyntaxNode::ElseClause(_) => vec![],
            SyntaxNode::BinaryExpressionSyntax(bin) => vec![
                (*bin.left).clone(),
                (*bin.operator_token).clone(),
                (*bin.right).clone(),
            ],
            SyntaxNode::IfStatementSyntax(if_statement) => vec![
                *if_statement.condition.clone(),
                *if_statement.then_statement.clone(),
            ],
            SyntaxNode::UnaryExpressionSyntax(un) => {
                vec![(*un.operator_token).clone(), (*un.operand).clone()]
            }
            SyntaxNode::VariableDeclaration(var) => vec![*var.initializer.clone()],
            SyntaxNode::StatementSyntax(stmt_syntax) => vec![],
            SyntaxNode::ExpressionSyntaxStatement(expression) => {
                vec![*expression.expression.clone()]
            }
            SyntaxNode::AssignmentExpressionSyntax(assign) => vec![(*assign.expression).clone()],
            SyntaxNode::NameExpressionSyntax(name) => vec![],
            SyntaxNode::ExpressionSyntax(exp) => vec![],
            SyntaxNode::BlockStatmentSyntax(block) => vec![],
            SyntaxNode::CompilationUnitSyntax(comp) => vec![],
            SyntaxNode::LiteralNode(num) => vec![],
            SyntaxNode::OperatorNode(op) => vec![],
            SyntaxNode::None => vec![],
            SyntaxNode::ParenthesizedExpression(par) => vec![*par.sub_expression.clone()],
        }
    }
}

impl fmt::Display for SyntaxNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.clone() {
            SyntaxNode::TypeClause(_) => write!(f, "type cluase"),
            SyntaxNode::Comma => write!(f, "comma"),
            SyntaxNode::CallExpressionSyntax(call) => write!(f, "call expression syntax"),
            SyntaxNode::SeperatedSyntaxList(_) => write!(f, "seperated syntax list"),
            SyntaxNode::ForStatementSytnax(_) => write!(f, "for"),
            SyntaxNode::WhileStatmentSyntax(_) => write!(f, "while "),
            SyntaxNode::ElseClause(_) => write!(f, "else "),
            SyntaxNode::IfStatementSyntax(_) => write!(f, "if statment "),
            SyntaxNode::VariableDeclaration(com) => write!(f, "variable declaration "),
            SyntaxNode::CompilationUnitSyntax(com) => write!(f, "compilation unit syntax "),
            SyntaxNode::StatementSyntax(statemnt_syntax) => write!(f, "statment syntaxx "),
            SyntaxNode::ExpressionSyntaxStatement(expression) => {
                write!(f, "expression statement syntax")
            }
            SyntaxNode::BlockStatmentSyntax(block) => write!(f, "block statement syntax "),
            SyntaxNode::ExpressionSyntax(ex) => write!(f, "expression statemnt syntax "),
            SyntaxNode::BinaryExpressionSyntax(bin) => write!(f, "binary expression "),
            SyntaxNode::LiteralNode(num) => write!(f, "number "),
            SyntaxNode::AssignmentExpressionSyntax(_) => write!(f, "assign "),
            SyntaxNode::NameExpressionSyntax(_) => write!(f, "name "),
            SyntaxNode::LiteralNode(bool) => write!(f, "bool "),
            SyntaxNode::OperatorNode(op) => write!(f, "operator "),
            SyntaxNode::UnaryExpressionSyntax(u) => write!(f, "unary expression "),
            SyntaxNode::None => write!(f, "unknown "),
            SyntaxNode::ParenthesizedExpression(par) => write!(f, "parenthesis "),
        }
    }
}

#[derive(Clone, Debug)]
struct SyntaxTree {
    text: SourceText,
    diagnostics: DiagnosticBag,
    root: CompilationUnit,
}

impl SyntaxTree {
    pub fn parse_tokens(text: String, source_text: SourceText) -> Vec<SyntaxToken> {
        let mut lexer = Lexer::new(source_text);
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

    pub fn new(source_text: SourceText) -> Self {
        let mut parser = Parser::new(source_text.clone());
        let root = parser.parse_compilation_unit(source_text.clone());
        let diagnostics = parser.diagnostics;

        Self {
            text: source_text,
            diagnostics: diagnostics,
            root: root,
        }
    }
}

impl Parser {
    pub fn new(source_text: SourceText) -> Self {
        let mut lexer = Lexer::new(source_text.clone());

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
            source_text: source_text.clone(),
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

    pub fn parse_statement(&mut self) -> SyntaxNode {
        if self.current().kind == SyntaxKind::OpenBraceToken {
            return self.parse_block_statement();
        }
        if self.current().kind == SyntaxKind::LetKeyword
            || self.current().kind == SyntaxKind::VarKeyword
        {
            return self.parse_variable_declartion();
        }
        if self.current().kind == SyntaxKind::IfKeyword {
            return self.parse_if_statement();
        }
        if self.current().kind == SyntaxKind::WhileKeyword {
            return self.parse_while_statement();
        }
        if self.current().kind == SyntaxKind::ForKeyword {
            return self.parse_for_statement();
        }
        return self.parse_expression_statement();
    }

    pub fn parse_for_statement(&mut self) -> SyntaxNode {
        let keyword = self.current();
        let identifier = self.next_token();
        let equals = self.next_token();
        self.next_token();
        let lower_bound = self.parse_expression();
        let to_keyword = self.next_token();
        let upper_bound = self.parse_expression();
        let body = self.parse_statement();
        return SyntaxNode::ForStatementSytnax(ForStatementSyntax::new(
            keyword,
            identifier,
            equals,
            lower_bound,
            to_keyword,
            upper_bound,
            body,
        ));
    }

    pub fn parse_while_statement(&mut self) -> SyntaxNode {
        let keyword = self.next_token();
        let condition = self.parse_expression();
        let body = self.parse_statement();
        return SyntaxNode::WhileStatmentSyntax(WhileStatmentSyntax::new(keyword, condition, body));
    }

    pub fn parse_if_statement(&mut self) -> SyntaxNode {
        let keyword = self.current();
        self.next_token();
        let condition = self.parse_expression();
        let statement = self.parse_statement();
        let else_clause = self.parse_optional_else_clause();
        SyntaxNode::IfStatementSyntax(IfStatementSyntax::new(
            keyword,
            condition,
            statement,
            else_clause,
        ))
    }

    pub fn parse_optional_else_clause(&mut self) -> Option<SyntaxNode> {
        if self.current().kind != SyntaxKind::ElseKeyword {
            return None;
        }
        let keyword = self.next_token();
        let statement = self.parse_statement();
        Some(SyntaxNode::ElseClause(ElseClause::new(keyword, statement)))
    }

    pub fn parse_variable_declartion(&mut self) -> SyntaxNode {
        let expected: SyntaxKind = if self.current().kind == SyntaxKind::LetKeyword {
            SyntaxKind::LetKeyword
        } else {
            SyntaxKind::VarKeyword
        };
        let keyword = self.current();
        let identifier = self.next_token();
        let type_clause = self.parse_optional_type_clause();
        let equals = self.next_token();
        self.next_token();
        let initializer = self.parse_expression();
        return SyntaxNode::VariableDeclaration(VariableDeclaration::new(
            keyword,
            identifier,
            type_clause,
            equals,
            initializer,
        ));
    }

    pub fn parse_optional_type_clause(&mut self) -> Option<SyntaxNode> {
        if self.current().kind != SyntaxKind::Colon {
            return None
        }

        return Some(self.parse_type_clause());
        
    }

    pub fn parse_type_clause(&mut self) -> SyntaxNode {
        let colon_token = self.current();
        let identifier_token = self.next_token();

        SyntaxNode::TypeClause(TypeClause::new(colon_token, identifier_token))
    }

    pub fn parse_expression_statement(&mut self) -> SyntaxNode {
        let expression = self.parse_expression();
        return expression;
    }

    pub fn parse_block_statement(&mut self) -> SyntaxNode {
        let open_brace = self.next_token();

        let mut statements = vec![];

        let mut start_token = self.current();

        while self.current().kind != SyntaxKind::EndOfFileToekn
            && self.current().kind != SyntaxKind::CloseBraceToken
        {
            let statement = self.parse_statement();
            statements.push(Box::from(statement));

            /// if parse_statement did not consume any tokens,
            /// let's skip the current token and continue we do not
            /// need to report and error because we'll aready tried
            /// to parse an expression statment
            if self.current() == start_token {
                self.next_token();
            }

            start_token = self.current();
        }

        let close_brace = self.next_token();
        return SyntaxNode::BlockStatmentSyntax(BlockStatmentSyntax {
            open_brace: open_brace,
            statements: statements,
            close_brace,
        });
    }

    pub fn parse_assignment_expression(&mut self) -> SyntaxNode {
        if self.peek(0).kind == SyntaxKind::IdentifierToken
            && self.peek(1).kind == SyntaxKind::Equals
        {
            let id = self.current();
            let identifier_token = self.next_token();
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

    pub fn parse_compilation_unit(&mut self, source_text: SourceText) -> CompilationUnit {
        let expression = self.parse_statement();
        let thing = self.peek(0).kind;
        if thing != SyntaxKind::EndOfFileToekn {
            self.diagnostics.report_unexpected_token(
                TextSpan::new(self.position as i32, 1),
                thing.clone(),
                SyntaxKind::EndOfFileToekn,
            );
        }

        CompilationUnit::new(expression, thing)
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

    fn parse_call_expression(&mut self) -> SyntaxNode {
        let iden = self.current();
        let identifier = self.next_token();
        let open_parenthese_token = self.next_token();
        let arguments = self.parse_arguments();
        let closed_parenthese_token = self.next_token();
        return SyntaxNode::CallExpressionSyntax(CallExpressionSyntax::new(iden, arguments))
    }

    fn parse_arguments(&mut self) -> SeperatedSyntaxList {
        let mut nodes_and_seperators = vec![];
        while self.current().kind != SyntaxKind::CloseParenthesisToken && self.current().kind != SyntaxKind::EndOfFileToekn {
            let expression = self.parse_expression();
            nodes_and_seperators.push(expression);

            if self.current().kind != SyntaxKind::CloseParenthesisToken {
                let comma = self.next_token();
                nodes_and_seperators.push(SyntaxNode::Comma)

            }
         }

        SeperatedSyntaxList::new(nodes_and_seperators)
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
            node = SyntaxNode::LiteralNode(Value::from_bool(value));
        } else if self.current().kind == SyntaxKind::IdentifierToken {
            let id = self.current();
            let next_token = self.peek(1);
            if next_token.kind == SyntaxKind::OpenParenthesisToken {
                return self.parse_call_expression();
            }
            self.next_token();

            return SyntaxNode::NameExpressionSyntax(NameExpressionSyntax::new(id));
        } else if self.current().kind == SyntaxKind::Number {
            node = SyntaxNode::LiteralNode(self.current().value.unwrap())
        } else if self.current().kind == SyntaxKind::StringToken {
            node = SyntaxNode::LiteralNode(self.current().value.unwrap())
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

#[derive(Clone, Debug)]
pub struct Compilation {
    syntax_tree: SyntaxTree,
    previous: Option<Box<Compilation>>,
    global_scope: Option<BoundGlobalScope>,
}

pub struct EvaluationResult {
    diagnostics: Vec<Diagnostic>,
    value: Value,
}

impl Compilation {
    pub fn new(syntax_tree: SyntaxTree) -> Self {
        Self {
            syntax_tree,
            global_scope: None,
            previous: None,
        }
    }

    pub fn new_with_prev(previous: Compilation, syntax_tree: SyntaxTree) -> Self {
        Self {
            syntax_tree,
            global_scope: None,
            previous: None,
        }
    }

    pub fn get_global_scope(&mut self) -> BoundGlobalScope {
        if let Some(global_scope) = self.global_scope.clone() {
            return global_scope;
        }
        let previous: Option<BoundGlobalScope> = if let Some(prev) = self.previous.clone() {
            prev.global_scope
        } else {
            None
        };
        let global_scope = Binder::bind_global_scope(previous, self.syntax_tree.root.clone());
        self.global_scope = Some(global_scope.clone());
        global_scope
    }

    pub fn continue_with(&mut self, syntax_tree: SyntaxTree) -> Self {
        self.get_global_scope();
        return Self {
            syntax_tree,
            previous: Some(Box::from(self.clone())),
            global_scope: None,
        };
    }

    pub fn evaluate<'a>(&mut self, variables: &'a mut HashMap<String, Value>) -> Value {
        let mut evaluator = Evaluator::new(
            BoundBlockStatement::new(vec![self.get_global_scope().expression]),
            variables,
        );
        evaluator.evaluate()
    }
}

impl EvaluationResult {
    pub fn new(diagnostics: Vec<Diagnostic>, value: Value) -> Self {
        Self { diagnostics, value }
    }
}

struct Evaluator<'a> {
    root: BoundBlockStatement,
    variables: &'a mut HashMap<String, Value>,
    statment_index: usize,
    label_to_index: HashMap<LabelSymbol, usize>,
}

impl<'a> Evaluator<'a> {
    pub fn new(node: BoundBlockStatement, variables: &'a mut HashMap<String, Value>) -> Self {
        return Evaluator {
            root: node,
            variables,
            statment_index: 0,
            label_to_index: HashMap::new(),
        };
    }

    pub fn evaluate(&mut self) -> Value {
        let statement = self.get_statement();

        for (i, s) in statement.statments.clone().iter().enumerate() {
            if let BoundNodeKind::BoundLabelStatement(label) = *s.clone() {
                println!("label statemnt ---> {label:?} {}", i + 1);
                self.label_to_index.insert(label.symbol, i + 1);
            }
        }

        let mut last_value = Value::default();

        // for (i, s) in statement.statments.clone().iter().enumerate() {
        //     last_value = self.evaluate_expression(*s.clone());
        // }

        while self.statment_index < statement.statments.len() {
            let statmentx = statement.statments.get(self.statment_index).unwrap();
            last_value = match *statmentx.clone() {
                BoundNodeKind::BoundConditionalGoToStatment(conditional_goto_statment) => {
                    let condition =
                        self.evaluate_expression(*conditional_goto_statment.condition.clone());
                    if condition.bool == Some(true) && !conditional_goto_statment.jump_if_false
                        || condition.bool == Some(false) && conditional_goto_statment.jump_if_false
                    {
                        self.statment_index = *self
                            .label_to_index
                            .get(&conditional_goto_statment.symbol)
                            .clone()
                            .unwrap();
                    } else {
                        self.statment_index += 1;
                    }

                    Value::default()
                }
                BoundNodeKind::BoundLabelStatement(label) => {
                    self.statment_index += 1;
                    Value::default()
                }
                BoundNodeKind::BoundGoToStatment(goto) => {
                    self.statment_index = *self.label_to_index.get(&goto.label).clone().unwrap();
                    Value::default()
                }
                BoundNodeKind::BoundVariableDeclaration(boundvar) => {
                    let expression = self.evaluate_expression(*boundvar.initializer.clone());
                    self.variables.insert(boundvar.variable.name, expression.clone());
                    self.statment_index += 1;
                    expression
                }
                BoundNodeKind::BoundExpressionStatement(expresison) => {
                    self.statment_index += 1;
                    self.evaluate_expression(*expresison.expression)
                }
                _ => {
                    self.statment_index += 1;
                    self.evaluate_expression(*statmentx.clone())
                }
            }
        }

        // return self.evaluate_expression(self.root.clone());

        return last_value;
    }

    pub fn get_statement(&mut self) -> BoundBlockStatement {
        let result = self.root.clone();
        return Lowerer::lower(result);
    }

    pub fn evaluate_call_expression(&mut self, node: BoundCallExpressionNode) -> Value {
        let built_in_functions = BuiltInFunctions::new();
        println!("node function {}", node.function.name);
        if node.function == built_in_functions.print_function {
            let message = self.evaluate_expression(*node.arguments.get(0).unwrap().clone());
            println!("RET {}", message.get_string());
        } else if node.function.name == built_in_functions.random_function.name {
            let max = self.evaluate_expression(*node.arguments.get(0).unwrap().clone());
            let mut nums: Vec<i32> = (1..max.get_num()).collect();
            let mut rng = rand::thread_rng();
            nums.shuffle(&mut rng);
            return Value::from_num(*nums.get(0).unwrap());
        
        } else {
            panic!()
        }

        Value::default()
    }


    pub fn evaluate_if_statmeent(&mut self, node: BoundIfStatement) -> Value {
        let condition = self.evaluate_expression(*node.condition);
        if condition.bool == Some(true) {
            self.evaluate_expression(*node.then_statment);
        } else if let Some(else_sttmt) = node.else_statement {
            self.evaluate_expression(*else_sttmt);
        }
        // error here
        Value::default()
    }

    pub fn evaluate_while_statement(&mut self, while_statment: BoundWhileStatment) -> Value {
        let mut lastn = Value::default();
        while Some(true)
            == self
                .evaluate_expression(*while_statment.condition.clone())
                .bool
        {
            lastn = self.evaluate_expression(*while_statment.body.clone());
        }
        lastn
    }

    pub fn evalaute_for_statement(&mut self, for_statement: BoundForStatement) -> Value {
        let lower_bound = self.evaluate_expression(*for_statement.lower_bound);
        let upper_bound = self.evaluate_expression(*for_statement.upper_bound);

        let mut last_value = Value::default();

        for i in lower_bound.get_num()..upper_bound.get_num() {
            self.variables
                .insert(for_statement.variable.name.clone(), Value::from_num(i));
            println!("self variables {:?}", self.variables);
            last_value = self.evaluate_expression(*for_statement.body.clone());
        }

        return last_value;
    }

    pub fn evaluate_expression(&mut self, root: BoundNodeKind) -> Value {
        // binary expression
        // number epxresion
        match root.clone() {
            // BoundNodeKind::BoundForStatment(for_statement) => {
            //     self.evalaute_for_statement(for_statement)
            // }
            // BoundNodeKind::BoundWhileStatment(while_statement) => {
            //     self.evaluate_while_statement(while_statement)
            // }
            // BoundNodeKind::BoundIfStatement(if_statement) => {
            //     print!("why ami i here....");
            //     self.evaluate_if_statmeent(if_statement)
            // }
            BoundNodeKind::BoundVariableExpression(bound_var) => {
                let vart = self.variables.get(&bound_var.name).unwrap().clone();
                return vart;
            }
            BoundNodeKind::BoundConversionExpression(conversion_expression) => {
                let value = self.evaluate_expression(*conversion_expression.expression);
                if conversion_expression._type == Type::Bool {
                    return match value.get_string().as_str() {
                        "true" => Value::from_bool(true),
                        _ => Value::from_bool(false),                     
                    }                 
                } else if conversion_expression._type == Type::Number {
                    let my_int = value.get_string().parse::<i32>().unwrap();
                    return Value::from_num(my_int)
                } else if conversion_expression._type == Type::String {
                    return Value::from_string(value.get_num().to_string())
                } 

                panic!()
            }
            BoundNodeKind::BoundAssignmentExpression(assing_var) => {
                let value = self.evaluate_expression(*assing_var.bound_expression).clone();
                self.variables.insert(assing_var.name, value.clone());
                return value;
            }
            BoundNodeKind::BoundCallExpresisonNode(call) => {
                self.evaluate_call_expression(call)
                
            }
            // BoundNodeKind::BoundBlockStatement(bb_statemetn) => {
            //     let evaluations = bb_statemetn
            //         .statments
            //         .iter()
            //         .map(|statemtent| self.evaluate_expression(*statemtent.clone()));

            //     return evaluations.last().unwrap();
            // }
            BoundNodeKind::BoundLiteralExpression(number_node) => return number_node.into_value(),
            // BoundNodeKind::ParenthesizedExpression(p) => return self.evaluate_expression(*p.sub_expression.clone()),
            BoundNodeKind::BoundUnaryExpressionNode(u) => {
                let operand = self.evaluate_expression(*u.operand);
                if operand.is_num() {
                    if BoundUnaryOperatorKind::Identity == u.operator_kind.operator_kind {
                        return Value::from_num(operand.get_num());
                    } else if BoundUnaryOperatorKind::Negation == u.operator_kind.operator_kind {
                        return Value::from_num(operand.get_num() * -1);
                    } else if BoundUnaryOperatorKind::OnesCompliment
                        == u.operator_kind.operator_kind
                    {
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
                        BoundBinaryOperatorKind::BitwiseAnd => {
                            Value::from_num(left.get_num() & right.get_num())
                        }
                        BoundBinaryOperatorKind::BitwiseOr => {
                            Value::from_num(left.get_num() | right.get_num())
                        }
                        BoundBinaryOperatorKind::BitwiseXor => {
                            Value::from_num(left.get_num() ^ right.get_num())
                        }
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
                        BoundBinaryOperatorKind::GreaterThan => {
                            Value::from_bool(left.get_num() > right.get_num())
                        }
                        BoundBinaryOperatorKind::GreaterThanEqualTo => {
                            Value::from_bool(left.get_num() >= right.get_num())
                        }
                        BoundBinaryOperatorKind::LessThan => {
                            Value::from_bool(left.get_num() < right.get_num())
                        }
                        BoundBinaryOperatorKind::LessThanEqualTo => {
                            Value::from_bool(left.get_num() <= right.get_num())
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
                        BoundBinaryOperatorKind::BitwiseAnd => {
                            Value::from_bool(left.get_bool() & right.get_bool())
                        }
                        BoundBinaryOperatorKind::BitwiseOr => {
                            Value::from_bool(left.get_bool() | right.get_bool())
                        }
                        BoundBinaryOperatorKind::BitwiseXor => {
                            Value::from_bool(left.get_bool() ^ right.get_bool())
                        }
                        _ => Value::from_num(-305),
                    };
                } else if left.is_string() && right.is_string() {
                    return match b.bound_binary_operator_kind.operator_kind {
                        BoundBinaryOperatorKind::Addition => {
                            Value::from_string(left.get_string() + &right.get_string())
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
            BoundNodeKind::Unkown => Value::from_num(-4893),
            _ => panic!("mishandled statment {root:?}"),
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
        SyntaxKind::EqEqToken
        | SyntaxKind::BangEqToken
        | SyntaxKind::LessThan
        | SyntaxKind::LessThanEqualTo
        | SyntaxKind::GreaterThanEqualTo
        | SyntaxKind::GreaterThan => 3,
        SyntaxKind::AmpersandAmpersandToken => 2,
        SyntaxKind::PipePipeToken
        | SyntaxKind::Pipe
        | SyntaxKind::Tilda
        | SyntaxKind::Carrot
        | SyntaxKind::AmpersandToken => 1,
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


enum SymbolKind {
    TypeSymbol(TypeSymbol),
    VariableSymbol(VariableSymbol),
}

pub struct TypeSymbol {

}

impl TypeSymbol {
    pub fn new(name: String) -> Self {
        Self { }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ConversionType {
    Identity,
    Explicit,
    None
}

pub struct Conversion {
    exists: bool, 
    is_identity: bool, 
    is_implicit: bool,
    is_explicit: bool
}

impl Conversion {

    pub fn new(exists: bool, is_identity: bool, is_implicit: bool) -> Self {
        Self {
            exists,
            is_identity,
            is_implicit,
            is_explicit: !is_implicit && exists
        }
    }

    pub fn classifiy(from: Type, to: Type) -> ConversionType {
        // identity conversion if the types are the same
        if from == to {
            return ConversionType::Identity
        } else if from == Type::Number || from == Type::Bool {
            if to == Type::String {
                return ConversionType::Explicit
            }
        }

        ConversionType::None
    }
}
