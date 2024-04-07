use std::{fmt::{self, format}, path::Display, vec};

use crate::{binding::Type, SyntaxKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextSpan {
    start: i32,
    length: i32
}

impl TextSpan {
    pub fn new(start: i32, length: i32) -> Self {
        Self { start, length }
    }

    pub fn get_length(&self) -> i32 {
        self.start + self.length
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    span: TextSpan,
    message: String
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Diagnostic {
    pub fn new(span: TextSpan, message: String) -> Self {
        Self { span, message }
    }
}

#[derive(Debug, Clone)]
pub struct DiagnosticBag {
    pub diagnostics: Vec<Diagnostic>
}

impl DiagnosticBag {

    pub fn new() -> Self {
        DiagnosticBag {diagnostics: vec![] }
    }

    pub fn report_unexpected_token(&mut self, span: TextSpan, kind: SyntaxKind, expected_kind: SyntaxKind) {
        let message = format!("Unexpected token <{}>, expected <{}>", "kind", "expected_kind"); // todo fix this busy work
        self.report(span, message)
    }

    pub fn report_bad_character(&mut self, position: i32, character: char) {
        let span = TextSpan::new(position, 1);
        let message = format!("Bad character input: {character}");
        self.report(span, message);
    }

    pub fn report_undefined_unary_operator(&mut self, span: TextSpan, type_: Type) {
        let message = format!("Unary operator is not defined for type");
        self.report(span, message);
    }

    pub fn report_unknown_variable(&mut self, span: TextSpan) {
        let message = format!("Variable name does not exists");
        self.report(span, message);
    }

    pub fn report_invalid_number(&mut self, span: TextSpan, type_: Type) {
        let message = format!("The number isn't valid");
        self.report(span, message);
    }

    pub fn report(&mut self, span: TextSpan, message: String) {
        let diag = Diagnostic::new(span, message);
        self.diagnostics.push(diag)
    }

    pub fn len(&self) -> i32 {
        self.diagnostics.len() as i32
    }

}