use crate::{SyntaxKind, SyntaxNode, SyntaxToken};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CompilationUnit {
    pub root: Box<SyntaxNode>,
    pub end_of_file: Box<SyntaxKind>,
}

impl CompilationUnit {
    pub fn new(root: SyntaxNode, end_of_file: SyntaxKind) -> Self {
        Self {
            root: Box::from(root),
            end_of_file: Box::from(end_of_file),
        }
    }
}
