use core::panic;
use std::{
    any::Any, borrow::Borrow, collections::{HashMap, VecDeque}, fmt, hash::Hash, ops::Bound, os::macos::raw::stat
};
use strum::IntoEnumIterator; // 0.17.1
use strum_macros::EnumIter; // 0.17.1

use crate::{
    compilation_unit::CompilationUnit, diagnostics::{self, Diagnostic, DiagnosticBag, TextSpan}, AssignmentExpressionSyntax, BinaryExpressionSyntax, BlockStatmentSyntax, CallExpressionSyntax, Conversion, ConversionType, ExpressionSyntax, ExpressionSyntaxStatement, ForStatementSyntax, IfStatementSyntax, NameExpressionSyntax, ParenthesizedExpression, StatementSyntax, SyntaxKind, SyntaxNode, SyntaxToken, TypeClause, UnaryExpressionSyntax, Value, VariableDeclaration, WhileStatmentSyntax
};

#[derive(Clone, Debug)]
pub enum BoundNodeKind {
    BoundConversionExpression(BoundConversionExpression),
    BoundCallExpresisonNode(BoundCallExpressionNode),
    BoundUnaryExpressionNode(BoundUnaryExpressionNode),
    BoundWhileStatment(BoundWhileStatment),
    BoundForStatment(BoundForStatement),
    BoundGoToStatment(BoundGoToStatment),
    BoundVariableDeclaration(BoundVariableDeclarion),
    BoundLiteralExpression(BoundLiteralExpression),
    BoundBinaryExpression(BoundBinaryExpression),
    BoundAssignmentExpression(BoundAssignmentExpression),
    BoundVariableExpression(BoundVariableExpression),
    BoundExpressionStatement(BoundExpressionStatement),
    BoundBlockStatement(BoundBlockStatement),
    BoundIfStatement(BoundIfStatement),
    BoundLabelStatement(BoundLabelStatement),
    BoundConditionalGoToStatment(BoundConditionalGoToStatment),
    BoundErrorExpression(BoundErrorExpression),
    Unkown,
}

#[derive(Clone, Debug)]
pub struct BoundConversionExpression {
    pub _type: Type, 
    pub expression: Box<BoundNodeKind> 
}

impl BoundConversionExpression {
    pub fn new(_type: Type, expression: BoundNodeKind) -> Self {
        Self {
            _type,
            expression: Box::from(expression),
        }
        }
}

#[derive(Clone, Debug)]
pub struct BoundErrorExpression {

}

impl BoundErrorExpression {
    pub fn new() -> Self {
        Self { }
    }
}

#[derive(Clone, Debug)]
pub struct BoundCallExpressionNode {
    pub function: FunctionSymbol,
    pub arguments: Vec<Box<BoundNodeKind>>
}

impl BoundCallExpressionNode {
    pub fn new(function: FunctionSymbol,
        arguments: Vec<BoundNodeKind>) -> Self {
        Self { function, arguments: arguments.iter().map(|a| Box::from(a.clone())).collect() }
    }
}

// low level represenation for an if
#[derive(Clone, Debug)]
pub struct BoundConditionalGoToStatment {
    pub symbol: LabelSymbol,
    pub condition: Box<BoundNodeKind>,
    pub jump_if_false: bool,
}

impl BoundConditionalGoToStatment {
    pub fn new(symbol: LabelSymbol, condition: BoundNodeKind, jump_if_false: bool) -> Self {
        Self {
            symbol,
            condition: Box::from(condition),
            jump_if_false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundLabelStatement {
    pub symbol: LabelSymbol,
}

impl BoundLabelStatement {
    pub fn new(symbol: LabelSymbol) -> Self {
        Self { symbol }
    }
}

#[derive(Clone, Debug)]
pub struct BoundForStatement {
    pub variable: VariableSymbol,
    pub lower_bound: Box<BoundNodeKind>,
    pub upper_bound: Box<BoundNodeKind>,
    pub body: Box<BoundNodeKind>,
}

impl BoundForStatement {
    pub fn new(
        variable: VariableSymbol,
        lower_bound: BoundNodeKind,
        upper_bound: BoundNodeKind,
        body: BoundNodeKind,
    ) -> Self {
        Self {
            variable,
            lower_bound: Box::from(lower_bound),
            upper_bound: Box::from(upper_bound),
            body: Box::from(body),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundWhileStatment {
    pub condition: Box<BoundNodeKind>,
    pub body: Box<BoundNodeKind>,
}

impl BoundWhileStatment {
    pub fn new(condition: BoundNodeKind, body: BoundNodeKind) -> Self {
        Self {
            condition: Box::from(condition),
            body: Box::from(body),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundIfStatement {
    pub condition: Box<BoundNodeKind>,
    pub then_statment: Box<BoundNodeKind>,
    pub else_statement: Option<Box<BoundNodeKind>>,
}

impl BoundIfStatement {
    pub fn new(
        condition: BoundNodeKind,
        then_statement: BoundNodeKind,
        else_statement: Option<BoundNodeKind>,
    ) -> Self {
        Self {
            condition: Box::from(condition),
            then_statment: Box::from(then_statement),
            else_statement: if let Some(else_statment) = else_statement {
                Some(Box::from(else_statment))
            } else {
                None
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundVariableDeclarion {
    pub variable: VariableSymbol,
    pub initializer: Box<BoundNodeKind>,
}

impl BoundVariableDeclarion {
    pub fn new(variable: VariableSymbol, initializer: BoundNodeKind) -> Self {
        Self {
            variable,
            initializer: Box::from(initializer),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundBlockStatement {
    pub statments: Vec<Box<BoundNodeKind>>,
}

impl BoundBlockStatement {
    pub fn new(statments: Vec<BoundNodeKind>) -> Self {
        Self {
            statments: statments
                .iter()
                .map(|bound| Box::from(bound.clone()))
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundExpressionStatement {
    pub expression: Box<BoundNodeKind>,
}

impl BoundExpressionStatement {
    pub fn new(expression: BoundNodeKind) -> Self {
        Self {
            expression: Box::from(expression),
        }
    }
}

#[derive(Clone, Debug)]
struct BoundExpression {}

impl BoundExpression {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Clone, Debug)]
pub struct BoundVariableExpression {
    pub name: String,
    pub _type: Type,
}

impl BoundVariableExpression {
    pub fn new(name: String, _type: Type) -> Self {
        Self { name, _type }
    }
}

#[derive(Clone, Debug)]
pub struct BoundAssignmentExpression {
    pub name: String,
    pub bound_expression: Box<BoundNodeKind>,
}

impl BoundAssignmentExpression {}

impl BoundNodeKind {
    pub fn get_type(&self) -> Type {
        if let BoundNodeKind::BoundLiteralExpression(lit) = self {
            if lit.is_bool() {
                return Type::Bool;
            } else {
                return Type::Number;
            }
        } else if let BoundNodeKind::BoundBinaryExpression(bin) = self {
            return bin.bound_binary_operator_kind.result_type.clone();
        } else if let BoundNodeKind::BoundVariableExpression(var) = self {
            return var._type.clone();
        } else if let BoundNodeKind::BoundUnaryExpressionNode(un) = self {
            return un.operator_kind.result_type.clone();
        }
        // panic!()
        Type::Unkown
    }
}

#[derive(Clone, Debug)]
pub struct BoundLiteralExpression {
    pub value_num: Option<i32>,
    pub value_bool: Option<bool>,
    pub value_string: Option<String>
}

impl BoundLiteralExpression {
    pub fn is_bool(&self) -> bool {
        self.value_bool.is_some()
    }

    pub fn is_num(&self) -> bool {
        self.value_num.is_some()
    }

    pub fn is_string(&self) -> bool {
        self.value_string.is_some()
    }

    pub fn is_void(&self) -> bool {
        !self.is_bool() && !self.is_num() && !self.is_string()
    }
}

impl BoundLiteralExpression {
    pub fn into_value(&self) -> Value {
        Value {
            value: self.value_num,
            bool: self.value_bool,
            string: self.value_string.clone(),
            _type: {
                if self.value_bool.is_some() {
                    Type::Bool
                } else if self.value_num.is_some() {
                    Type::Number
                } else if self.value_string.is_some() {
                    Type::String
                } else {
                    Type::Unkown
                }
            }
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum BoundUnaryOperatorKind {
    Identity,
    Negation,
    OnesCompliment,
    LogicalNegation,
}

#[derive(Clone, EnumIter, Debug)]
pub enum BoundBinaryOperatorKind {
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Addition,
    Subtraction,
    Multiplication,
    Division,
    LogicalAnd,
    LessThan,
    GreaterThan,
    LessThanEqualTo,
    GreaterThanEqualTo,
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

#[derive(PartialEq, Clone, Debug, PartialOrd, Eq, Hash)]
pub enum Type {
    Number,
    Bool,
    Any,
    Error,
    Void,
    String,
    Unkown,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        return match self {
            Type::Error => write!(f, "error"),
            Type::String => write!(f, "stirng"),
            Type::Number => write!(f, "number"),
            Type::Bool => write!(f, "bool"),
            Type::Void => write!(f, "void"),
            Type::Any => write!(f, "any"),
            Type::Unkown => write!(f, "unkonw"),
        };
    }
}

pub struct BuiltInFunctions {
    pub functions: Vec<FunctionSymbol>,
    pub print_function: FunctionSymbol,
    pub input_function: FunctionSymbol,
    pub random_function: FunctionSymbol

}

impl BuiltInFunctions {
    pub fn new() -> Self {
        Self {
            functions: vec![
                FunctionSymbol::new("print".to_string(), vec![ParameterSymbol::new("text".to_string(), Type::String)], Type::Void),
                FunctionSymbol::new("input".to_string(), vec![], Type::String),
                FunctionSymbol::new("rnd".to_string(), vec![ParameterSymbol::new("min".to_string(), Type::Number)], Type::Number),

            ],
            print_function: FunctionSymbol::new("print".to_string(), vec![ParameterSymbol::new("text".to_string(), Type::String)], Type::Void),
            input_function: FunctionSymbol::new("input".to_string(), vec![], Type::String),
            random_function: FunctionSymbol::new("rnd".to_string(), vec![ParameterSymbol::new("num".to_string(), Type::Number)], Type::Number),

        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundUnaryExpressionNode {
    pub operator_kind: Box<BoundUnaryOperator>,
    pub operand: Box<BoundNodeKind>,
}

#[derive(Clone, Debug)]
pub struct BoundBinaryExpression {
    pub left: Box<BoundNodeKind>,
    pub bound_binary_operator_kind: Box<BoundBinaryOperator>,
    pub right: Box<BoundNodeKind>,
}

#[derive(Clone, Debug)]
pub struct BoundGoToStatment {
    pub label: LabelSymbol,
}

impl BoundGoToStatment {
    pub fn new(label: LabelSymbol) -> Self {
        Self { label }
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Hash)]
pub struct LabelSymbol {
    name: String,
}

impl LabelSymbol {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Hash)]
pub struct ParameterSymbol {
    name: String,
    _type: Type
}

impl ParameterSymbol {
    pub fn new(name: String, _type: Type) -> Self {
        Self { name, _type }
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Hash)]
pub struct FunctionSymbol {
    pub name: String,
    paremeters: Vec<ParameterSymbol>
}

impl FunctionSymbol {
    pub fn new(name: String, paremeters: Vec<ParameterSymbol>, return_type: Type) -> Self {
        Self { name, paremeters }
    }
}

#[derive(Clone, Debug)]
pub struct BoundUnaryOperator {
    pub syntax_kind: SyntaxKind,
    pub operator_kind: BoundUnaryOperatorKind,
    pub operand_type: Type,
    pub result_type: Type,
}

impl BoundUnaryOperator {
    pub fn new(
        syntax_kind: SyntaxKind,
        operator_kind: BoundUnaryOperatorKind,
        operand_type: Type,
        result_type: Type,
    ) -> Self {
        Self {
            syntax_kind,
            operator_kind,
            operand_type,
            result_type,
        }
    }

    pub fn new_same_type(
        syntax_kind: SyntaxKind,
        operator_kind: BoundUnaryOperatorKind,
        operand_type: Type,
    ) -> Self {
        Self {
            syntax_kind,
            operator_kind,
            operand_type: operand_type.clone(),
            result_type: operand_type,
        }
    }

    pub fn bind(syntax_kind: SyntaxKind, operand_type: Type) -> Option<BoundUnaryOperator> {
        let thing = vec![
            Self::new_same_type(
                SyntaxKind::PlusToken,
                BoundUnaryOperatorKind::Identity,
                Type::Number,
            ),
            Self::new_same_type(
                SyntaxKind::MinusToken,
                BoundUnaryOperatorKind::Negation,
                Type::Number,
            ),
            Self::new_same_type(
                SyntaxKind::BangToken,
                BoundUnaryOperatorKind::LogicalNegation,
                Type::Bool,
            ),
            Self::new_same_type(
                SyntaxKind::Tilda,
                BoundUnaryOperatorKind::OnesCompliment,
                Type::Number,
            ),
        ];

        for op in thing {
            if op.syntax_kind == syntax_kind && op.operand_type == operand_type {
                return Some(op);
            }
        }

        return None
    }
}

#[derive(Clone, Debug)]
pub struct BoundBinaryOperator {
    pub syntax_kind: SyntaxKind,
    pub operator_kind: BoundBinaryOperatorKind,
    pub left_type: Type,
    pub right_type: Type,
    pub result_type: Type,
}

impl BoundBinaryOperator {
    pub fn new(
        syntax_kind: SyntaxKind,
        operator_kind: BoundBinaryOperatorKind,
        left_type: Type,
        right_type: Type,
        result_type: Type,
    ) -> Self {
        Self {
            syntax_kind,
            operator_kind,
            left_type,
            right_type,
            result_type,
        }
    }

    pub fn new_same_type(
        syntax_kind: SyntaxKind,
        operator_kind: BoundBinaryOperatorKind,
        operand_type: Type,
    ) -> Self {
        Self {
            syntax_kind,
            operator_kind,
            left_type: operand_type.clone(),
            right_type: operand_type.clone(),
            result_type: operand_type.clone(),
        }
    }

    pub fn different_output_type(
        syntax_kind: SyntaxKind,
        operator_kind: BoundBinaryOperatorKind,
        operand_type: Type,
        output_type: Type,
    ) -> Self {
        Self {
            syntax_kind,
            operator_kind,
            left_type: operand_type.clone(),
            right_type: operand_type.clone(),
            result_type: output_type.clone(),
        }
    }

    pub fn bind(syntax_kind: SyntaxKind, left_type: Type, right_type: Type) -> Option<BoundBinaryOperator> {
        if syntax_kind == SyntaxKind::EqEqToken {
            return Some(Self::new(
                SyntaxKind::EqEqToken,
                BoundBinaryOperatorKind::Equals,
                left_type,
                right_type,
                Type::Bool,
            ));
        } else if syntax_kind == SyntaxKind::BangEqToken {
            return Some(Self::new(
                SyntaxKind::BangEqToken,
                BoundBinaryOperatorKind::DoesNotEqual,
                left_type,
                right_type,
                Type::Bool,
            ));
        }

        let thing = vec![
            Self::new_same_type(
                SyntaxKind::AmpersandToken,
                BoundBinaryOperatorKind::BitwiseAnd,
                Type::Bool,
            ),
            Self::new_same_type(
                SyntaxKind::Pipe,
                BoundBinaryOperatorKind::BitwiseOr,
                Type::Bool,
            ),
            Self::new_same_type(
                SyntaxKind::PlusToken,
                BoundBinaryOperatorKind::Addition,
                Type::String,
            ),
            Self::new_same_type(
                SyntaxKind::Tilda,
                BoundBinaryOperatorKind::BitwiseXor,
                Type::Bool,
            ),
            Self::new_same_type(
                SyntaxKind::AmpersandToken,
                BoundBinaryOperatorKind::BitwiseAnd,
                Type::Number,
            ),
            Self::new_same_type(
                SyntaxKind::Pipe,
                BoundBinaryOperatorKind::BitwiseOr,
                Type::Number,
            ),
            Self::new_same_type(
                SyntaxKind::Tilda,
                BoundBinaryOperatorKind::BitwiseXor,
                Type::Number,
            ),
            Self::different_output_type(
                SyntaxKind::GreaterThan,
                BoundBinaryOperatorKind::GreaterThan,
                Type::Number,
                Type::Bool,
            ),
            Self::different_output_type(
                SyntaxKind::GreaterThanEqualTo,
                BoundBinaryOperatorKind::GreaterThanEqualTo,
                Type::Number,
                Type::Bool,
            ),
            Self::different_output_type(
                SyntaxKind::LessThan,
                BoundBinaryOperatorKind::LessThanEqualTo,
                Type::Number,
                Type::Bool,
            ),
            Self::different_output_type(
                SyntaxKind::LessThanEqualTo,
                BoundBinaryOperatorKind::LessThanEqualTo,
                Type::Number,
                Type::Bool,
            ),
            Self::new_same_type(
                SyntaxKind::PlusToken,
                BoundBinaryOperatorKind::Addition,
                Type::Number,
            ),
            Self::new_same_type(
                SyntaxKind::MinusToken,
                BoundBinaryOperatorKind::Subtraction,
                Type::Number,
            ),
            Self::new_same_type(
                SyntaxKind::MulToken,
                BoundBinaryOperatorKind::Multiplication,
                Type::Number,
            ),
            Self::new_same_type(
                SyntaxKind::DivToken,
                BoundBinaryOperatorKind::Division,
                Type::Number,
            ),
            Self::new_same_type(
                SyntaxKind::PipePipeToken,
                BoundBinaryOperatorKind::LogicalOr,
                Type::Bool,
            ),
            Self::new_same_type(
                SyntaxKind::AmpersandAmpersandToken,
                BoundBinaryOperatorKind::LogicalAnd,
                Type::Bool,
            ),
        ];

        for op in thing {
            if op.syntax_kind == syntax_kind
                && op.left_type == left_type
                && op.right_type == right_type
            {
                return Some(op);
            }
        }

        return None;
    }
}

pub struct Binder {
    pub diagnostics: DiagnosticBag,
    pub scope: Box<Option<BoundScope>>,
}

impl Binder {
    pub fn new(scopeee: Option<BoundScope>) -> Self {
        Self {
            diagnostics: DiagnosticBag::new(),
            scope: Box::from(scopeee),
        }
    }

    pub fn create_parent(mut previous: Option<BoundGlobalScope>) -> Option<BoundScope> {
        let mut stack = VecDeque::new();
        while let Some(prev) = previous.clone() {
            stack.push_back(previous.clone().unwrap());
            previous = if let Some(prev) = prev.previous.clone() {
                Some(*prev)
            } else {
                None
            }
        }

        let mut parent = Some(Binder::create_root_scope());

        while stack.len() > 0 {
            previous = stack.pop_back();
            let mut scope = BoundScope::new(parent);
            for v in previous.unwrap().variables {
                scope.try_declare_variable(v.clone().name, v.clone());
            }
            parent = Some(scope);
        }

        return parent;
    }

    pub fn create_root_scope() -> BoundScope {
        let mut result = BoundScope::new(None);
        let built_in_funcitons = BuiltInFunctions::new();

        for func in &built_in_funcitons.functions {
            result.try_declare_function(func.name.clone(), func.clone());
        }

        return result;
    }

    pub fn bind_global_scope(
        previous: Option<BoundGlobalScope>,
        compilation_unit: CompilationUnit,
    ) -> BoundGlobalScope {
        let parent_scope = Binder::create_parent(previous.clone());
        let mut scope = Some(BoundScope::new(parent_scope.clone()));
        let mut binder = Binder::new(scope);
        let expression = binder.bind_statement(*compilation_unit.root);

        let variables = if let Some(scope) = *binder.scope {
            scope
                .variables
                .values()
                .cloned()
                .collect::<Vec<VariableSymbol>>()
        } else {
            vec![]
        };

        BoundGlobalScope::new(
            previous.clone(),
            DiagnosticBag::new(),
            variables,
            expression,
        )
    }

    pub fn get_diagnostics(&self) -> DiagnosticBag {
        return self.diagnostics.clone();
    }

    pub fn bind_expression(&mut self, syntax: SyntaxNode, can_be_void: bool) -> BoundNodeKind {
        let result = self.bind_expression_internal(syntax);
        if let BoundNodeKind::BoundLiteralExpression(lit) = result.clone() {
            if lit.is_void() && !can_be_void {
                // report expression must have value ....
                return BoundNodeKind::BoundErrorExpression(BoundErrorExpression::new());
            } else {
                return result.clone()
            }
        }
        return result.clone()
    }

    pub fn bind_expression_internal(&mut self, expression_kind: SyntaxNode) -> BoundNodeKind {
        return match expression_kind {
            SyntaxNode::CallExpressionSyntax(call) => self.bind_call_expression(call),
            SyntaxNode::NameExpressionSyntax(name) => self.bind_name_expression(name),
            SyntaxNode::AssignmentExpressionSyntax(asign) => self.bind_assignment_expression(asign),
            SyntaxNode::ParenthesizedExpression(par) => self.bind_parenthesis_expression(par),
            SyntaxNode::LiteralNode(number_node) => self.bind_literal_expression(number_node),
            SyntaxNode::UnaryExpressionSyntax(unary_node) => {
                self.bind_unary_expresssion(unary_node)
            }
            SyntaxNode::BinaryExpressionSyntax(binary) => {
                self.bind_binary_expression(binary)
            }
            SyntaxNode::VariableDeclaration(var_decl) => self.bind_variable_declaration(var_decl),
            _ => {
                println!("expression kind {}", expression_kind);
                BoundNodeKind::Unkown
            }
        };
    }

    pub fn lookup_type(&self, name: String) -> Type {
        match name.as_str() {
            "bool" => Type::Bool,
            "string" => Type::String,
            "int" => Type::Number,
            _ => Type::Any

        }
    }

    /// somethign that we are not doing here is optiimizing memory ... this can be done by checking if rewirtetend if the same and than ereturnt the same thing

    pub fn bind_conversion(&mut self, _type: Type, syntax: SyntaxNode, allow_explicit: bool) -> BoundNodeKind {
        let expression = self.bind_expression(syntax.clone(), false);
        if let BoundNodeKind::BoundErrorExpression(_) = expression {
            // error here cannot convert
            return BoundNodeKind::BoundErrorExpression(BoundErrorExpression::new())
        }
        let conversion = Conversion::classifiy(expression.get_type(), _type.clone());
        if conversion == ConversionType::None {
            // report cannot convert error here
            return BoundNodeKind::BoundErrorExpression(BoundErrorExpression::new())
        }

        if !allow_explicit && false { // and check if it's expecity
            // report cannot convert error here
            return BoundNodeKind::BoundErrorExpression(BoundErrorExpression::new())
        }

        return BoundNodeKind::BoundConversionExpression(BoundConversionExpression::new(_type.clone(), expression))

    }

    pub fn bind_call_expression(&mut self, expression: CallExpressionSyntax) -> BoundNodeKind {

        if expression.arguments.seperators_and_nodes.len() == 1 && self.lookup_type(expression.clone().identifier.text) != Type::Any {
            return self.bind_conversion(self.lookup_type(expression.clone().identifier.text.clone()), *expression.clone().arguments.clone().seperators_and_nodes.clone().get(0).clone().unwrap().clone(), true)
        }

        let functions = BuiltInFunctions::new();

        let mut function_option = functions.functions.iter().find(|f| f.name == expression.identifier.text);

        if function_option == None {
            if !self.scope.clone().unwrap().try_lookup_function(expression.identifier.text) {
                return BoundNodeKind::BoundErrorExpression(BoundErrorExpression::new())
            }
        }

        let function = function_option.unwrap();

        let arguments: Vec<&Box<SyntaxNode>> = expression.arguments.seperators_and_nodes.iter().filter(|n| ***n != SyntaxNode::Comma ).collect();
        let arguments_binded: Vec<BoundNodeKind> = arguments.iter().map(|a| self.bind_expression(*a.clone().clone(), false )).collect();


        if arguments_binded.len() != function.paremeters.len() {
            println!("erroring here 2");
            // report wrong argument
            return BoundNodeKind::BoundErrorExpression(BoundErrorExpression::new())
        }

        for i in 0..function.paremeters.len() {
            let parameter = function.paremeters.get(i).unwrap();
            let argument = arguments_binded.get(i).unwrap();
            if let BoundNodeKind::BoundLiteralExpression(lit) = argument {
                if parameter._type != lit.into_value()._type {
                    // error here with wrong argument type
                    println!("erroring here 1");
                    return BoundNodeKind::BoundErrorExpression(BoundErrorExpression::new())
                }
            }
            
        }


        BoundNodeKind::BoundCallExpresisonNode(BoundCallExpressionNode::new(function.clone(), arguments_binded))
    }

    pub fn bind_statement(&mut self, expression_kind: SyntaxNode) -> BoundNodeKind {
        return match expression_kind {
            SyntaxNode::ForStatementSytnax(for_statement) => self.bind_for_statement(for_statement),
            SyntaxNode::BlockStatmentSyntax(block_stament) => {
                self.bind_block_statement(block_stament)
            }
            SyntaxNode::ExpressionSyntaxStatement(expression_syntax_statement) => {
                self.bind_expression_statement_syntax(expression_syntax_statement)
            }
            SyntaxNode::IfStatementSyntax(if_statement_syntax) => {
                self.bind_if_statement(if_statement_syntax)
            }
            SyntaxNode::WhileStatmentSyntax(while_statment_sytnax) => {
                self.bind_while_statment(while_statment_sytnax)
            }
            _ => self.bind_expression(expression_kind, false),
        };
    }

    pub fn bind_for_statement(&mut self, statement: ForStatementSyntax) -> BoundNodeKind {
        let lower_bound = self.bind_expression(*statement.lower_bound, false);
        let upper_bound = self.bind_expression(*statement.upper_bound, false);

        self.scope = Box::from(Some(BoundScope::new(*self.scope.clone())));

        let name = statement.identifier.text;
        let variable = VariableSymbol::new(name.clone(), Type::Number);
        if !self
            .scope
            .as_mut()
            .as_mut()
            .unwrap()
            .try_declare_variable(name, variable.clone())
        {
            // error here already declared ...
        }
        // he abstracted this to anohter function bind_variable
        

        let body = self.bind_statement(*statement.body);

        self.scope = Box::from(Some(*self.scope.clone().unwrap().parent.unwrap()));

        BoundNodeKind::BoundForStatment(BoundForStatement::new(
            variable.clone(),
            lower_bound,
            upper_bound,
            body,
        ))
    }


    pub fn bind_variable(&mut self, identifier: SyntaxToken, _type: Type) {
        let name = identifier.text;
        
    }

    pub fn bind_while_statment(&mut self, statement: WhileStatmentSyntax) -> BoundNodeKind {
        let condition = self.bind_expression(*statement.condiiton, false);
        let body = self.bind_statement(*statement.body);
        BoundNodeKind::BoundWhileStatment(BoundWhileStatment::new(condition, body))
    }

    pub fn bind_if_statement(&mut self, statement: IfStatementSyntax) -> BoundNodeKind {
        let condition = self.bind_expression_target_type(*statement.condition, Type::Bool);
        let then_statement = self.bind_statement(*statement.then_statement);
        let else_statement = if let Some(else_statement) = statement.else_clause {
            if let SyntaxNode::ElseClause(else_clause) = *else_statement {
                Some(self.bind_statement(*else_clause.else_statement))
            } else {
                None
            }
        } else {
            None
        };
        return BoundNodeKind::BoundIfStatement(BoundIfStatement::new(
            condition,
            then_statement,
            else_statement,
        ));
    }

    pub fn bind_expression_target_type(
        &mut self,
        syntax: SyntaxNode,
        _type: Type,
    ) -> BoundNodeKind {
        let expression = self.bind_expression(syntax, false);
        if let BoundNodeKind::BoundLiteralExpression(lit) = expression.clone() {
            let thing = match _type {
                Type::Bool => lit.is_bool(),
                Type::Number => lit.is_num(),
                _ => false,
            };
            if !thing {
                // cannot convert here ....
            }
        } else {
            // cannot convert
        }
        expression.clone()
    }

    pub fn bind_expression_statement_syntax(
        &mut self,
        expression_syntax: ExpressionSyntaxStatement,
    ) -> BoundNodeKind {
        self.bind_expression(*expression_syntax.expression, true)
    }

    pub fn bind_block_statement(&mut self, block_statement: BlockStatmentSyntax) -> BoundNodeKind {
        let mut statments = vec![];

        self.scope = Box::from(Some(BoundScope::new(*self.scope.clone())));

        block_statement.statements.iter().for_each(|statement| {
            let statement = self.bind_statement(*statement.clone());
            statments.push(statement);
        });

        self.scope = Box::from(Some(*self.scope.clone().unwrap().parent.unwrap()));

        BoundNodeKind::BoundBlockStatement(BoundBlockStatement::new(statments))
    }

    pub fn bind_name_expression(&mut self, name: NameExpressionSyntax) -> BoundNodeKind {
        if !self
            .scope
            .clone()
            .unwrap()
            .try_lookup_variable(name.token.text.clone())
        {
            self.diagnostics.report_unknown_variable(name.token.span);
            return BoundNodeKind::BoundErrorExpression(BoundErrorExpression::new());
        }

        return BoundNodeKind::BoundVariableExpression(BoundVariableExpression {
            name: name.token.text.clone(),
            _type: Type::Number,
        });
    }

    ///
    /// explicit vs implicit ones
    /// 
    /// imlpicit is when you don't loose data
    /// 
    /// expliciti is when programmer needs to specififcy which data

    pub fn bind_variable_declaration(&mut self, syntax: VariableDeclaration) -> BoundNodeKind {
        let mut name = syntax.clone().identifier.text;

        // TODO: check if the identifier is missing and then declare only if the variable is not missing, this helps the case where we have Let
        let is_read_only = syntax.clone().identifier.kind == SyntaxKind::LetKeyword;
        let _type = self.bind_type_cluase(syntax.clone().type_clause);
        let expression = self.bind_expression(*syntax.clone().initializer, false);
        let variable_type = if let Some(_type) = _type { _type } else { if let BoundNodeKind::BoundLiteralExpression(lit) = expression.clone() { lit.into_value()._type } else { panic!()}};
        let converted_initializer = self.bind_conversion(variable_type, *syntax.initializer, false);
        let variable = VariableSymbol {
            name: name.clone(),
            _type: Type::Any,
        };

        if self
            .scope
            .as_mut()
            .as_mut()
            .unwrap()
            .try_declare_variable(name.clone(), variable.clone())
        {
            self.get_diagnostics().diagnostics;
        }

        // check to ensure bound node kind is expresison

        BoundNodeKind::BoundVariableDeclaration(BoundVariableDeclarion::new(variable, converted_initializer))
    }

    pub fn bind_type_cluase(&mut self, syntax: Option<Box<SyntaxNode>>) -> Option<Type> {
        if let Some(box_thing) = syntax {
            if let SyntaxNode::TypeClause(t_c) = *box_thing {
            let _type = self.lookup_type(t_c.identifier.text);
            // check if type is none and then report error if it is....
            return Some(_type);
        }
        }
        None
    }


    pub fn bind_assignment_expression(
        &mut self,
        syntax: AssignmentExpressionSyntax,
    ) -> BoundNodeKind {
        let bound_expression = self.bind_expression(*syntax.expression, false);
        let name = syntax.identifier_token.text;

        if let Some(scope) = self.scope.as_mut() {
            let variable_name = VariableSymbol {
                name: name.clone(),
                _type: bound_expression.get_type(),
            };

            if !scope.try_lookup_variable(name.clone()) {
                panic!("need to declare first ...");
            }

            // if !scope.try_declare_variable(name.clone(), variable_name) {

            // }

            // add type checking here ....
        }

        // bind_converion here...
        // let converted_expression = self.bind_conversion(T, syntax);

        return BoundNodeKind::BoundAssignmentExpression(BoundAssignmentExpression {
            name: name.clone(),
            bound_expression: Box::from(bound_expression),
        });
    }

    pub fn bind_literal_expression(&mut self, syntax: Value) -> BoundNodeKind {
        if let Some(value) = syntax.value {
            return BoundNodeKind::BoundLiteralExpression(BoundLiteralExpression {
                value_num: Some(value),
                value_bool: None,
                value_string: None,
            });
        } else if let Some(bool) = syntax.bool {
            return BoundNodeKind::BoundLiteralExpression(BoundLiteralExpression {
                value_num: None,
                value_bool: Some(bool),
                value_string: None,
            });
        } else if let Some(string) = syntax.string {
            return BoundNodeKind::BoundLiteralExpression(BoundLiteralExpression {
                value_num: None,
                value_bool: None,
                value_string: Some(string),
            });
        } 

        return BoundNodeKind::BoundLiteralExpression(BoundLiteralExpression {
            value_bool: None,
            value_num: None,
            value_string: None,
        });
    }

    pub fn bind_unary_expresssion(
        &mut self,
        bound_unary_expression: UnaryExpressionSyntax,
    ) -> BoundNodeKind {
        let bound_operand = self.bind_expression(*bound_unary_expression.operand.clone(), false);

        if let BoundNodeKind::BoundErrorExpression(_) = bound_operand {
            return BoundNodeKind::BoundErrorExpression(BoundErrorExpression::new())
        }

        let bound_operator_kind = BoundUnaryOperator::bind(
            bound_unary_expression
                .clone()
                .operator_token
                .try_into_syntax_kind(),
            bound_operand.get_type(),
        );

        if bound_operator_kind.is_none() == true {
            return BoundNodeKind::BoundErrorExpression(BoundErrorExpression::new());
        }


        return BoundNodeKind::BoundUnaryExpressionNode(BoundUnaryExpressionNode {
            operator_kind: Box::from(bound_operator_kind.unwrap()),
            operand: Box::from(bound_operand),
        });
    }

    pub fn bind_binary_expression(
        &mut self,
        bound_binary_expression: BinaryExpressionSyntax,
    ) -> BoundNodeKind {
        let left = self.bind_expression(*bound_binary_expression.left, false);
        let right = self.bind_expression(*bound_binary_expression.right, false);

        if let BoundNodeKind::BoundErrorExpression(_) = left {
            return BoundNodeKind::BoundErrorExpression(BoundErrorExpression::new())
        }

        if let BoundNodeKind::BoundErrorExpression(_) = right {
            return BoundNodeKind::BoundErrorExpression(BoundErrorExpression::new())
        }

        let bound_operand = BoundBinaryOperator::bind(
            bound_binary_expression
                .operator_token
                .try_into_syntax_kind(),
            left.get_type(),
            right.get_type(),
        );

        if bound_operand.is_none() == true {
            return BoundNodeKind::BoundErrorExpression(BoundErrorExpression::new());
        }


        return BoundNodeKind::BoundBinaryExpression(BoundBinaryExpression {
            left: Box::from(left),
            bound_binary_operator_kind: Box::from(bound_operand.unwrap()),
            right: Box::from(right),
        });
    }

    pub fn bind_parenthesis_expression(
        &mut self,
        expression: ParenthesizedExpression,
    ) -> BoundNodeKind {
        return self.bind_expression(*expression.sub_expression, false);
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

#[derive(Clone, Debug)]
pub struct VariableSymbol {
    pub name: String,
    pub _type: Type,
}

impl VariableSymbol {
    pub fn new(name: String, _type: Type) -> Self {
        VariableSymbol { name, _type }
    }
}

#[derive(Clone, Debug)]
pub struct BoundGlobalScope {
    previous: Option<Box<BoundGlobalScope>>,
    diagnostics: DiagnosticBag,
    variables: Vec<VariableSymbol>,
    pub expression: BoundNodeKind,
}

impl BoundGlobalScope {
    pub fn new(
        previous: Option<BoundGlobalScope>,
        diagnostics: DiagnosticBag,
        variables: Vec<VariableSymbol>,
        expression: BoundNodeKind,
    ) -> Self {
        let previous = if let Some(previous) = previous {
            Some(Box::from(previous))
        } else {
            None
        };
        Self {
            previous: previous,
            diagnostics,
            variables,
            expression,
        }
    }
}

#[derive(Clone)]
pub struct BoundScope {
    variables: HashMap<String, VariableSymbol>,
    functions: HashMap<String, FunctionSymbol>,
    parent: Option<Box<BoundScope>>,
}

impl BoundScope {
    pub fn new(parent: Option<BoundScope>) -> Self {
        let parent = if let Some(parent) = parent {
            Some(Box::from(parent))
        } else {
            None
        };

        BoundScope {
            variables: HashMap::new(),
            functions: HashMap::new(),
            parent,
        }
    }

    pub fn try_declare_variable(&mut self, name: String, variable: VariableSymbol) -> bool {
        ///
        /// we want to support nested scopes
        ///
        /// var x = 10
        /// {
        ///     var x = False // this should be ok.
        /// }
        ///
        /// var x = False // this should error
        ///
        if self.variables.clone().contains_key(&variable.name) {
            return false;
        }

        self.variables
            .insert(variable.clone().name, variable.clone());
        return true;
    }

    pub fn try_lookup_variable(&mut self, name: String) -> bool {
        if self.variables.contains_key(&name) {
            return true;
        }

        if self.parent.is_none() {
            return false;
        }

        return self.parent.as_mut().unwrap().try_lookup_variable(name);
    }

    pub fn try_declare_function(&mut self, name: String, function: FunctionSymbol) -> bool {
        ///
        /// we want to support nested scopes
        ///
        /// var x = 10
        /// {
        ///     var x = False // this should be ok.
        /// }
        ///
        /// var x = False // this should error
        ///
        if self.functions.clone().contains_key(&function.name) {
            return false;
        }

        self.functions
            .insert(function.clone().name, function.clone());
        return true;
    }

    pub fn try_lookup_function(&mut self, name: String) -> bool {
        if self.functions.contains_key(&name) {
            return true;
        }

        if self.parent.is_none() {
            return false;
        }

        return self.parent.as_mut().unwrap().try_lookup_function(name);
    }

    pub fn get_declared_variables(&self) -> Vec<VariableSymbol> {
        self.variables
            .values()
            .cloned()
            .collect::<Vec<VariableSymbol>>()
    }

    pub fn get_declared_functions(&self) -> Vec<FunctionSymbol> {
        self.functions
            .values()
            .cloned()
            .collect::<Vec<FunctionSymbol>>()
    }
}

pub struct Lowerer {
    label_count: i128,
}

impl Lowerer {
    pub fn generate_label(&mut self) -> LabelSymbol {
        let name = format!("Label{}", self.label_count);
        self.label_count += 1;
        return LabelSymbol::new(name);
    }

    pub fn new() -> Self {
        Self { label_count: 0 }
    }

    fn flatten(statment: BoundNodeKind) -> BoundBlockStatement {
        let mut all_statements = vec![];
        let mut stack = VecDeque::new();
        stack.push_back(statment);
        while stack.len() > 0 {
            let current = stack.pop_back().unwrap();
            if let BoundNodeKind::BoundBlockStatement(block) = current {
                let mut reversed = block.statments.clone();
                reversed.reverse();
                reversed
                    .iter()
                    .for_each(|stmt| stack.push_back(*stmt.clone()))
            } else {
                all_statements.push(current)
            }
        }

        BoundBlockStatement::new(all_statements)
    }

    pub fn lower(statement: BoundBlockStatement) -> BoundBlockStatement {
        let mut lowerer = Lowerer::new();
        let result = lowerer.rewrite_statment(BoundNodeKind::BoundBlockStatement(statement));

        let flatten_result = Lowerer::flatten(result);
        flatten_result
    }

    pub fn rewrite_bound_conversion_expression(&mut self, exp: BoundConversionExpression) -> BoundNodeKind {
        let expresion = self.rewrite_expression(*exp.expression.clone());
        BoundNodeKind::BoundConversionExpression(BoundConversionExpression::new(exp._type, expresion))
    }

    pub fn rewrite_statment(&mut self, bound_statment: BoundNodeKind) -> BoundNodeKind {
        match bound_statment {
            BoundNodeKind::BoundConversionExpression(boudn_conversion) => {
                self.rewrite_bound_conversion_expression(boudn_conversion)
            }
            BoundNodeKind::BoundCallExpresisonNode(bound_call_expression_node) => {
                self.rewrite_bound_call_expression(bound_call_expression_node)
            }
            BoundNodeKind::BoundBlockStatement(bound_block_statment) => {
                self.rewrite_bound_block_statement(bound_block_statment)
            }
            BoundNodeKind::BoundVariableDeclaration(bound_variable_declaration) => {
                self.rewrite_variable_declaration(bound_variable_declaration)
            }
            BoundNodeKind::BoundConditionalGoToStatment(bound_conditional_goto) => {
                self.rewrite_bound_conditional_goto(bound_conditional_goto)
            }
            BoundNodeKind::BoundGoToStatment(bound_goto_statement) => {
                self.rewrite_bound_goto(bound_goto_statement)
            }
            BoundNodeKind::BoundLabelStatement(bound_label_statment) => {
                self.rewrite_bound_label(bound_label_statment)
            }
            BoundNodeKind::BoundIfStatement(bound_if_statment) => {
                self.rewrite_if_statement(bound_if_statment)
            }
            BoundNodeKind::BoundWhileStatment(bound_while_statmten) => {
                self.rewrite_while_statement(bound_while_statmten)
            }
            BoundNodeKind::BoundForStatment(bound_for_statement) => {
                self.rewrite_bound_for_statement(bound_for_statement)
            }
            BoundNodeKind::BoundExpressionStatement(bound_expression_statement) => {
                self.rewrite_expression_statment(bound_expression_statement)
            }
            BoundNodeKind::BoundLiteralExpression(bound_literal_expression) => {
                self.rewrite_bound_literal_expression(bound_literal_expression)
            }
            BoundNodeKind::BoundBinaryExpression(bound_binary_expression) => {
                self.rewrite_bound_binary_expression(bound_binary_expression)
            }
            BoundNodeKind::BoundUnaryExpressionNode(bound_unary_expression) => {
                self.rewrite_bound_unary_expression(bound_unary_expression)
            }
            BoundNodeKind::BoundVariableExpression(bound_variable_expression) => {
                self.rewrite_bound_variable_expression(bound_variable_expression)
            }
            BoundNodeKind::BoundAssignmentExpression(bound_assignment_expression) => {
                self.rewrite_bound_assignment_expression(bound_assignment_expression)
            }
            _ => { println!("bound_statment {bound_statment:?}"); BoundNodeKind::Unkown },
        }
    }

    pub fn rewrite_bound_label(&mut self, node: BoundLabelStatement) -> BoundNodeKind {
        return BoundNodeKind::BoundLabelStatement(node);
    }

    pub fn rewrite_bound_call_expression(&mut self, node: BoundCallExpressionNode) -> BoundNodeKind {
        let new_argumtns = node.arguments.iter().map(|a| self.rewrite_expression(*a.clone())).collect();
        BoundNodeKind::BoundCallExpresisonNode(BoundCallExpressionNode::new(node.function, new_argumtns))
    }

    pub fn rewrite_bound_goto(&mut self, node: BoundGoToStatment) -> BoundNodeKind {
        return BoundNodeKind::BoundGoToStatment(node);
    }

    pub fn rewrite_bound_conditional_goto(
        &mut self,
        node: BoundConditionalGoToStatment,
    ) -> BoundNodeKind {
        let condition = self.rewrite_expression(*node.condition);
        return BoundNodeKind::BoundConditionalGoToStatment(BoundConditionalGoToStatment::new(
            node.symbol,
            condition,
            node.jump_if_false,
        ));
    }

    pub fn rewrite_bound_block_statement(
        &mut self,
        statment: BoundBlockStatement,
    ) -> BoundNodeKind {
        let mut statments_rewritten = vec![];
        for statment in statment.statments {
            let statment = self.rewrite_statment(*statment);
            statments_rewritten.push(statment)
        }

        // can be an optimization that only allocates if they are diffent

        BoundNodeKind::BoundBlockStatement(BoundBlockStatement::new(statments_rewritten))
    }

    // pub fn rewrite_if_statement(&self, statment: BoundIfStatement) -> BoundNodeKind {
    //     let condition = self.rewrite_statment(*statment.condition);
    //     let then_statement = self.rewrite_statment(*statment.then_statment);
    //     let else_statement = if let Some(else_st) = statment.else_statement { Some(self.rewrite_statment(*else_st)) } else { None};
    //     return BoundNodeKind::BoundIfStatement(BoundIfStatement::new(condition, then_statement, else_statement))
    // }

    pub fn rewrite_if_statement(&mut self, statment: BoundIfStatement) -> BoundNodeKind {
        ///
        /// if <condition>
        ///     <then>
        ///
        /// ---->
        ///
        /// gotoIfFalse <condition> end
        /// <then>
        /// end:
        ///
        ///
        /// ================
        ///
        /// if <condition>
        ///     <then>
        /// else
        ///     <else>
        ///
        /// ---->
        ///
        /// gotoIfFasle <condition> else
        /// <then>
        /// gotoend
        /// else:
        /// <else>
        /// end:
        ///
        ///
        if let Some(else_stmt) = statment.else_statement {
            let else_label = self.generate_label();
            let end_label = self.generate_label();

            let goto_false = BoundNodeKind::BoundConditionalGoToStatment(
                BoundConditionalGoToStatment::new(else_label.clone(), *statment.condition, true),
            );
            let goto_end_statment =
                BoundNodeKind::BoundGoToStatment(BoundGoToStatment::new(end_label.clone()));
            let else_label_statement =
                BoundNodeKind::BoundLabelStatement(BoundLabelStatement::new(else_label));
            let end_label_statement =
                BoundNodeKind::BoundLabelStatement(BoundLabelStatement::new(end_label));
            let result = BoundNodeKind::BoundBlockStatement(BoundBlockStatement::new(vec![
                goto_false,
                *statment.then_statment,
                goto_end_statment,
                else_label_statement,
                *else_stmt,
                end_label_statement,
            ]));
            return self.rewrite_statment(result);

            BoundNodeKind::Unkown
        } else {
            let end_label = self.generate_label();
            let goto_false = BoundNodeKind::BoundConditionalGoToStatment(
                BoundConditionalGoToStatment::new(end_label.clone(), *statment.condition, true),
            );
            let end_label_statement =
                BoundNodeKind::BoundLabelStatement(BoundLabelStatement::new(end_label));
            let result = BoundNodeKind::BoundBlockStatement(BoundBlockStatement::new(vec![
                goto_false,
                *statment.then_statment,
                end_label_statement,
            ]));
            return self.rewrite_statment(result);
        }
    }

    // pub fn rewrite_while_statement(&self, statment: BoundWhileStatment) -> BoundNodeKind {
    //     let condition = self.rewrite_statment(*statment.condition);
    //     let body = self.rewrite_statment(*statment.body);
    //     return BoundNodeKind::BoundWhileStatment(BoundWhileStatment::new(condition, body))
    // }

    pub fn rewrite_while_statement(&mut self, statment: BoundWhileStatment) -> BoundNodeKind {
        ///
        ///
        /// while <condition>
        ///     <body>
        ///
        /// ----->
        ///
        /// goto check:
        /// continue:
        /// <body>
        /// check:
        /// gotoTrue <condition> continue
        /// end:
        ///
        ///        
        let continue_label = self.generate_label();
        let check_label = self.generate_label();
        let end_label = self.generate_label();

        let goto_check =
            BoundNodeKind::BoundGoToStatment(BoundGoToStatment::new(check_label.clone()));
        let continue_label_statment =
            BoundNodeKind::BoundLabelStatement(BoundLabelStatement::new((continue_label.clone())));
        let check_label_statment =
            BoundNodeKind::BoundLabelStatement(BoundLabelStatement::new((check_label)));
        let goto_true = BoundNodeKind::BoundConditionalGoToStatment(
            BoundConditionalGoToStatment::new(continue_label, *statment.condition, false),
        );
        let end_label_statment =
            BoundNodeKind::BoundLabelStatement(BoundLabelStatement::new((end_label)));

        let result = BoundNodeKind::BoundBlockStatement(BoundBlockStatement::new(vec![
            goto_check,
            continue_label_statment,
            *statment.body,
            check_label_statment,
            goto_true,
            end_label_statment,
        ]));
        return result;
    }

    // pub fn rewrite_bound_for_statement(&self, statment: BoundForStatement) -> BoundNodeKind {
    //     let lower_bound = self.rewrite_statment(*statment.lower_bound);
    //     let upper_bound = self.rewrite_statment(*statment.upper_bound);
    //     let body = self.rewrite_statment(*statment.body);

    //     return BoundNodeKind::BoundForStatment(BoundForStatement::new(statment.variable, lower_bound, upper_bound, body))
    // }

    pub fn rewrite_bound_for_statement(&mut self, statment: BoundForStatement) -> BoundNodeKind {
        // for <var> = <lower> to <upper>
        ///
        /// ---->
        ///
        ///
        /// {
        ///     var <var> = <lower>
        ///     while (<var> <= <upper>)
        ///     {
        ///         <body>
        ///         <var> = <var> + 1
        ///     }
        /// }
        // let lower_bound = self.rewrite_statment(*statment.lower_bound);
        // let upper_bound = self.rewrite_statment(*statment.upper_bound);
        // let body = self.rewrite_statment(*statment.body);
        let variable_declaration = BoundNodeKind::BoundVariableDeclaration(
            BoundVariableDeclarion::new(statment.variable.clone(), *statment.lower_bound),
        );
        let condition = BoundNodeKind::BoundBinaryExpression(BoundBinaryExpression {
            left: Box::from(BoundNodeKind::BoundVariableExpression(
                BoundVariableExpression::new(
                    statment.variable.clone().name,
                    statment.variable.clone()._type,
                ),
            )),
            bound_binary_operator_kind: Box::from(BoundBinaryOperator::bind(
                SyntaxKind::LessThanEqualTo,
                Type::Number,
                Type::Number,
            ).unwrap()),
            right: statment.upper_bound,
        });
        let increment = BoundNodeKind::BoundAssignmentExpression(BoundAssignmentExpression {
            name: statment.variable.clone().name,
            bound_expression: Box::from(BoundNodeKind::BoundBinaryExpression(
                BoundBinaryExpression {
                    left: Box::from(BoundNodeKind::BoundVariableExpression(
                        BoundVariableExpression::new(
                            statment.variable.clone().name,
                            statment.variable.clone()._type,
                        ),
                    )),
                    bound_binary_operator_kind: Box::from(BoundBinaryOperator::bind(
                        SyntaxKind::PlusToken,
                        Type::Number,
                        Type::Number,
                    ).unwrap()),
                    right: Box::from(BoundNodeKind::BoundLiteralExpression(
                        BoundLiteralExpression {
                            value_num: Some(1),
                            value_bool: None,
                            value_string: None
                        },
                    )),
                },
            )),
        });

        let while_block = BoundNodeKind::BoundBlockStatement(BoundBlockStatement::new(vec![
            *statment.body,
            increment,
        ]));
        let while_statement =
            BoundNodeKind::BoundWhileStatment(BoundWhileStatment::new(condition, while_block));
        let result = BoundNodeKind::BoundBlockStatement(BoundBlockStatement::new(vec![
            variable_declaration,
            while_statement,
        ]));

        return self.rewrite_statment(result);
    }

    pub fn rewrite_variable_declaration(
        &mut self,
        statment: BoundVariableDeclarion,
    ) -> BoundNodeKind {
        let initializer = self.rewrite_statment(*statment.initializer);
        BoundNodeKind::BoundVariableDeclaration(BoundVariableDeclarion::new(
            statment.variable,
            initializer,
        ))
    }

    pub fn rewrite_expression(&self, bound_statment: BoundNodeKind) -> BoundNodeKind {
        match bound_statment {
            BoundNodeKind::BoundErrorExpression(bound_error_experssion) => {
                self.rewrite_errror_expression(bound_error_experssion)
            }
            BoundNodeKind::BoundExpressionStatement(bound_expression_statment) => {
                self.rewrite_expression_statment(bound_expression_statment)
            }
            BoundNodeKind::BoundLiteralExpression(bound_literal_expression) => {
                self.rewrite_bound_literal_expression(bound_literal_expression)
            }
            BoundNodeKind::BoundBinaryExpression(bound_binary_expression) => {
                self.rewrite_bound_binary_expression(bound_binary_expression)
            }
            BoundNodeKind::BoundUnaryExpressionNode(bound_unary_expression) => {
                self.rewrite_bound_unary_expression(bound_unary_expression)
            }
            BoundNodeKind::BoundVariableExpression(bound_variable_expression) => {
                self.rewrite_bound_variable_expression(bound_variable_expression)
            }
            BoundNodeKind::BoundAssignmentExpression(bound_assignment_expression) => {
                self.rewrite_bound_assignment_expression(bound_assignment_expression)
            }
            _ => panic!(),
        }
    }

    pub fn rewrite_errror_expression(&self, expression: BoundErrorExpression) -> BoundNodeKind {
        return BoundNodeKind::BoundErrorExpression(expression)
    }

    pub fn rewrite_expression_statment(
        &self,
        expression: BoundExpressionStatement,
    ) -> BoundNodeKind {
        BoundNodeKind::BoundExpressionStatement(expression)
    }

    pub fn rewrite_bound_literal_expression(
        &self,
        expression: BoundLiteralExpression,
    ) -> BoundNodeKind {
        BoundNodeKind::BoundLiteralExpression(expression)
    }

    pub fn rewrite_bound_binary_expression(
        &self,
        expression: BoundBinaryExpression,
    ) -> BoundNodeKind {
        let left = self.rewrite_expression(*expression.left);
        let right = self.rewrite_expression(*expression.right);
        return BoundNodeKind::BoundBinaryExpression(BoundBinaryExpression {
            left: Box::from(left),
            right: Box::from(right),
            bound_binary_operator_kind: expression.bound_binary_operator_kind,
        });
    }

    pub fn rewrite_bound_unary_expression(
        &self,
        expression: BoundUnaryExpressionNode,
    ) -> BoundNodeKind {
        let operand = self.rewrite_expression(*expression.operand);
        return BoundNodeKind::BoundUnaryExpressionNode(BoundUnaryExpressionNode {
            operator_kind: expression.operator_kind,
            operand: Box::from(operand),
        });
    }

    pub fn rewrite_bound_variable_expression(
        &self,
        expression: BoundVariableExpression,
    ) -> BoundNodeKind {
        BoundNodeKind::BoundVariableExpression(expression)
    }

    pub fn rewrite_bound_assignment_expression(
        &self,
        expression: BoundAssignmentExpression,
    ) -> BoundNodeKind {
        let expression_new = self.rewrite_expression(*expression.bound_expression);
        return BoundNodeKind::BoundAssignmentExpression(BoundAssignmentExpression {
            name: expression.name,
            bound_expression: Box::from(expression_new),
        });
    }
}
