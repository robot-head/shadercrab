use super::builtins;
use super::value::Value;
use naga::{
    BinaryOperator, Expression, Function, Handle, Literal, MathFunction, Module,
    ScalarKind, Statement, Type, TypeInner, UnaryOperator,
};
use std::collections::HashMap;

/// GLSL interpreter that executes Naga IR on the CPU
pub struct GlslInterpreter<'a> {
    module: &'a Module,
    /// Expression cache for current function
    expr_cache: HashMap<Handle<Expression>, Value>,
    /// Local variable storage
    locals: HashMap<Handle<Expression>, Value>,
    /// Function arguments
    args: Vec<Value>,
    /// Current function handle (for expression arena access)
    current_func: Option<Handle<Function>>,
}

/// Control flow signal for early returns
enum ControlFlow {
    Continue,
    Return(Option<Value>),
    Break,
}

impl<'a> GlslInterpreter<'a> {
    pub fn new(module: &'a Module) -> Self {
        Self {
            module,
            expr_cache: HashMap::new(),
            locals: HashMap::new(),
            args: Vec::new(),
            current_func: None,
        }
    }

    /// Find the mainSound function handle
    pub fn find_main_sound(&self) -> Option<Handle<Function>> {
        for (handle, func) in self.module.functions.iter() {
            if func.name.as_deref() == Some("mainSound") {
                return Some(handle);
            }
        }
        None
    }

    /// Call mainSound(time) and return the result
    pub fn call_main_sound(&mut self, time: f32) -> [f32; 2] {
        let func_handle = self
            .find_main_sound()
            .expect("mainSound function not found");

        let result = self.call_function(func_handle, vec![Value::Float(time)]);

        match result {
            Some(Value::Vec2(v)) => v,
            Some(v) => {
                eprintln!("Warning: mainSound returned {:?}, expected vec2", v);
                [0.0, 0.0]
            }
            None => [0.0, 0.0],
        }
    }

    /// Call a function with arguments
    fn call_function(
        &mut self,
        func_handle: Handle<Function>,
        arguments: Vec<Value>,
    ) -> Option<Value> {
        // Save state
        let old_cache = std::mem::take(&mut self.expr_cache);
        let old_locals = std::mem::take(&mut self.locals);
        let old_args = std::mem::replace(&mut self.args, arguments);
        let old_func = self.current_func.replace(func_handle);

        let func = &self.module.functions[func_handle];
        
        // Initialize local variables with their initializers
        for (lv_handle, local_var) in func.local_variables.iter() {
            if let Some(init_expr) = local_var.init {
                // The init expression might be in global_expressions (constants) or 
                // function expressions (for non-constant initializers)
                let init_val = if (init_expr.index() as usize) < self.module.global_expressions.len() {
                    self.evaluate_const_expression(init_expr)
                } else {
                    // Evaluate from function expressions arena
                    self.evaluate_expression(init_expr)
                };
                
                // Find the expression handle that references this local variable
                for (expr_handle, expr) in func.expressions.iter() {
                    if let Expression::LocalVariable(lv) = expr {
                        if *lv == lv_handle {
                            self.locals.insert(expr_handle, init_val);
                            break;
                        }
                    }
                }
            }
        }
        
        // Execute function body
        let body = func.body.clone();
        let result = self.execute_block(&body);

        // Restore state
        self.expr_cache = old_cache;
        self.locals = old_locals;
        self.args = old_args;
        self.current_func = old_func;

        match result {
            ControlFlow::Return(v) => v,
            _ => None,
        }
    }

    /// Execute a block of statements
    fn execute_block(&mut self, block: &naga::Block) -> ControlFlow {
        for stmt in block.iter() {
            match self.execute_statement(stmt) {
                ControlFlow::Continue => {}
                cf => return cf,
            }
        }
        ControlFlow::Continue
    }

    /// Execute a single statement
    fn execute_statement(&mut self, stmt: &Statement) -> ControlFlow {
        match stmt {
            Statement::Emit(range) => {
                // Evaluate all expressions in the range
                for handle in range.clone() {
                    self.evaluate_expression(handle);
                }
                ControlFlow::Continue
            }

            Statement::Block(block) => self.execute_block(block),

            Statement::If {
                condition,
                accept,
                reject,
            } => {
                let cond = self.evaluate_expression(*condition).as_bool();
                if cond {
                    self.execute_block(accept)
                } else {
                    self.execute_block(reject)
                }
            }

            Statement::Switch { selector, cases } => {
                let sel = self.evaluate_expression(*selector).as_int();
                for case in cases {
                    let matches = match case.value {
                        naga::SwitchValue::I32(v) => sel == v,
                        naga::SwitchValue::U32(v) => sel as u32 == v,
                        naga::SwitchValue::Default => true,
                    };
                    if matches {
                        match self.execute_block(&case.body) {
                            ControlFlow::Break => return ControlFlow::Continue, // break from switch
                            ControlFlow::Return(v) => return ControlFlow::Return(v),
                            ControlFlow::Continue => {
                                if case.fall_through {
                                    continue;
                                } else {
                                    return ControlFlow::Continue;
                                }
                            }
                        }
                    }
                }
                ControlFlow::Continue
            }

            Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                loop {
                    match self.execute_block(body) {
                        ControlFlow::Break => return ControlFlow::Continue,
                        ControlFlow::Return(v) => return ControlFlow::Return(v),
                        ControlFlow::Continue => {}
                    }

                    match self.execute_block(continuing) {
                        ControlFlow::Break => return ControlFlow::Continue,
                        ControlFlow::Return(v) => return ControlFlow::Return(v),
                        ControlFlow::Continue => {}
                    }

                    if let Some(break_cond) = break_if {
                        if self.evaluate_expression(*break_cond).as_bool() {
                            return ControlFlow::Continue;
                        }
                    }
                }
            }

            Statement::Break => ControlFlow::Break,

            Statement::Continue => ControlFlow::Continue,

            Statement::Return { value } => {
                let val = value.map(|h| self.evaluate_expression(h));
                ControlFlow::Return(val)
            }

            Statement::Store { pointer, value } => {
                let val = self.evaluate_expression(*value);
                self.store_to_pointer(*pointer, val);
                ControlFlow::Continue
            }

            Statement::Call {
                function,
                arguments,
                result,
            } => {
                let args: Vec<Value> = arguments
                    .iter()
                    .map(|h| self.evaluate_expression(*h))
                    .collect();
                let ret = self.call_function(*function, args);
                if let Some(result_handle) = result {
                    if let Some(val) = ret {
                        self.expr_cache.insert(*result_handle, val);
                    }
                }
                ControlFlow::Continue
            }

            Statement::Atomic { .. } => {
                eprintln!("Warning: Atomic operations not supported in interpreter");
                ControlFlow::Continue
            }

            Statement::WorkGroupUniformLoad { .. } => {
                eprintln!("Warning: WorkGroupUniformLoad not supported in interpreter");
                ControlFlow::Continue
            }

            Statement::ImageStore { .. } => {
                eprintln!("Warning: ImageStore not supported in interpreter");
                ControlFlow::Continue
            }

            Statement::Barrier(_) => ControlFlow::Continue,

            Statement::RayQuery { .. } => {
                eprintln!("Warning: RayQuery not supported in interpreter");
                ControlFlow::Continue
            }

            Statement::SubgroupBallot { .. }
            | Statement::SubgroupCollectiveOperation { .. }
            | Statement::SubgroupGather { .. } => {
                eprintln!("Warning: Subgroup operations not supported in interpreter");
                ControlFlow::Continue
            }

            Statement::Kill => ControlFlow::Return(None),
        }
    }

    /// Store a value to a pointer expression
    fn store_to_pointer(&mut self, pointer: Handle<Expression>, value: Value) {
        let func_handle = self.current_func.expect("No current function");
        let func = &self.module.functions[func_handle];
        let expr = &func.expressions[pointer];

        match expr {
            Expression::LocalVariable(_) => {
                self.locals.insert(pointer, value);
            }
            Expression::AccessIndex { base, index } => {
                let base_copy = *base;
                let index_copy = *index;
                let base_val = self.evaluate_expression(base_copy);
                let new_val = base_val.with_component(index_copy as usize, value.as_float());
                self.store_to_pointer(base_copy, new_val);
            }
            Expression::Access { base, index } => {
                let base_copy = *base;
                let index_copy = *index;
                let idx = self.evaluate_expression(index_copy).as_int() as usize;
                let base_val = self.evaluate_expression(base_copy);
                let new_val = base_val.with_component(idx, value.as_float());
                self.store_to_pointer(base_copy, new_val);
            }
            _ => {
                eprintln!("Warning: Cannot store to expression {:?}", expr);
            }
        }
    }

    /// Evaluate an expression and return its value
    fn evaluate_expression(&mut self, handle: Handle<Expression>) -> Value {
        // Check cache first
        if let Some(val) = self.expr_cache.get(&handle) {
            return *val;
        }

        let func_handle = self.current_func.expect("No current function");
        let func = &self.module.functions[func_handle];
        let expr = func.expressions[handle].clone();
        let value = self.evaluate_expression_inner(&expr, handle);

        self.expr_cache.insert(handle, value);
        value
    }

    fn evaluate_expression_inner(
        &mut self,
        expr: &Expression,
        handle: Handle<Expression>,
    ) -> Value {
        match expr {
            Expression::Literal(lit) => match lit {
                Literal::F32(f) => Value::Float(*f),
                Literal::F64(f) => Value::Float(*f as f32),
                Literal::I32(i) => Value::Int(*i),
                Literal::U32(u) => Value::UInt(*u),
                Literal::Bool(b) => Value::Bool(*b),
                Literal::I64(i) => Value::Int(*i as i32),
                Literal::U64(u) => Value::UInt(*u as u32),
                Literal::AbstractInt(i) => Value::Int(*i as i32),
                Literal::AbstractFloat(f) => Value::Float(*f as f32),
            },

            Expression::Constant(const_handle) => {
                let constant = &self.module.constants[*const_handle];
                self.evaluate_const_expression(constant.init)
            }

            Expression::Override(_) => {
                eprintln!("Warning: Override expressions not supported");
                Value::Float(0.0)
            }

            Expression::ZeroValue(ty_handle) => self.zero_value_for_type(*ty_handle),

            Expression::Compose { ty, components } => {
                let values: Vec<Value> = components.iter().map(|h| self.evaluate_expression(*h)).collect();
                self.compose_value(*ty, &values)
            }

            Expression::Access { base, index } => {
                let base_val = self.evaluate_expression(*base);
                let idx = self.evaluate_expression(*index).as_int() as usize;
                base_val.component(idx)
            }

            Expression::AccessIndex { base, index } => {
                let base_val = self.evaluate_expression(*base);
                base_val.component(*index as usize)
            }

            Expression::Splat { size, value } => {
                let v = self.evaluate_expression(*value).as_float();
                match size {
                    naga::VectorSize::Bi => Value::Vec2([v, v]),
                    naga::VectorSize::Tri => Value::Vec3([v, v, v]),
                    naga::VectorSize::Quad => Value::Vec4([v, v, v, v]),
                }
            }

            Expression::Swizzle {
                size: _,
                vector,
                pattern,
            } => {
                let vec_val = self.evaluate_expression(*vector);
                vec_val.swizzle(pattern)
            }

            Expression::FunctionArgument(idx) => self.args.get(*idx as usize).copied().unwrap_or(Value::Float(0.0)),

            Expression::GlobalVariable(_) => {
                // Global variables in sound shaders are typically uniforms
                // For now, return zero - in full implementation we'd look these up
                Value::Float(0.0)
            }

            Expression::LocalVariable(_) => {
                // Look up in locals map using the handle
                self.locals.get(&handle).copied().unwrap_or(Value::Float(0.0))
            }

            Expression::Load { pointer } => {
                // First get what kind of expression this pointer is
                let func_handle = self.current_func.expect("No current function");
                let func = &self.module.functions[func_handle];
                let ptr_expr = func.expressions[*pointer].clone();
                
                match &ptr_expr {
                    Expression::LocalVariable(_) => {
                        self.locals.get(pointer).copied().unwrap_or(Value::Float(0.0))
                    }
                    Expression::AccessIndex { base, index } => {
                        let base_val = self.evaluate_expression(*base);
                        base_val.component(*index as usize)
                    }
                    Expression::Access { base, index } => {
                        let base_val = self.evaluate_expression(*base);
                        let idx = self.evaluate_expression(*index).as_int() as usize;
                        base_val.component(idx)
                    }
                    _ => self.evaluate_expression(*pointer),
                }
            }

            Expression::Unary { op, expr } => {
                let val = self.evaluate_expression(*expr);
                match op {
                    UnaryOperator::Negate => -val,
                    UnaryOperator::LogicalNot => Value::Bool(!val.as_bool()),
                    UnaryOperator::BitwiseNot => builtins::bitwise_not(val),
                }
            }

            Expression::Binary { op, left, right } => {
                let l = self.evaluate_expression(*left);
                let r = self.evaluate_expression(*right);
                self.eval_binary_op(*op, l, r)
            }

            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                let cond = self.evaluate_expression(*condition).as_bool();
                if cond {
                    self.evaluate_expression(*accept)
                } else {
                    self.evaluate_expression(*reject)
                }
            }

            Expression::Relational { fun, argument } => {
                let val = self.evaluate_expression(*argument);
                match fun {
                    naga::RelationalFunction::All => {
                        // All components are true
                        match val {
                            Value::Bool(b) => Value::Bool(b),
                            _ => Value::Bool(true), // For vectors, would check all
                        }
                    }
                    naga::RelationalFunction::Any => {
                        match val {
                            Value::Bool(b) => Value::Bool(b),
                            _ => Value::Bool(true),
                        }
                    }
                    naga::RelationalFunction::IsNan => {
                        Value::Bool(val.as_float().is_nan())
                    }
                    naga::RelationalFunction::IsInf => {
                        Value::Bool(val.as_float().is_infinite())
                    }
                }
            }

            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3: _,
            } => {
                let a = self.evaluate_expression(*arg);
                let b = arg1.map(|h| self.evaluate_expression(h));
                let c = arg2.map(|h| self.evaluate_expression(h));
                self.eval_math_function(*fun, a, b, c)
            }

            Expression::As {
                expr,
                kind,
                convert,
            } => {
                let val = self.evaluate_expression(*expr);
                if convert.is_some() {
                    self.convert_value(val, *kind)
                } else {
                    self.bitcast_value(val, *kind)
                }
            }

            Expression::CallResult(_) => {
                // Result should have been cached by the Call statement
                self.expr_cache.get(&handle).copied().unwrap_or(Value::Float(0.0))
            }

            Expression::AtomicResult { .. } => {
                eprintln!("Warning: AtomicResult not supported");
                Value::Int(0)
            }

            Expression::ArrayLength(_) => {
                eprintln!("Warning: ArrayLength not fully supported");
                Value::UInt(0)
            }

            Expression::ImageSample { .. }
            | Expression::ImageLoad { .. }
            | Expression::ImageQuery { .. } => {
                eprintln!("Warning: Image operations not supported in interpreter");
                Value::Vec4([0.0, 0.0, 0.0, 1.0])
            }

            Expression::Derivative { .. } => {
                // Derivatives are GPU-only, return 0
                Value::Float(0.0)
            }

            Expression::RayQueryProceedResult
            | Expression::RayQueryGetIntersection { .. } => {
                eprintln!("Warning: RayQuery not supported");
                Value::Bool(false)
            }

            Expression::SubgroupBallotResult
            | Expression::SubgroupOperationResult { .. } => {
                eprintln!("Warning: Subgroup operations not supported");
                Value::UInt(0)
            }

            Expression::WorkGroupUniformLoadResult { .. } => {
                eprintln!("Warning: WorkGroupUniformLoad not supported");
                Value::Float(0.0)
            }
        }
    }

    /// Evaluate a constant expression
    fn evaluate_const_expression(&self, handle: Handle<Expression>) -> Value {
        let expr = &self.module.global_expressions[handle];
        match expr {
            Expression::Literal(lit) => match lit {
                Literal::F32(f) => Value::Float(*f),
                Literal::F64(f) => Value::Float(*f as f32),
                Literal::I32(i) => Value::Int(*i),
                Literal::U32(u) => Value::UInt(*u),
                Literal::Bool(b) => Value::Bool(*b),
                Literal::I64(i) => Value::Int(*i as i32),
                Literal::U64(u) => Value::UInt(*u as u32),
                Literal::AbstractInt(i) => Value::Int(*i as i32),
                Literal::AbstractFloat(f) => Value::Float(*f as f32),
            },
            Expression::Compose { ty, components } => {
                let values: Vec<Value> = components
                    .iter()
                    .map(|h| self.evaluate_const_expression(*h))
                    .collect();
                self.compose_value(*ty, &values)
            }
            Expression::ZeroValue(ty) => self.zero_value_for_type(*ty),
            _ => {
                eprintln!("Warning: Unsupported constant expression {:?}", expr);
                Value::Float(0.0)
            }
        }
    }

    /// Create a zero value for a given type
    fn zero_value_for_type(&self, ty_handle: Handle<Type>) -> Value {
        let ty = &self.module.types[ty_handle];
        match &ty.inner {
            TypeInner::Scalar(scalar) => match scalar.kind {
                ScalarKind::Float => Value::Float(0.0),
                ScalarKind::Sint => Value::Int(0),
                ScalarKind::Uint => Value::UInt(0),
                ScalarKind::Bool => Value::Bool(false),
                ScalarKind::AbstractInt => Value::Int(0),
                ScalarKind::AbstractFloat => Value::Float(0.0),
            },
            TypeInner::Vector { size, scalar } => {
                let zero = match scalar.kind {
                    ScalarKind::Float | ScalarKind::AbstractFloat => 0.0f32,
                    _ => 0.0,
                };
                match size {
                    naga::VectorSize::Bi => Value::Vec2([zero, zero]),
                    naga::VectorSize::Tri => Value::Vec3([zero, zero, zero]),
                    naga::VectorSize::Quad => Value::Vec4([zero, zero, zero, zero]),
                }
            }
            _ => Value::Float(0.0),
        }
    }

    /// Compose a vector/struct value from components
    fn compose_value(&self, ty_handle: Handle<Type>, values: &[Value]) -> Value {
        let ty = &self.module.types[ty_handle];
        match &ty.inner {
            TypeInner::Vector { size, .. } => {
                // Flatten all input values into floats
                let mut floats = Vec::new();
                for v in values {
                    match v {
                        Value::Float(f) => floats.push(*f),
                        Value::Vec2(arr) => floats.extend_from_slice(arr),
                        Value::Vec3(arr) => floats.extend_from_slice(arr),
                        Value::Vec4(arr) => floats.extend_from_slice(arr),
                        Value::Int(i) => floats.push(*i as f32),
                        Value::UInt(u) => floats.push(*u as f32),
                        _ => floats.push(0.0),
                    }
                }
                match size {
                    naga::VectorSize::Bi => {
                        Value::Vec2([floats.get(0).copied().unwrap_or(0.0), floats.get(1).copied().unwrap_or(0.0)])
                    }
                    naga::VectorSize::Tri => Value::Vec3([
                        floats.get(0).copied().unwrap_or(0.0),
                        floats.get(1).copied().unwrap_or(0.0),
                        floats.get(2).copied().unwrap_or(0.0),
                    ]),
                    naga::VectorSize::Quad => Value::Vec4([
                        floats.get(0).copied().unwrap_or(0.0),
                        floats.get(1).copied().unwrap_or(0.0),
                        floats.get(2).copied().unwrap_or(0.0),
                        floats.get(3).copied().unwrap_or(0.0),
                    ]),
                }
            }
            TypeInner::Scalar(_) => values.first().copied().unwrap_or(Value::Float(0.0)),
            _ => values.first().copied().unwrap_or(Value::Float(0.0)),
        }
    }

    /// Evaluate a binary operator
    fn eval_binary_op(&self, op: BinaryOperator, left: Value, right: Value) -> Value {
        match op {
            BinaryOperator::Add => left + right,
            BinaryOperator::Subtract => left - right,
            BinaryOperator::Multiply => left * right,
            BinaryOperator::Divide => left / right,
            BinaryOperator::Modulo => builtins::glsl_mod(left, right),
            BinaryOperator::Equal => Value::Bool(left == right),
            BinaryOperator::NotEqual => Value::Bool(left != right),
            BinaryOperator::Less => Value::Bool(left < right),
            BinaryOperator::LessEqual => Value::Bool(left <= right),
            BinaryOperator::Greater => Value::Bool(left > right),
            BinaryOperator::GreaterEqual => Value::Bool(left >= right),
            BinaryOperator::And => builtins::bitwise_and(left, right),
            BinaryOperator::InclusiveOr => builtins::bitwise_or(left, right),
            BinaryOperator::ExclusiveOr => builtins::bitwise_xor(left, right),
            BinaryOperator::LogicalAnd => Value::Bool(left.as_bool() && right.as_bool()),
            BinaryOperator::LogicalOr => Value::Bool(left.as_bool() || right.as_bool()),
            BinaryOperator::ShiftLeft => builtins::shift_left(left, right),
            BinaryOperator::ShiftRight => builtins::shift_right(left, right),
        }
    }

    /// Evaluate a math function
    fn eval_math_function(
        &self,
        fun: MathFunction,
        arg: Value,
        arg1: Option<Value>,
        arg2: Option<Value>,
    ) -> Value {
        match fun {
            // Trigonometric
            MathFunction::Sin => builtins::sin(arg),
            MathFunction::Cos => builtins::cos(arg),
            MathFunction::Tan => builtins::tan(arg),
            MathFunction::Asin => builtins::asin(arg),
            MathFunction::Acos => builtins::acos(arg),
            MathFunction::Atan => builtins::atan(arg),
            MathFunction::Atan2 => builtins::atan2(arg, arg1.unwrap()),
            MathFunction::Sinh => builtins::sinh(arg),
            MathFunction::Cosh => builtins::cosh(arg),
            MathFunction::Tanh => builtins::tanh(arg),
            MathFunction::Asinh => builtins::asinh(arg),
            MathFunction::Acosh => builtins::acosh(arg),
            MathFunction::Atanh => builtins::atanh(arg),

            // Exponential
            MathFunction::Exp => builtins::exp(arg),
            MathFunction::Exp2 => builtins::exp2(arg),
            MathFunction::Log => builtins::log(arg),
            MathFunction::Log2 => builtins::log2(arg),
            MathFunction::Pow => builtins::pow(arg, arg1.unwrap()),
            MathFunction::Sqrt => builtins::sqrt(arg),
            MathFunction::InverseSqrt => builtins::inversesqrt(arg),

            // Common
            MathFunction::Abs => builtins::abs(arg),
            MathFunction::Sign => builtins::sign(arg),
            MathFunction::Floor => builtins::floor(arg),
            MathFunction::Ceil => builtins::ceil(arg),
            MathFunction::Round => builtins::round(arg),
            MathFunction::Trunc => builtins::trunc(arg),
            MathFunction::Fract => builtins::fract(arg),
            MathFunction::Min => builtins::min(arg, arg1.unwrap()),
            MathFunction::Max => builtins::max(arg, arg1.unwrap()),
            MathFunction::Clamp => builtins::clamp(arg, arg1.unwrap(), arg2.unwrap()),
            MathFunction::Saturate => builtins::saturate(arg),
            MathFunction::Mix => builtins::mix(arg, arg1.unwrap(), arg2.unwrap()),
            MathFunction::Step => builtins::step(arg, arg1.unwrap()),
            MathFunction::SmoothStep => builtins::smoothstep(arg, arg1.unwrap(), arg2.unwrap()),

            // Geometric
            MathFunction::Length => builtins::length(arg),
            MathFunction::Distance => builtins::distance(arg, arg1.unwrap()),
            MathFunction::Dot => builtins::dot(arg, arg1.unwrap()),
            MathFunction::Cross => builtins::cross(arg, arg1.unwrap()),
            MathFunction::Normalize => builtins::normalize(arg),
            MathFunction::FaceForward => {
                builtins::faceforward(arg, arg1.unwrap(), arg2.unwrap())
            }
            MathFunction::Reflect => builtins::reflect(arg, arg1.unwrap()),
            MathFunction::Refract => builtins::refract(arg, arg1.unwrap(), arg2.unwrap()),

            // Other
            MathFunction::Radians => {
                let deg = arg.as_float();
                Value::Float(deg.to_radians())
            }
            MathFunction::Degrees => {
                let rad = arg.as_float();
                Value::Float(rad.to_degrees())
            }
            MathFunction::Modf | MathFunction::Frexp | MathFunction::Ldexp => {
                eprintln!("Warning: {:?} not fully supported", fun);
                arg
            }
            MathFunction::Outer | MathFunction::Transpose | MathFunction::Determinant | MathFunction::Inverse => {
                eprintln!("Warning: Matrix operations not fully supported");
                arg
            }
            MathFunction::CountTrailingZeros
            | MathFunction::CountLeadingZeros
            | MathFunction::CountOneBits
            | MathFunction::ReverseBits
            | MathFunction::FirstTrailingBit
            | MathFunction::FirstLeadingBit => {
                eprintln!("Warning: Bit operations not fully supported");
                Value::Int(0)
            }
            MathFunction::ExtractBits | MathFunction::InsertBits => {
                eprintln!("Warning: Bit extraction/insertion not supported");
                Value::Int(0)
            }
            MathFunction::Pack2x16float
            | MathFunction::Pack2x16snorm
            | MathFunction::Pack2x16unorm
            | MathFunction::Pack4x8snorm
            | MathFunction::Pack4x8unorm
            | MathFunction::Pack4xI8
            | MathFunction::Pack4xU8
            | MathFunction::Unpack2x16float
            | MathFunction::Unpack2x16snorm
            | MathFunction::Unpack2x16unorm
            | MathFunction::Unpack4x8snorm
            | MathFunction::Unpack4x8unorm
            | MathFunction::Unpack4xI8
            | MathFunction::Unpack4xU8 => {
                eprintln!("Warning: Pack/unpack operations not supported");
                Value::UInt(0)
            }
            // Catch-all for any other functions
            _ => {
                eprintln!("Warning: Math function {:?} not supported", fun);
                arg
            }
        }
    }

    /// Convert a value to a different scalar kind
    fn convert_value(&self, val: Value, kind: ScalarKind) -> Value {
        match kind {
            ScalarKind::Float | ScalarKind::AbstractFloat => Value::Float(val.as_float()),
            ScalarKind::Sint | ScalarKind::AbstractInt => Value::Int(val.as_int()),
            ScalarKind::Uint => Value::UInt(val.as_uint()),
            ScalarKind::Bool => Value::Bool(val.as_bool()),
        }
    }

    /// Bitcast a value (reinterpret bits)
    fn bitcast_value(&self, val: Value, kind: ScalarKind) -> Value {
        match (val, kind) {
            (Value::Float(f), ScalarKind::Sint | ScalarKind::AbstractInt) => {
                Value::Int(f.to_bits() as i32)
            }
            (Value::Float(f), ScalarKind::Uint) => Value::UInt(f.to_bits()),
            (Value::Int(i), ScalarKind::Float | ScalarKind::AbstractFloat) => {
                Value::Float(f32::from_bits(i as u32))
            }
            (Value::UInt(u), ScalarKind::Float | ScalarKind::AbstractFloat) => {
                Value::Float(f32::from_bits(u))
            }
            (Value::Int(i), ScalarKind::Uint) => Value::UInt(i as u32),
            (Value::UInt(u), ScalarKind::Sint | ScalarKind::AbstractInt) => Value::Int(u as i32),
            _ => val,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_operations() {
        let a = Value::Float(2.0);
        let b = Value::Float(3.0);
        assert_eq!((a + b).as_float(), 5.0);
        assert_eq!((a * b).as_float(), 6.0);
    }

    #[test]
    fn test_vec_operations() {
        let a = Value::Vec2([1.0, 2.0]);
        let b = Value::Vec2([3.0, 4.0]);
        let sum = a + b;
        match sum {
            Value::Vec2(v) => assert_eq!(v, [4.0, 6.0]),
            _ => panic!("Expected Vec2"),
        }
    }
}
