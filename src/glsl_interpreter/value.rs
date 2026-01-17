use std::ops::{Add, Div, Mul, Neg, Sub};

/// Runtime value for GLSL interpreter
#[derive(Debug, Clone, Copy)]
pub enum Value {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Int(i32),
    UInt(u32),
    Bool(bool),
}

impl Value {
    /// Get as f32, panics if not a float
    pub fn as_float(&self) -> f32 {
        match self {
            Value::Float(f) => *f,
            Value::Int(i) => *i as f32,
            Value::UInt(u) => *u as f32,
            _ => panic!("Expected float, got {:?}", self),
        }
    }

    /// Get as int, panics if not an int
    pub fn as_int(&self) -> i32 {
        match self {
            Value::Int(i) => *i,
            Value::Float(f) => *f as i32,
            Value::UInt(u) => *u as i32,
            _ => panic!("Expected int, got {:?}", self),
        }
    }

    /// Get as uint, panics if not a uint
    pub fn as_uint(&self) -> u32 {
        match self {
            Value::UInt(u) => *u,
            Value::Int(i) => *i as u32,
            Value::Float(f) => *f as u32,
            _ => panic!("Expected uint, got {:?}", self),
        }
    }

    /// Get as bool, panics if not a bool
    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            _ => panic!("Expected bool, got {:?}", self),
        }
    }

    /// Get a component from a vector value
    pub fn component(&self, index: usize) -> Value {
        match self {
            Value::Vec2(v) => Value::Float(v[index]),
            Value::Vec3(v) => Value::Float(v[index]),
            Value::Vec4(v) => Value::Float(v[index]),
            Value::Float(f) => Value::Float(*f), // Scalar splatting
            _ => panic!("Cannot get component from {:?}", self),
        }
    }

    /// Set a component in a vector value (returns new value)
    pub fn with_component(self, index: usize, val: f32) -> Value {
        match self {
            Value::Vec2(mut v) => {
                v[index] = val;
                Value::Vec2(v)
            }
            Value::Vec3(mut v) => {
                v[index] = val;
                Value::Vec3(v)
            }
            Value::Vec4(mut v) => {
                v[index] = val;
                Value::Vec4(v)
            }
            _ => panic!("Cannot set component on {:?}", self),
        }
    }

    /// Swizzle operation for vectors
    pub fn swizzle(&self, pattern: &[naga::SwizzleComponent]) -> Value {
        let get_comp = |c: naga::SwizzleComponent| -> f32 {
            let idx = match c {
                naga::SwizzleComponent::X => 0,
                naga::SwizzleComponent::Y => 1,
                naga::SwizzleComponent::Z => 2,
                naga::SwizzleComponent::W => 3,
            };
            self.component(idx).as_float()
        };

        match pattern.len() {
            1 => Value::Float(get_comp(pattern[0])),
            2 => Value::Vec2([get_comp(pattern[0]), get_comp(pattern[1])]),
            3 => Value::Vec3([
                get_comp(pattern[0]),
                get_comp(pattern[1]),
                get_comp(pattern[2]),
            ]),
            4 => Value::Vec4([
                get_comp(pattern[0]),
                get_comp(pattern[1]),
                get_comp(pattern[2]),
                get_comp(pattern[3]),
            ]),
            _ => panic!("Invalid swizzle pattern length"),
        }
    }

    /// Create a zero value of the same type
    pub fn zero_like(&self) -> Value {
        match self {
            Value::Float(_) => Value::Float(0.0),
            Value::Vec2(_) => Value::Vec2([0.0; 2]),
            Value::Vec3(_) => Value::Vec3([0.0; 3]),
            Value::Vec4(_) => Value::Vec4([0.0; 4]),
            Value::Int(_) => Value::Int(0),
            Value::UInt(_) => Value::UInt(0),
            Value::Bool(_) => Value::Bool(false),
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::Float(0.0)
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (Value::Vec2(a), Value::Vec2(b)) => Value::Vec2([a[0] + b[0], a[1] + b[1]]),
            (Value::Vec3(a), Value::Vec3(b)) => {
                Value::Vec3([a[0] + b[0], a[1] + b[1], a[2] + b[2]])
            }
            (Value::Vec4(a), Value::Vec4(b)) => {
                Value::Vec4([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
            }
            (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
            (Value::UInt(a), Value::UInt(b)) => Value::UInt(a + b),
            _ => panic!("Cannot add {:?} and {:?}", self, rhs),
        }
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            (Value::Vec2(a), Value::Vec2(b)) => Value::Vec2([a[0] - b[0], a[1] - b[1]]),
            (Value::Vec3(a), Value::Vec3(b)) => {
                Value::Vec3([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
            }
            (Value::Vec4(a), Value::Vec4(b)) => {
                Value::Vec4([a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]])
            }
            (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
            (Value::UInt(a), Value::UInt(b)) => Value::UInt(a - b),
            _ => panic!("Cannot subtract {:?} and {:?}", self, rhs),
        }
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (Value::Vec2(a), Value::Vec2(b)) => Value::Vec2([a[0] * b[0], a[1] * b[1]]),
            (Value::Vec3(a), Value::Vec3(b)) => {
                Value::Vec3([a[0] * b[0], a[1] * b[1], a[2] * b[2]])
            }
            (Value::Vec4(a), Value::Vec4(b)) => {
                Value::Vec4([a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]])
            }
            // Scalar * Vector
            (Value::Float(s), Value::Vec2(v)) | (Value::Vec2(v), Value::Float(s)) => {
                Value::Vec2([v[0] * s, v[1] * s])
            }
            (Value::Float(s), Value::Vec3(v)) | (Value::Vec3(v), Value::Float(s)) => {
                Value::Vec3([v[0] * s, v[1] * s, v[2] * s])
            }
            (Value::Float(s), Value::Vec4(v)) | (Value::Vec4(v), Value::Float(s)) => {
                Value::Vec4([v[0] * s, v[1] * s, v[2] * s, v[3] * s])
            }
            (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
            (Value::UInt(a), Value::UInt(b)) => Value::UInt(a * b),
            _ => panic!("Cannot multiply {:?} and {:?}", self, rhs),
        }
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Float(a), Value::Float(b)) => Value::Float(a / b),
            (Value::Vec2(a), Value::Vec2(b)) => Value::Vec2([a[0] / b[0], a[1] / b[1]]),
            (Value::Vec3(a), Value::Vec3(b)) => {
                Value::Vec3([a[0] / b[0], a[1] / b[1], a[2] / b[2]])
            }
            (Value::Vec4(a), Value::Vec4(b)) => {
                Value::Vec4([a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]])
            }
            // Vector / Scalar
            (Value::Vec2(v), Value::Float(s)) => Value::Vec2([v[0] / s, v[1] / s]),
            (Value::Vec3(v), Value::Float(s)) => Value::Vec3([v[0] / s, v[1] / s, v[2] / s]),
            (Value::Vec4(v), Value::Float(s)) => {
                Value::Vec4([v[0] / s, v[1] / s, v[2] / s, v[3] / s])
            }
            (Value::Int(a), Value::Int(b)) => Value::Int(a / b),
            (Value::UInt(a), Value::UInt(b)) => Value::UInt(a / b),
            _ => panic!("Cannot divide {:?} by {:?}", self, rhs),
        }
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Value {
        match self {
            Value::Float(a) => Value::Float(-a),
            Value::Vec2(v) => Value::Vec2([-v[0], -v[1]]),
            Value::Vec3(v) => Value::Vec3([-v[0], -v[1], -v[2]]),
            Value::Vec4(v) => Value::Vec4([-v[0], -v[1], -v[2], -v[3]]),
            Value::Int(a) => Value::Int(-a),
            _ => panic!("Cannot negate {:?}", self),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Vec2(a), Value::Vec2(b)) => a == b,
            (Value::Vec3(a), Value::Vec3(b)) => a == b,
            (Value::Vec4(a), Value::Vec4(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::UInt(a), Value::UInt(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            _ => false,
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::Int(a), Value::Int(b)) => a.partial_cmp(b),
            (Value::UInt(a), Value::UInt(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}
