use super::value::Value;

/// Apply a unary f32 function component-wise to a value
fn apply_unary<F: Fn(f32) -> f32>(v: Value, f: F) -> Value {
    match v {
        Value::Float(x) => Value::Float(f(x)),
        Value::Vec2(arr) => Value::Vec2([f(arr[0]), f(arr[1])]),
        Value::Vec3(arr) => Value::Vec3([f(arr[0]), f(arr[1]), f(arr[2])]),
        Value::Vec4(arr) => Value::Vec4([f(arr[0]), f(arr[1]), f(arr[2]), f(arr[3])]),
        _ => panic!("Cannot apply unary function to {:?}", v),
    }
}

/// Apply a binary f32 function component-wise to two values
fn apply_binary<F: Fn(f32, f32) -> f32>(a: Value, b: Value, f: F) -> Value {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => Value::Float(f(x, y)),
        (Value::Vec2(x), Value::Vec2(y)) => Value::Vec2([f(x[0], y[0]), f(x[1], y[1])]),
        (Value::Vec3(x), Value::Vec3(y)) => {
            Value::Vec3([f(x[0], y[0]), f(x[1], y[1]), f(x[2], y[2])])
        }
        (Value::Vec4(x), Value::Vec4(y)) => {
            Value::Vec4([f(x[0], y[0]), f(x[1], y[1]), f(x[2], y[2]), f(x[3], y[3])])
        }
        // Scalar broadcast
        (Value::Float(s), Value::Vec2(v)) => Value::Vec2([f(s, v[0]), f(s, v[1])]),
        (Value::Vec2(v), Value::Float(s)) => Value::Vec2([f(v[0], s), f(v[1], s)]),
        (Value::Float(s), Value::Vec3(v)) => Value::Vec3([f(s, v[0]), f(s, v[1]), f(s, v[2])]),
        (Value::Vec3(v), Value::Float(s)) => Value::Vec3([f(v[0], s), f(v[1], s), f(v[2], s)]),
        (Value::Float(s), Value::Vec4(v)) => {
            Value::Vec4([f(s, v[0]), f(s, v[1]), f(s, v[2]), f(s, v[3])])
        }
        (Value::Vec4(v), Value::Float(s)) => {
            Value::Vec4([f(v[0], s), f(v[1], s), f(v[2], s), f(v[3], s)])
        }
        _ => panic!("Cannot apply binary function to {:?} and {:?}", a, b),
    }
}

/// Apply a ternary f32 function component-wise
fn apply_ternary<F: Fn(f32, f32, f32) -> f32>(a: Value, b: Value, c: Value, f: F) -> Value {
    match (a, b, c) {
        (Value::Float(x), Value::Float(y), Value::Float(z)) => Value::Float(f(x, y, z)),
        (Value::Vec2(x), Value::Vec2(y), Value::Vec2(z)) => {
            Value::Vec2([f(x[0], y[0], z[0]), f(x[1], y[1], z[1])])
        }
        (Value::Vec3(x), Value::Vec3(y), Value::Vec3(z)) => Value::Vec3([
            f(x[0], y[0], z[0]),
            f(x[1], y[1], z[1]),
            f(x[2], y[2], z[2]),
        ]),
        (Value::Vec4(x), Value::Vec4(y), Value::Vec4(z)) => Value::Vec4([
            f(x[0], y[0], z[0]),
            f(x[1], y[1], z[1]),
            f(x[2], y[2], z[2]),
            f(x[3], y[3], z[3]),
        ]),
        // Mix with scalar t
        (Value::Vec2(x), Value::Vec2(y), Value::Float(t)) => {
            Value::Vec2([f(x[0], y[0], t), f(x[1], y[1], t)])
        }
        (Value::Vec3(x), Value::Vec3(y), Value::Float(t)) => {
            Value::Vec3([f(x[0], y[0], t), f(x[1], y[1], t), f(x[2], y[2], t)])
        }
        (Value::Vec4(x), Value::Vec4(y), Value::Float(t)) => Value::Vec4([
            f(x[0], y[0], t),
            f(x[1], y[1], t),
            f(x[2], y[2], t),
            f(x[3], y[3], t),
        ]),
        // Clamp with scalar bounds
        (Value::Vec2(x), Value::Float(lo), Value::Float(hi)) => {
            Value::Vec2([f(x[0], lo, hi), f(x[1], lo, hi)])
        }
        (Value::Vec3(x), Value::Float(lo), Value::Float(hi)) => {
            Value::Vec3([f(x[0], lo, hi), f(x[1], lo, hi), f(x[2], lo, hi)])
        }
        (Value::Vec4(x), Value::Float(lo), Value::Float(hi)) => {
            Value::Vec4([f(x[0], lo, hi), f(x[1], lo, hi), f(x[2], lo, hi), f(x[3], lo, hi)])
        }
        _ => panic!(
            "Cannot apply ternary function to {:?}, {:?}, {:?}",
            a, b, c
        ),
    }
}

// ============================================================================
// Trigonometric functions
// ============================================================================

pub fn sin(v: Value) -> Value {
    apply_unary(v, f32::sin)
}

pub fn cos(v: Value) -> Value {
    apply_unary(v, f32::cos)
}

pub fn tan(v: Value) -> Value {
    apply_unary(v, f32::tan)
}

pub fn asin(v: Value) -> Value {
    apply_unary(v, f32::asin)
}

pub fn acos(v: Value) -> Value {
    apply_unary(v, f32::acos)
}

pub fn atan(v: Value) -> Value {
    apply_unary(v, f32::atan)
}

pub fn atan2(y: Value, x: Value) -> Value {
    apply_binary(y, x, f32::atan2)
}

pub fn sinh(v: Value) -> Value {
    apply_unary(v, f32::sinh)
}

pub fn cosh(v: Value) -> Value {
    apply_unary(v, f32::cosh)
}

pub fn tanh(v: Value) -> Value {
    apply_unary(v, f32::tanh)
}

pub fn asinh(v: Value) -> Value {
    apply_unary(v, f32::asinh)
}

pub fn acosh(v: Value) -> Value {
    apply_unary(v, f32::acosh)
}

pub fn atanh(v: Value) -> Value {
    apply_unary(v, f32::atanh)
}

// ============================================================================
// Exponential functions
// ============================================================================

pub fn exp(v: Value) -> Value {
    apply_unary(v, f32::exp)
}

pub fn exp2(v: Value) -> Value {
    apply_unary(v, f32::exp2)
}

pub fn log(v: Value) -> Value {
    apply_unary(v, f32::ln)
}

pub fn log2(v: Value) -> Value {
    apply_unary(v, f32::log2)
}

pub fn pow(base: Value, exp: Value) -> Value {
    apply_binary(base, exp, f32::powf)
}

pub fn sqrt(v: Value) -> Value {
    apply_unary(v, f32::sqrt)
}

pub fn inversesqrt(v: Value) -> Value {
    apply_unary(v, |x| 1.0 / x.sqrt())
}

// ============================================================================
// Common functions
// ============================================================================

pub fn abs(v: Value) -> Value {
    match v {
        Value::Int(x) => Value::Int(x.abs()),
        _ => apply_unary(v, f32::abs),
    }
}

pub fn sign(v: Value) -> Value {
    apply_unary(v, |x| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

pub fn floor(v: Value) -> Value {
    apply_unary(v, f32::floor)
}

pub fn ceil(v: Value) -> Value {
    apply_unary(v, f32::ceil)
}

pub fn round(v: Value) -> Value {
    apply_unary(v, f32::round)
}

pub fn trunc(v: Value) -> Value {
    apply_unary(v, f32::trunc)
}

pub fn fract(v: Value) -> Value {
    apply_unary(v, |x| x - x.floor())
}

pub fn glsl_mod(x: Value, y: Value) -> Value {
    apply_binary(x, y, |a, b| a - b * (a / b).floor())
}

pub fn min(a: Value, b: Value) -> Value {
    apply_binary(a, b, f32::min)
}

pub fn max(a: Value, b: Value) -> Value {
    apply_binary(a, b, f32::max)
}

pub fn clamp(x: Value, min_val: Value, max_val: Value) -> Value {
    apply_ternary(x, min_val, max_val, |v, lo, hi| v.clamp(lo, hi))
}

pub fn saturate(x: Value) -> Value {
    clamp(x, Value::Float(0.0), Value::Float(1.0))
}

pub fn mix(x: Value, y: Value, a: Value) -> Value {
    apply_ternary(x, y, a, |x, y, a| x * (1.0 - a) + y * a)
}

pub fn step(edge: Value, x: Value) -> Value {
    apply_binary(edge, x, |e, v| if v < e { 0.0 } else { 1.0 })
}

pub fn smoothstep(edge0: Value, edge1: Value, x: Value) -> Value {
    apply_ternary(edge0, edge1, x, |e0, e1, v| {
        let t = ((v - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    })
}

// ============================================================================
// Geometric functions
// ============================================================================

pub fn length(v: Value) -> Value {
    match v {
        Value::Float(x) => Value::Float(x.abs()),
        Value::Vec2(arr) => Value::Float((arr[0] * arr[0] + arr[1] * arr[1]).sqrt()),
        Value::Vec3(arr) => {
            Value::Float((arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2]).sqrt())
        }
        Value::Vec4(arr) => Value::Float(
            (arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2] + arr[3] * arr[3]).sqrt(),
        ),
        _ => panic!("Cannot compute length of {:?}", v),
    }
}

pub fn distance(a: Value, b: Value) -> Value {
    length(a - b)
}

pub fn dot(a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => Value::Float(x * y),
        (Value::Vec2(x), Value::Vec2(y)) => Value::Float(x[0] * y[0] + x[1] * y[1]),
        (Value::Vec3(x), Value::Vec3(y)) => {
            Value::Float(x[0] * y[0] + x[1] * y[1] + x[2] * y[2])
        }
        (Value::Vec4(x), Value::Vec4(y)) => {
            Value::Float(x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3])
        }
        _ => panic!("Cannot compute dot product of {:?} and {:?}", a, b),
    }
}

pub fn cross(a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::Vec3(x), Value::Vec3(y)) => Value::Vec3([
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ]),
        _ => panic!("cross requires vec3, got {:?} and {:?}", a, b),
    }
}

pub fn normalize(v: Value) -> Value {
    let len = length(v).as_float();
    if len == 0.0 {
        v
    } else {
        v / Value::Float(len)
    }
}

pub fn faceforward(n: Value, i: Value, nref: Value) -> Value {
    if dot(nref, i).as_float() < 0.0 {
        n
    } else {
        -n
    }
}

pub fn reflect(i: Value, n: Value) -> Value {
    let d = dot(n, i).as_float();
    i - n * Value::Float(2.0 * d)
}

pub fn refract(i: Value, n: Value, eta: Value) -> Value {
    let eta = eta.as_float();
    let dot_ni = dot(n, i).as_float();
    let k = 1.0 - eta * eta * (1.0 - dot_ni * dot_ni);
    if k < 0.0 {
        i.zero_like()
    } else {
        i * Value::Float(eta) - n * Value::Float(eta * dot_ni + k.sqrt())
    }
}

// ============================================================================
// Integer operations
// ============================================================================

pub fn bitwise_and(a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Value::Int(x & y),
        (Value::UInt(x), Value::UInt(y)) => Value::UInt(x & y),
        _ => panic!("Cannot bitwise AND {:?} and {:?}", a, b),
    }
}

pub fn bitwise_or(a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Value::Int(x | y),
        (Value::UInt(x), Value::UInt(y)) => Value::UInt(x | y),
        _ => panic!("Cannot bitwise OR {:?} and {:?}", a, b),
    }
}

pub fn bitwise_xor(a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Value::Int(x ^ y),
        (Value::UInt(x), Value::UInt(y)) => Value::UInt(x ^ y),
        _ => panic!("Cannot bitwise XOR {:?} and {:?}", a, b),
    }
}

pub fn bitwise_not(v: Value) -> Value {
    match v {
        Value::Int(x) => Value::Int(!x),
        Value::UInt(x) => Value::UInt(!x),
        _ => panic!("Cannot bitwise NOT {:?}", v),
    }
}

pub fn shift_left(a: Value, b: Value) -> Value {
    let shift = b.as_uint();
    match a {
        Value::Int(x) => Value::Int(x << shift),
        Value::UInt(x) => Value::UInt(x << shift),
        _ => panic!("Cannot shift left {:?}", a),
    }
}

pub fn shift_right(a: Value, b: Value) -> Value {
    let shift = b.as_uint();
    match a {
        Value::Int(x) => Value::Int(x >> shift),
        Value::UInt(x) => Value::UInt(x >> shift),
        _ => panic!("Cannot shift right {:?}", a),
    }
}
