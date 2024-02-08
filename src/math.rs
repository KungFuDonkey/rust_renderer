#![allow(dead_code)]

use std::ops::*;

pub const EPSILON: f32 = 0.00005;
pub const EPSILON_VECTOR: Float3 = Float3 { x: EPSILON, y: EPSILON, z: EPSILON};

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct Int2 {
    pub x: i32,
    pub y: i32
}

impl Int2 {
    pub fn zero() -> Self {
        Int2 { x: 0, y: 0 }
    }

    pub fn from_xy(x: i32, y: i32) -> Self
    {
        Int2 { x, y }
    }

    pub fn from_a(a: i32) -> Self {
        Int2 { x:a, y:a }
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct Uint2 {
    pub x: u32,
    pub y: u32
}

impl Uint2 {
    pub fn zero() -> Self {
        Uint2 { x: 0, y: 0 }
    }

    pub fn from_xy(x: u32, y: u32) -> Self
    {
        Uint2 { x, y }
    }

    pub fn from_a(a: u32) -> Self {
        Uint2 { x:a, y:a }
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct Float2 {
    pub x: f32,
    pub y: f32
}

impl Float2 {
    pub fn zero() -> Self {
        Float2 { x: 0.0, y: 0.0 }
    }

    pub fn from_xy(x: f32, y: f32) -> Self
    {
        Float2 { x, y }
    }

    pub fn from_a(a: f32) -> Self {
        Float2 { x:a, y:a }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Int3 {
    pub x: i32,
    pub y: i32,
    pub z: i32
}

impl Int3 {
    pub fn zero() -> Self {
        Int3 { x: 0, y: 0, z: 0 }
    }

    pub fn from_xyz(x: i32, y: i32, z: i32) -> Self
    {
        Int3 { x, y, z }
    }

    pub fn from_a(a: i32) -> Self {
        Int3 { x: a, y: a, z: a }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Uint3 {
    pub x: u32,
    pub y: u32,
    pub z: u32
}

impl Uint3 {
    pub fn zero() -> Self {
        Uint3 { x: 0, y: 0, z: 0 }
    }

    pub fn from_xyz(x: u32, y: u32, z: u32) -> Self
    {
        Uint3 { x, y, z }
    }

    pub fn from_a(a: u32) -> Self {
        Uint3 { x: a, y: a, z: a }
    }

}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32
}

impl Float3 {
    pub fn zero() -> Self {
        Float3 { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self
    {
        Float3 { x, y, z }
    }

    pub fn from_a(a: f32) -> Self {
        Float3 { x: a, y: a, z: a }
    }

    pub fn get_axis(&self, axis: usize) -> f32
    {
        match axis {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => panic!("outside of bounds in get_axis()")
        }
    }

    pub fn set_axis(&mut self, axis: usize, value: f32)
    {
        match axis {
            0 => self.x = value,
            1 => self.y = value,
            2 => self.z = value,
            _ => panic!("outside of bounds in set_axis()")
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Int4 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32
}

impl Int4 {
    pub fn zero() -> Self {
        Int4 { x: 0, y: 0, z: 0, w: 0 }
    }

    pub fn from_xyzw(x: i32, y: i32, z: i32, w: i32) -> Self
    {
        Int4 { x, y, z, w }
    }

    pub fn from_a(a: i32) -> Self {
        Int4 { x: a, y: a, z: a, w: a }
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct Uint4 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32
}

impl Uint4 {
    pub fn zero() -> Self {
        Uint4 { x: 0, y: 0, z: 0, w: 0 }
    }

    pub fn from_xyzw(x: u32, y: u32, z: u32, w: u32) -> Self
    {
        Uint4 { x, y, z, w }
    }

    pub fn from_a(a: u32) -> Self {
        Uint4 { x: a, y: a, z: a, w: a }
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct Float4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32
}

impl Float4 {
    pub fn zero() -> Self {
        Float4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }
    }

    pub fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self
    {
        Float4 { x, y, z, w }
    }

    pub fn from_a(a: f32) -> Self {
        Float4 { x: a, y: a, z: a, w: a }
    }
}

// random numbers
static mut SEED: u32 = 0x12345678;

#[inline(always)]
fn wang_hash(s: u32) -> u32
{
    let mut v: u128 = ((s ^ 61) ^ (s >> 16)) as u128;
    v *= 9;
    v = v ^ (v >> 4);
    v *= 0x27d4eb2d;
    v = v ^ (v >> 15);
    return v as u32;
}

#[inline(always)]
pub fn init_seed(seed_base: u32) -> u32
{
    return wang_hash((seed_base + 1) * 17)
}

#[inline(always)]
pub fn random_uint() -> u32
{
    let v: u32 = 0;
    unsafe {
        SEED ^= SEED << 13;
        SEED ^= SEED >> 17;
        SEED ^= SEED << 5;
    }
    return v;
}

#[inline(always)]
pub fn random_uint_s(seed: &mut u32) -> u32
{
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return *seed;
}

#[inline(always)]
pub fn random_float() -> f32
{
    (random_uint() as f32) * 2.3283064365387e-10
}

#[inline(always)]
pub fn random_float_s(seed: &mut u32) -> f32
{
    (random_uint_s(seed) as f32) * 2.3283064365387e-10
}

#[inline(always)]
pub fn rand(range: f32) -> f32
{
    random_float() * range
}

static NUMX: i32 = 512;
static NUMY: i32 = 512;
static NUMOCTAVES: i32 = 7;
static PRIMEINDEX: i32 = 0;
static PERSISTENCE: f32 = 0.5;

static PRIMES: [[i32; 3]; 10] = [
    [ 995615039, 600173719, 701464987 ],
    [ 831731269, 162318869, 136250887 ],
    [ 174329291, 946737083, 245679977 ],
    [ 362489573, 795918041, 350777237 ],
    [ 457025711, 880830799, 909678923 ],
    [ 787070341, 177340217, 593320781 ],
    [ 405493717, 291031019, 391950901 ],
    [ 458904767, 676625681, 424452397 ],
    [ 531736441, 939683957, 810651871 ],
    [ 997169939, 842027887, 423882827 ]
];

#[inline(always)]
pub fn noise(i: usize, x: i32, y: i32) -> f32
{
    let mut n = x + y * 57;
    n = (n << 13) ^ n;
    let a = PRIMES[i][0];
    let b = PRIMES[i][1];
    let c = PRIMES[i][2];
    let t: i32 = (n * (n * n * a + b) + c) & 0x7fffffff;
    return 1.0 - (t as f32) / 1073741824.0;
}

pub fn expf( a: &Float3) -> Float3
{
    Float3::from_xyz(fast_math::exp(a.x), fast_math::exp(a.y), fast_math::exp(a.z))
}

macro_rules! impl_default_operator_2
{
    ($type:ty, $trait:ty, $trait_fn:ident) => {
        impl $trait for $type
        {
            type Output = $type;

            #[inline(always)]
            fn $trait_fn(self, rhs: Self) -> Self::Output
            {
                <$type>::from_xy(self.x.$trait_fn(rhs.x), self.y.$trait_fn(rhs.y))
            }
        }
    }
}

macro_rules! impl_default_assign_operator_2
{
    ($type:ty, $trait:ty, $trait_fn:ident) => {
        impl $trait for $type
        {
            #[inline(always)]
            fn $trait_fn(&mut self, rhs: Self)
            {
                self.x.$trait_fn(rhs.x);
                self.y.$trait_fn(rhs.y);
            }
        }
    }
}

macro_rules! impl_single_operator_2 {
    ($type:ty, $subtype:ty, $trait:ty, $trait_t:ty, $trait_fn:ident) => {
        impl $trait for $type
        {
            type Output = $type;

            #[inline(always)]
            fn $trait_fn(self, rhs: $trait_t) -> Self::Output
            {
                <$type>::from_xy(self.x.$trait_fn(rhs as $subtype), self.y.$trait_fn(rhs as $subtype))
            }
        }
    };
}

macro_rules! impl_single_assign_operator_2 {
    ($type:ty, $subtype:ty, $trait:ty, $trait_t:ty, $trait_fn:ident) => {
        impl $trait for $type
        {
            #[inline(always)]
            fn $trait_fn(&mut self, rhs: $trait_t)
            {
                self.x.$trait_fn(rhs as $subtype);
                self.y.$trait_fn(rhs as $subtype);
            }
        }
    };
}

macro_rules! impl_single_operator_2_reversed {
    ($type:ty, $subtype:ty, $trait:ty, $trait_t:ty, $trait_fn:ident) => {
        impl $trait for $trait_t
        {
            type Output = $type;

            #[inline(always)]
            fn $trait_fn(self, rhs: $type) -> Self::Output
            {
                <$type>::from_xy(rhs.x.$trait_fn(self as $subtype), rhs.y.$trait_fn(self as $subtype))
            }
        }
    };
}

macro_rules! impl_operators_2 {
    ($type:ty, $subtype:ty) => {

        impl_default_operator_2!($type, Sub, sub);
        impl_default_assign_operator_2!($type, SubAssign, sub_assign);
        impl_single_operator_2!($type, $subtype, Sub<i32>, i32, sub);
        impl_single_assign_operator_2!($type, $subtype, SubAssign<i32>, i32, sub_assign);
        impl_single_operator_2!($type, $subtype, Sub<u32>, u32, sub);
        impl_single_assign_operator_2!($type, $subtype, SubAssign<u32>, u32, sub_assign);
        impl_single_operator_2!($type, $subtype, Sub<f32>, f32, sub);
        impl_single_assign_operator_2!($type, $subtype, SubAssign<f32>, f32, sub_assign);

        impl_default_operator_2!($type, Add, add);
        impl_default_assign_operator_2!($type, AddAssign, add_assign);
        impl_single_operator_2!($type, $subtype, Add<i32>, i32, add);
        impl_single_assign_operator_2!($type, $subtype, AddAssign<i32>, i32, add_assign);
        impl_single_operator_2!($type, $subtype, Add<u32>, u32, add);
        impl_single_assign_operator_2!($type, $subtype, AddAssign<u32>, u32, add_assign);
        impl_single_operator_2!($type, $subtype, Add<f32>, f32, add);
        impl_single_assign_operator_2!($type, $subtype, AddAssign<f32>, f32, add_assign);

        impl_default_operator_2!($type, Mul, mul);
        impl_default_assign_operator_2!($type, MulAssign, mul_assign);
        impl_single_operator_2!($type, $subtype, Mul<i32>, i32, mul);
        impl_single_operator_2_reversed!($type, $subtype, Mul<$type>, i32, mul);
        impl_single_assign_operator_2!($type, $subtype, MulAssign<i32>, i32, mul_assign);
        impl_single_operator_2!($type, $subtype, Mul<u32>, u32, mul);
        impl_single_operator_2_reversed!($type, $subtype, Mul<$type>, u32, mul);
        impl_single_assign_operator_2!($type, $subtype, MulAssign<u32>, u32, mul_assign);
        impl_single_operator_2!($type, $subtype, Mul<f32>, f32, mul);
        impl_single_operator_2_reversed!($type, $subtype, Mul<$type>, f32, mul);
        impl_single_assign_operator_2!($type, $subtype, MulAssign<f32>, f32, mul_assign);
    };
}

macro_rules! impl_additional_operators_2 {
    ($type:ty) => {
        impl Neg for $type
        {
            type Output = $type;

            #[inline(always)]
            fn neg(self) -> Self::Output {
                <$type>::from_xy(-self.x, -self.y)
            }
        }
    };
}

impl_operators_2!(Int2, i32);
impl_additional_operators_2!(Int2);
impl_operators_2!(Float2, f32);
impl_additional_operators_2!(Float2);
impl_operators_2!(Uint2, u32);
impl_default_operator_2!(Float2, Div, div);
impl_single_operator_2!(Float2, f32, Div<f32>, f32, div);
impl_single_operator_2!(Int2, i32, Shl<i32>, i32, shl);
impl_single_operator_2!(Int2, i32, Shr<i32>, i32, shr);


macro_rules! impl_default_operator_3
{
    ($type:ty, $trait:ty, $trait_fn:ident) => {
        impl $trait for $type
        {
            type Output = $type;
            #[inline(always)]
            fn $trait_fn(self, rhs: Self) -> Self::Output
            {
                <$type>::from_xyz(self.x.$trait_fn(rhs.x), self.y.$trait_fn(rhs.y), self.z.$trait_fn(rhs.z))
            }
        }
    }
}

macro_rules! impl_default_assign_operator_3
{
    ($type:ty, $trait:ty, $trait_fn:ident) => {
        impl $trait for $type
        {
            #[inline(always)]
            fn $trait_fn(&mut self, rhs: Self)
            {
                self.x.$trait_fn(rhs.x);
                self.y.$trait_fn(rhs.y);
                self.z.$trait_fn(rhs.z);
            }
        }
    }
}

macro_rules! impl_single_operator_3 {
    ($type:ty, $subtype:ty, $trait:ty, $trait_t:ty, $trait_fn:ident) => {
        impl $trait for $type
        {
            type Output = $type;
            #[inline(always)]
            fn $trait_fn(self, rhs: $trait_t) -> Self::Output
            {
                <$type>::from_xyz(self.x.$trait_fn(rhs as $subtype), self.y.$trait_fn(rhs as $subtype), self.z.$trait_fn(rhs as $subtype))
            }
        }
    };
}

macro_rules! impl_single_assign_operator_3 {
    ($type:ty, $subtype:ty, $trait:ty, $trait_t:ty, $trait_fn:ident) => {
        impl $trait for $type
        {
            #[inline(always)]
            fn $trait_fn(&mut self, rhs: $trait_t)
            {
                self.x.$trait_fn(rhs as $subtype);
                self.y.$trait_fn(rhs as $subtype);
                self.z.$trait_fn(rhs as $subtype);
            }
        }
    };
}

macro_rules! impl_single_operator_3_reversed {
    ($type:ty, $subtype:ty, $trait:ty, $trait_t:ty, $trait_fn:ident) => {
        impl $trait for $trait_t
        {
            type Output = $type;
            #[inline(always)]
            fn $trait_fn(self, rhs: $type) -> Self::Output
            {
                <$type>::from_xyz(rhs.x.$trait_fn(self as $subtype), rhs.y.$trait_fn(self as $subtype), rhs.z.$trait_fn(self as $subtype))
            }
        }
    };
}

macro_rules! impl_operators_3 {
    ($type:ty, $subtype:ty) => {

        impl_default_operator_3!($type, Sub, sub);
        impl_default_assign_operator_3!($type, SubAssign, sub_assign);
        impl_single_operator_3!($type, $subtype, Sub<i32>, i32, sub);
        impl_single_assign_operator_3!($type, $subtype, SubAssign<i32>, i32, sub_assign);
        impl_single_operator_3!($type, $subtype, Sub<u32>, u32, sub);
        impl_single_assign_operator_3!($type, $subtype, SubAssign<u32>, u32, sub_assign);
        impl_single_operator_3!($type, $subtype, Sub<f32>, f32, sub);
        impl_single_assign_operator_3!($type, $subtype, SubAssign<f32>, f32, sub_assign);

        impl_default_operator_3!($type, Add, add);
        impl_default_assign_operator_3!($type, AddAssign, add_assign);
        impl_single_operator_3!($type, $subtype, Add<i32>, i32, add);
        impl_single_assign_operator_3!($type, $subtype, AddAssign<i32>, i32, add_assign);
        impl_single_operator_3!($type, $subtype, Add<u32>, u32, add);
        impl_single_assign_operator_3!($type, $subtype, AddAssign<u32>, u32, add_assign);
        impl_single_operator_3!($type, $subtype, Add<f32>, f32, add);
        impl_single_assign_operator_3!($type, $subtype, AddAssign<f32>, f32, add_assign);

        impl_default_operator_3!($type, Mul, mul);
        impl_default_assign_operator_3!($type, MulAssign, mul_assign);
        impl_single_operator_3!($type, $subtype, Mul<i32>, i32, mul);
        impl_single_operator_3_reversed!($type, $subtype, Mul<$type>, i32, mul);
        impl_single_assign_operator_3!($type, $subtype, MulAssign<i32>, i32, mul_assign);
        impl_single_operator_3!($type, $subtype, Mul<u32>, u32, mul);
        impl_single_operator_3_reversed!($type, $subtype, Mul<$type>, u32, mul);
        impl_single_assign_operator_3!($type, $subtype, MulAssign<u32>, u32, mul_assign);
        impl_single_operator_3!($type, $subtype, Mul<f32>, f32, mul);
        impl_single_operator_3_reversed!($type, $subtype, Mul<$type>, f32, mul);
        impl_single_assign_operator_3!($type, $subtype, MulAssign<f32>, f32, mul_assign);
    };
}

macro_rules! impl_additional_operators_3 {
    ($type:ty) => {
        impl Neg for $type
        {
            type Output = $type;

            #[inline(always)]
            fn neg(self) -> Self::Output {
                <$type>::from_xyz(-self.x, -self.y, -self.z)
            }
        }
    };
}

impl_operators_3!(Int3, i32);
impl_additional_operators_3!(Int3);
impl_operators_3!(Float3, f32);
impl_additional_operators_3!(Float3);
impl_operators_3!(Uint3, u32);
impl_default_operator_3!(Float3, Div, div);
impl_single_operator_3!(Float3, f32, Div<f32>, f32, div);
impl_single_operator_3!(Int3, i32, Shl<i32>, i32, shl);
impl_single_operator_3!(Int3, i32, Shr<i32>, i32, shr);

macro_rules! impl_default_operator_4
{
    ($type:ty, $trait:ty, $trait_fn:ident) => {
        impl $trait for $type
        {
            type Output = $type;
            #[inline(always)]
            fn $trait_fn(self, rhs: Self) -> Self::Output
            {
                <$type>::from_xyzw(self.x.$trait_fn(rhs.x), self.y.$trait_fn(rhs.y), self.z.$trait_fn(rhs.z), self.w.$trait_fn(rhs.w))
            }
        }
    }
}

macro_rules! impl_default_assign_operator_4
{
    ($type:ty, $trait:ty, $trait_fn:ident) => {
        impl $trait for $type
        {
            #[inline(always)]
            fn $trait_fn(&mut self, rhs: Self)
            {
                self.x.$trait_fn(rhs.x);
                self.y.$trait_fn(rhs.y);
                self.z.$trait_fn(rhs.z);
                self.w.$trait_fn(rhs.w);
            }
        }
    }
}

macro_rules! impl_single_operator_4 {
    ($type:ty, $subtype:ty, $trait:ty, $trait_t:ty, $trait_fn:ident) => {
        impl $trait for $type
        {
            type Output = $type;
            #[inline(always)]
            fn $trait_fn(self, rhs: $trait_t) -> Self::Output
            {
                <$type>::from_xyzw(self.x.$trait_fn(rhs as $subtype), self.y.$trait_fn(rhs as $subtype), self.z.$trait_fn(rhs as $subtype), self.w.$trait_fn(rhs as $subtype))
            }
        }
    };
}

macro_rules! impl_single_assign_operator_4 {
    ($type:ty, $subtype:ty, $trait:ty, $trait_t:ty, $trait_fn:ident) => {
        impl $trait for $type
        {
            #[inline(always)]
            fn $trait_fn(&mut self, rhs: $trait_t)
            {
                self.x.$trait_fn(rhs as $subtype);
                self.y.$trait_fn(rhs as $subtype);
                self.z.$trait_fn(rhs as $subtype);
                self.w.$trait_fn(rhs as $subtype);
            }
        }
    };
}

macro_rules! impl_single_operator_4_reversed {
    ($type:ty, $subtype:ty, $trait:ty, $trait_t:ty, $trait_fn:ident) => {
        impl $trait for $trait_t
        {
            type Output = $type;
            #[inline(always)]
            fn $trait_fn(self, rhs: $type) -> Self::Output
            {
                <$type>::from_xyzw(rhs.x.$trait_fn(self as $subtype), rhs.y.$trait_fn(self as $subtype), rhs.z.$trait_fn(self as $subtype), rhs.w.$trait_fn(self as $subtype))
            }
        }
    };
}

macro_rules! impl_operators_4 {
    ($type:ty, $subtype:ty) => {

        impl_default_operator_4!($type, Sub, sub);
        impl_default_assign_operator_4!($type, SubAssign, sub_assign);
        impl_single_operator_4!($type, $subtype, Sub<i32>, i32, sub);
        impl_single_assign_operator_4!($type, $subtype, SubAssign<i32>, i32, sub_assign);
        impl_single_operator_4!($type, $subtype, Sub<u32>, u32, sub);
        impl_single_assign_operator_4!($type, $subtype, SubAssign<u32>, u32, sub_assign);
        impl_single_operator_4!($type, $subtype, Sub<f32>, f32, sub);
        impl_single_assign_operator_4!($type, $subtype, SubAssign<f32>, f32, sub_assign);

        impl_default_operator_4!($type, Add, add);
        impl_default_assign_operator_4!($type, AddAssign, add_assign);
        impl_single_operator_4!($type, $subtype, Add<i32>, i32, add);
        impl_single_assign_operator_4!($type, $subtype, AddAssign<i32>, i32, add_assign);
        impl_single_operator_4!($type, $subtype, Add<u32>, u32, add);
        impl_single_assign_operator_4!($type, $subtype, AddAssign<u32>, u32, add_assign);
        impl_single_operator_4!($type, $subtype, Add<f32>, f32, add);
        impl_single_assign_operator_4!($type, $subtype, AddAssign<f32>, f32, add_assign);

        impl_default_operator_4!($type, Mul, mul);
        impl_default_assign_operator_4!($type, MulAssign, mul_assign);
        impl_single_operator_4!($type, $subtype, Mul<i32>, i32, mul);
        impl_single_operator_4_reversed!($type, $subtype, Mul<$type>, i32, mul);
        impl_single_assign_operator_4!($type, $subtype, MulAssign<i32>, i32, mul_assign);
        impl_single_operator_4!($type, $subtype, Mul<u32>, u32, mul);
        impl_single_operator_4_reversed!($type, $subtype, Mul<$type>, u32, mul);
        impl_single_assign_operator_4!($type, $subtype, MulAssign<u32>, u32, mul_assign);
        impl_single_operator_4!($type, $subtype, Mul<f32>, f32, mul);
        impl_single_operator_4_reversed!($type, $subtype, Mul<$type>, f32, mul);
        impl_single_assign_operator_4!($type, $subtype, MulAssign<f32>, f32, mul_assign);
    };
}

macro_rules! impl_additional_operators_4 {
    ($type:ty) => {
        impl Neg for $type
        {
            type Output = $type;

            fn neg(self) -> Self::Output {
                <$type>::from_xyzw(-self.x, -self.y, -self.z, -self.w)
            }
        }
    };
}

impl_operators_4!(Int4, i32);
impl_additional_operators_4!(Int4);
impl_operators_4!(Float4, f32);
impl_additional_operators_4!(Float4);
impl_operators_4!(Uint4, u32);
impl_default_operator_4!(Float4, Div, div);
impl_single_operator_4!(Float4, f32, Div<f32>, f32, div);
impl_single_operator_4!(Int4, i32, Shl<i32>, i32, shl);
impl_single_operator_4!(Int4, i32, Shr<i32>, i32, shr);

pub trait VectorMinMax
{
    fn min(&self, rhs: &Self) -> Self;

    fn max(&self, rhs: &Self) -> Self;

}

impl VectorMinMax for Float2
{
    fn min(&self, rhs: &Self) -> Self {
        Self::from_xy( self.x.min(rhs.x) , self.y.min(rhs.y) )
    }

    fn max(&self, rhs: &Self) -> Self {
        Self::from_xy( self.x.max( rhs.x ), self.y.max( rhs.y ) )
    }
}

impl VectorMinMax for Float3
{
    #[inline(always)]
    fn min(&self, rhs: &Self) -> Self {
        Self::from_xyz( self.x.min(rhs.x), self.y.min(rhs.y), self.z.min( rhs.z ) )
    }

    #[inline(always)]
    fn max(&self, rhs: &Self) -> Self {
        Self::from_xyz( self.x.max( rhs.x ), self.y.max( rhs.y ) , self.z.max( rhs.z ) )
    }
}

impl VectorMinMax for Float4
{
    #[inline(always)]
    fn min(&self, rhs: &Self) -> Self {
        Self::from_xyzw( self.x.min(rhs.x), self.y.min(rhs.y), self.z.min( rhs.z ) , self.w.min( rhs.w ) )
    }

    #[inline(always)]
    fn max(&self, rhs: &Self) -> Self {
        Self::from_xyzw( self.x.max( rhs.x ), self.y.max( rhs.y ) , self.z.max( rhs.z ) , self.w.max(rhs.w ) )
    }
}

impl VectorMinMax for Int2
{
    #[inline(always)]
    fn min(&self, rhs: &Self) -> Self {
        Self::from_xy( self.x.min(rhs.x) , self.y.min(rhs.y) )
    }

    #[inline(always)]
    fn max(&self, rhs: &Self) -> Self {
        Self::from_xy( self.x.max( rhs.x ), self.y.max( rhs.y ) )
    }
}

impl VectorMinMax for Int3
{
    #[inline(always)]
    fn min(&self, rhs: &Self) -> Self {
        Self::from_xyz( self.x.min(rhs.x), self.y.min(rhs.y), self.z.min( rhs.z ) )
    }

    #[inline(always)]
    fn max(&self, rhs: &Self) -> Self {
        Self::from_xyz( self.x.max( rhs.x ), self.y.max( rhs.y ) , self.z.max( rhs.z ) )
    }
}

impl VectorMinMax for Int4
{
    #[inline(always)]
    fn min(&self, rhs: &Self) -> Self {
        Self::from_xyzw( self.x.min(rhs.x), self.y.min(rhs.y), self.z.min( rhs.z ) , self.w.min( rhs.w ) )
    }

    #[inline(always)]
    fn max(&self, rhs: &Self) -> Self {
        Self::from_xyzw( self.x.max( rhs.x ), self.y.max( rhs.y ) , self.z.max( rhs.z ) , self.w.max(rhs.w ) )
    }
}

impl VectorMinMax for Uint2
{
    #[inline(always)]
    fn min(&self, rhs: &Self) -> Self {
        Self::from_xy( self.x.min(rhs.x) , self.y.min(rhs.y) )
    }

    #[inline(always)]
    fn max(&self, rhs: &Self) -> Self {
        Self::from_xy( self.x.max( rhs.x ), self.y.max( rhs.y ) )
    }
}

impl VectorMinMax for Uint3
{
    #[inline(always)]
    fn min(&self, rhs: &Self) -> Self {
        Self::from_xyz( self.x.min(rhs.x), self.y.min(rhs.y), self.z.min( rhs.z ) )
    }

    #[inline(always)]
    fn max(&self, rhs: &Self) -> Self {
        Self::from_xyz( self.x.max( rhs.x ), self.y.max( rhs.y ) , self.z.max( rhs.z ) )
    }
}

impl VectorMinMax for Uint4
{
    #[inline(always)]
    fn min(&self, rhs: &Self) -> Self {
        Self::from_xyzw( self.x.min(rhs.x), self.y.min(rhs.y), self.z.min( rhs.z ) , self.w.min( rhs.w ) )
    }

    #[inline(always)]
    fn max(&self, rhs: &Self) -> Self {
        Self::from_xyzw( self.x.max( rhs.x ), self.y.max( rhs.y ) , self.z.max( rhs.z ) , self.w.max(rhs.w ) )
    }
}

pub trait Lerp<StepType>
{
    fn lerp(a: &Self, b: &Self, t: StepType) -> Self;
}

impl Lerp<f32> for f32
{
    #[inline(always)]
    fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        *a + t * (*b - *a)
    }
}

impl Lerp<f32> for Float2
{
    #[inline(always)]
    fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        *a + (*b - *a) * t
    }
}

impl Lerp<f32> for Float3
{
    #[inline(always)]
    fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        *a + (*b - *a) * t
    }
}

impl Lerp<f32> for Float4
{
    #[inline(always)]
    fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        *a + (*b - *a) * t
    }
}

#[inline(always)]
pub fn lerp<T, StepType>(a: &T, b: &T, t: StepType) -> T
    where T: Lerp<StepType>
{
    T::lerp(a, b, t)
}

pub trait Clamp<DelimiterType>
{
    fn clamp(&self, a: DelimiterType, b: DelimiterType) -> Self;
}

impl Clamp<f32> for f32
{
    #[inline(always)]
    fn clamp(&self, a: f32, b: f32) -> Self {
        a.max(self.min( b ))
    }
}

impl Clamp<i32> for i32
{
    #[inline(always)]
    fn clamp(&self, a: i32, b: i32) -> Self
    {
        a.max((*self).min( b ))
    }
}

impl Clamp<u32> for u32
{
    #[inline(always)]
    fn clamp(&self, a: u32, b: u32) -> Self
    {
        a.max((*self).min( b ))
    }
}

impl Clamp<f32> for Float2
{
    #[inline(always)]
    fn clamp(&self, a: f32, b: f32) -> Self {
        Float2::from_xy(self.x.clamp(a, b), self.y.clamp(a, b))
    }
}

impl Clamp<i32> for Int2
{
    #[inline(always)]
    fn clamp(&self, a: i32, b: i32) -> Self
    {
        Int2::from_xy(self.x.clamp(a, b), self.y.clamp(a, b))
    }
}

impl Clamp<u32> for Uint2
{
    #[inline(always)]
    fn clamp(&self, a: u32, b: u32) -> Self
    {
        Uint2::from_xy(self.x.clamp(a, b), self.y.clamp(a, b))
    }
}

impl Clamp<f32> for Float3
{
    #[inline(always)]
    fn clamp(&self, a: f32, b: f32) -> Self {
        Float3::from_xyz(self.x.clamp(a, b), self.y.clamp(a, b), self.z.clamp(a, b))
    }
}

impl Clamp<i32> for Int3
{
    #[inline(always)]
    fn clamp(&self, a: i32, b: i32) -> Self {
        Int3::from_xyz(self.x.clamp(a, b), self.y.clamp(a, b), self.z.clamp(a, b))
    }
}

impl Clamp<u32> for Uint3
{
    #[inline(always)]
    fn clamp(&self, a: u32, b: u32) -> Self {
        Uint3::from_xyz(self.x.clamp(a, b), self.y.clamp(a, b), self.z.clamp(a, b))
    }
}

impl Clamp<f32> for Float4
{
    #[inline(always)]
    fn clamp(&self, a: f32, b: f32) -> Self {
        Float4::from_xyzw(self.x.clamp(a, b), self.y.clamp(a, b), self.z.clamp(a, b), self.w.clamp(a, b))
    }
}

impl Clamp<i32> for Int4
{
    #[inline(always)]
    fn clamp(&self, a: i32, b: i32) -> Self {
        Int4::from_xyzw(self.x.clamp(a, b), self.y.clamp(a, b), self.z.clamp(a, b), self.w.clamp(a, b))
    }
}

impl Clamp<u32> for Uint4
{
    #[inline(always)]
    fn clamp(&self, a: u32, b: u32) -> Self {
        Uint4::from_xyzw(self.x.clamp(a, b), self.y.clamp(a, b), self.z.clamp(a, b), self.w.clamp(a, b))
    }
}

pub trait VectorDot
{
    type Output;

    fn dot(a: &Self, b: &Self) -> Self::Output;
}

impl VectorDot for Float2
{
    type Output = f32;

    #[inline(always)]
    fn dot(a: &Self, b: &Self) -> Self::Output {
        a.x * b.x + a.y * b.y
    }
}

impl VectorDot for Float3
{
    type Output = f32;

    #[inline(always)]
    fn dot(a: &Self, b: &Self) -> Self::Output {
        a.x * b.x + a.y * b.y + a.z * b.z
    }
}

impl VectorDot for Float4
{
    type Output = f32;

    #[inline(always)]
    fn dot(a: &Self, b: &Self) -> Self::Output {
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
    }
}

impl VectorDot for Int2
{
    type Output = i32;

    #[inline(always)]
    fn dot(a: &Self, b: &Self) -> Self::Output {
        a.x * b.x + a.y * b.y
    }
}

impl VectorDot for Int3
{
    type Output = i32;

    #[inline(always)]
    fn dot(a: &Self, b: &Self) -> Self::Output {
        a.x * b.x + a.y * b.y + a.z * b.z
    }
}

impl VectorDot for Int4
{
    type Output = i32;

    #[inline(always)]
    fn dot(a: &Self, b: &Self) -> Self::Output {
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
    }
}

impl VectorDot for Uint2
{
    type Output = u32;

    #[inline(always)]
    fn dot(a: &Self, b: &Self) -> Self::Output {
        a.x * b.x + a.y * b.y
    }
}

impl VectorDot for Uint3
{
    type Output = u32;

    #[inline(always)]
    fn dot(a: &Self, b: &Self) -> Self::Output {
        a.x * b.x + a.y * b.y + a.z * b.z
    }
}

impl VectorDot for Uint4
{
    type Output = u32;

    #[inline(always)]
    fn dot(a: &Self, b: &Self) -> Self::Output {
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
    }
}

#[inline(always)]
pub fn dot<T>(a: &T, b: &T) -> <T as VectorDot>::Output
    where T: VectorDot
{
    T::dot(a, b)
}

pub trait VectorLength
{
    type SqrLengthOutput;

    type LengthOutput;

    fn sqr_length(&self) -> Self::SqrLengthOutput;

    fn length(&self) -> Self::LengthOutput;
}

impl VectorLength for Float2
{
    type SqrLengthOutput = f32;
    type LengthOutput = f32;

    #[inline(always)]
    fn sqr_length(&self) -> Self::SqrLengthOutput {
        Self::dot(self, self)
    }

    #[inline(always)]
    fn length(&self) -> Self::LengthOutput {
        Self::dot(self, self).sqrt()
    }
}

impl VectorLength for Float3
{
    type SqrLengthOutput = f32;
    type LengthOutput = f32;

    #[inline(always)]
    fn sqr_length(&self) -> Self::SqrLengthOutput {
        Self::dot(self, self)
    }

    #[inline(always)]
    fn length(&self) -> Self::LengthOutput {
        Self::dot(self, self).sqrt()
    }
}

impl VectorLength for Float4
{
    type SqrLengthOutput = f32;
    type LengthOutput = f32;

    #[inline(always)]
    fn sqr_length(&self) -> Self::SqrLengthOutput {
        Self::dot(self, self)
    }

    #[inline(always)]
    fn length(&self) -> Self::LengthOutput {
        Self::dot(self, self).sqrt()
    }
}

impl VectorLength for Int2
{
    type SqrLengthOutput = i32;
    type LengthOutput = f32;

    #[inline(always)]
    fn sqr_length(&self) -> Self::SqrLengthOutput {
        Self::dot(self, self)
    }

    #[inline(always)]
    fn length(&self) -> Self::LengthOutput {
        (Self::dot(self, self) as f32).sqrt()
    }
}

impl VectorLength for Int3
{
    type SqrLengthOutput = i32;
    type LengthOutput = f32;

    #[inline(always)]
    fn sqr_length(&self) -> Self::SqrLengthOutput {
        Self::dot(self, self)
    }

    #[inline(always)]
    fn length(&self) -> Self::LengthOutput {
        (Self::dot(self, self) as f32).sqrt()
    }
}

impl VectorLength for Int4
{
    type SqrLengthOutput = i32;
    type LengthOutput = f32;

    #[inline(always)]
    fn sqr_length(&self) -> Self::SqrLengthOutput {
        Self::dot(self, self)
    }

    #[inline(always)]
    fn length(&self) -> Self::LengthOutput {
        (Self::dot(self, self) as f32).sqrt()
    }
}

#[inline(always)]
pub fn length<T>(a: &T) -> <T as VectorLength>::LengthOutput
    where T: VectorLength
{
    a.length()
}

#[inline(always)]
pub fn sqr_length<T>(a: &T) -> <T as VectorLength>::SqrLengthOutput
    where T: VectorLength
{
    a.sqr_length()
}

pub trait VectorNormalize
{
    fn normalize(&self) -> Self;
}

impl VectorNormalize for Float2
{
    #[inline(always)]
    fn normalize(&self) -> Self {
        return *self * (1.0 / self.length())
    }
}

impl VectorNormalize for Float3
{
    #[inline(always)]
    fn normalize(&self) -> Self {
        return *self * (1.0 / self.length())
    }
}

impl VectorNormalize for Float4
{
    #[inline(always)]
    fn normalize(&self) -> Self {
        return *self * (1.0 / self.length())
    }
}

#[inline(always)]
pub fn normalize<T>(a: &T) -> T
    where T: VectorNormalize
{
    a.normalize()
}

pub trait VectorReflect
{
    fn reflect(&self, n: &Self) -> Self;
}


impl VectorReflect for Float3
{
    #[inline(always)]
    fn reflect(&self, n: &Self) -> Self {
        *self - *n * Self::dot(self, n) * 2.0
    }
}

#[inline(always)]
pub fn reflect<T>(a: &T, b: &T) -> T
    where T: VectorReflect
{
    T::reflect(a, b)
}

pub trait VectorCross
{
    fn cross(a: &Self, b: &Self) -> Self;
}

impl VectorCross for Float3
{
    #[inline(always)]
    fn cross(a: &Self, b: &Self) -> Self {
        Float3::from_xyz( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x )
    }
}

#[inline(always)]
pub fn cross<T>(a: &T, b: &T) -> T
    where T: VectorCross
{
    T::cross(a, b)
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4
{
    pub cell: [f32; 16]
}

impl Mat4
{
    #[inline(always)]
    pub fn identity_matrix() -> Self
    {
        Mat4 { cell: [ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] }
    }

    #[inline(always)]
    pub fn zero_matrix() -> Self
    {
        Mat4 { cell: [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] }
    }

    #[inline(always)]
    pub fn get_translation(&self) -> Float3
    {
        Float3::from_xyz(self.cell[3], self.cell[7], self.cell[11])
    }

    #[inline(always)]
    pub fn rotate_x(a: f32) -> Self
    {
        let mut r = Mat4::identity_matrix();
        let ca = a.cos();
        let sa = a.sin();
        r.cell[5] = ca;
        r.cell[6] = -sa;
        r.cell[9] = sa;
        r.cell[10] = ca;
        return r;
    }

    #[inline(always)]
    pub fn rotate_y(a: f32) -> Self
    {
        let mut r = Mat4::identity_matrix();
        let ca = a.cos();
        let sa = a.sin();
        r.cell[0] = ca;
        r.cell[2] = sa;
        r.cell[8] = -sa;
        r.cell[10] = ca;
        return r;
    }

    #[inline(always)]
    pub fn rotate_z(a: f32) -> Self
    {
        let mut r = Mat4::identity_matrix();
        let ca = a.cos();
        let sa = a.sin();
        r.cell[0] = ca;
        r.cell[1] = -sa;
        r.cell[4] = sa;
        r.cell[5] = ca;
        return r;
    }

    #[inline(always)]
    pub fn translate(p: &Float3) -> Self
    {
        let mut ret_val = Mat4::identity_matrix();
        ret_val.cell[3] = p.x;
        ret_val.cell[7] = p.y;
        ret_val.cell[11] = p.z;
        return ret_val;
    }

    #[inline(always)]
    pub fn scale(p: f32) -> Self
    {
        let mut ret_val = Mat4::identity_matrix();
        ret_val.cell[0] = p;
        ret_val.cell[5] = p;
        ret_val.cell[10] = p;
        return ret_val;
    }

    #[inline(always)]
    pub fn inverted(&self) -> Mat4
    {
        let inv = [
            self.cell[5] * self.cell[10] * self.cell[15] - self.cell[5] * self.cell[11] * self.cell[14] - self.cell[9] * self.cell[6] * self.cell[15] +
                self.cell[9] * self.cell[7] * self.cell[14] + self.cell[13] * self.cell[6] * self.cell[11] - self.cell[13] * self.cell[7] * self.cell[10],
            -self.cell[1] * self.cell[10] * self.cell[15] + self.cell[1] * self.cell[11] * self.cell[14] + self.cell[9] * self.cell[2] * self.cell[15] -
                self.cell[9] * self.cell[3] * self.cell[14] - self.cell[13] * self.cell[2] * self.cell[11] + self.cell[13] * self.cell[3] * self.cell[10],
            self.cell[1] * self.cell[6] * self.cell[15] - self.cell[1] * self.cell[7] * self.cell[14] - self.cell[5] * self.cell[2] * self.cell[15] +
                self.cell[5] * self.cell[3] * self.cell[14] + self.cell[13] * self.cell[2] * self.cell[7] - self.cell[13] * self.cell[3] * self.cell[6],
            -self.cell[1] * self.cell[6] * self.cell[11] + self.cell[1] * self.cell[7] * self.cell[10] + self.cell[5] * self.cell[2] * self.cell[11] -
                self.cell[5] * self.cell[3] * self.cell[10] - self.cell[9] * self.cell[2] * self.cell[7] + self.cell[9] * self.cell[3] * self.cell[6],
            -self.cell[4] * self.cell[10] * self.cell[15] + self.cell[4] * self.cell[11] * self.cell[14] + self.cell[8] * self.cell[6] * self.cell[15] -
                self.cell[8] * self.cell[7] * self.cell[14] - self.cell[12] * self.cell[6] * self.cell[11] + self.cell[12] * self.cell[7] * self.cell[10],
            self.cell[0] * self.cell[10] * self.cell[15] - self.cell[0] * self.cell[11] * self.cell[14] - self.cell[8] * self.cell[2] * self.cell[15] +
                self.cell[8] * self.cell[3] * self.cell[14] + self.cell[12] * self.cell[2] * self.cell[11] - self.cell[12] * self.cell[3] * self.cell[10],
            -self.cell[0] * self.cell[6] * self.cell[15] + self.cell[0] * self.cell[7] * self.cell[14] + self.cell[4] * self.cell[2] * self.cell[15] -
                self.cell[4] * self.cell[3] * self.cell[14] - self.cell[12] * self.cell[2] * self.cell[7] + self.cell[12] * self.cell[3] * self.cell[6],
            self.cell[0] * self.cell[6] * self.cell[11] - self.cell[0] * self.cell[7] * self.cell[10] - self.cell[4] * self.cell[2] * self.cell[11] +
                self.cell[4] * self.cell[3] * self.cell[10] + self.cell[8] * self.cell[2] * self.cell[7] - self.cell[8] * self.cell[3] * self.cell[6],
            self.cell[4] * self.cell[9] * self.cell[15] - self.cell[4] * self.cell[11] * self.cell[13] - self.cell[8] * self.cell[5] * self.cell[15] +
                self.cell[8] * self.cell[7] * self.cell[13] + self.cell[12] * self.cell[5] * self.cell[11] - self.cell[12] * self.cell[7] * self.cell[9],
            -self.cell[0] * self.cell[9] * self.cell[15] + self.cell[0] * self.cell[11] * self.cell[13] + self.cell[8] * self.cell[1] * self.cell[15] -
                self.cell[8] * self.cell[3] * self.cell[13] - self.cell[12] * self.cell[1] * self.cell[11] + self.cell[12] * self.cell[3] * self.cell[9],
            self.cell[0] * self.cell[5] * self.cell[15] - self.cell[0] * self.cell[7] * self.cell[13] - self.cell[4] * self.cell[1] * self.cell[15] +
                self.cell[4] * self.cell[3] * self.cell[13] + self.cell[12] * self.cell[1] * self.cell[7] - self.cell[12] * self.cell[3] * self.cell[5],
            -self.cell[0] * self.cell[5] * self.cell[11] + self.cell[0] * self.cell[7] * self.cell[9] + self.cell[4] * self.cell[1] * self.cell[11] -
                self.cell[4] * self.cell[3] * self.cell[9] - self.cell[8] * self.cell[1] * self.cell[7] + self.cell[8] * self.cell[3] * self.cell[5],
            -self.cell[4] * self.cell[9] * self.cell[14] + self.cell[4] * self.cell[10] * self.cell[13] + self.cell[8] * self.cell[5] * self.cell[14] -
                self.cell[8] * self.cell[6] * self.cell[13] - self.cell[12] * self.cell[5] * self.cell[10] + self.cell[12] * self.cell[6] * self.cell[9],
            self.cell[0] * self.cell[9] * self.cell[14] - self.cell[0] * self.cell[10] * self.cell[13] - self.cell[8] * self.cell[1] * self.cell[14] +
                self.cell[8] * self.cell[2] * self.cell[13] + self.cell[12] * self.cell[1] * self.cell[10] - self.cell[12] * self.cell[2] * self.cell[9],
            -self.cell[0] * self.cell[5] * self.cell[14] + self.cell[0] * self.cell[6] * self.cell[13] + self.cell[4] * self.cell[1] * self.cell[14] -
                self.cell[4] * self.cell[2] * self.cell[13] - self.cell[12] * self.cell[1] * self.cell[6] + self.cell[12] * self.cell[2] * self.cell[5],
            self.cell[0] * self.cell[5] * self.cell[10] - self.cell[0] * self.cell[6] * self.cell[9] - self.cell[4] * self.cell[1] * self.cell[10] +
                self.cell[4] * self.cell[2] * self.cell[9] + self.cell[8] * self.cell[1] * self.cell[6] - self.cell[8] * self.cell[2] * self.cell[5]
        ];
        let det = self.cell[0] * inv[0] + self.cell[1] * inv[4] + self.cell[2] * inv[8] + self.cell[3] * inv[12];
        let mut ret_val = Mat4::identity_matrix();
        if det != 0.0
        {
            let inv_det = 1.0 / det;
            for i in 0..16
            {
                ret_val.cell[i] = inv[i] * inv_det;
            }
        }
        return ret_val;
    }

    #[inline(always)]
    pub fn inverted_no_scale(&self) -> Mat4
    {
        let mut r = Mat4::identity_matrix();
        r.cell[0] = self.cell[0];
        r.cell[1] = self.cell[4];
        r.cell[2] = self.cell[8];
        r.cell[4] = self.cell[1];
        r.cell[5] = self.cell[5];
        r.cell[6] = self.cell[9];
        r.cell[8] = self.cell[2];
        r.cell[9] = self.cell[6];
        r.cell[10] = self.cell[10];
        r.cell[3] = -(self.cell[3] * r.cell[0] + self.cell[7] * r.cell[1] + self.cell[11] * r.cell[2]);
        r.cell[7] = -(self.cell[3] * r.cell[4] + self.cell[7] * r.cell[5] + self.cell[11] * r.cell[6]);
        r.cell[11] = -(self.cell[3] * r.cell[8] + self.cell[7] * r.cell[9] + self.cell[11] * r.cell[10]);
        return r;
    }
}

impl Mul<&Mat4> for Float4
{
    type Output = Float4;

    #[inline(always)]
    fn mul(self, rhs: &Mat4) -> Self::Output {
        Float4::from_xyzw(
            rhs.cell[0] * self.x + rhs.cell[1] * self.y + rhs.cell[2] * self.z + rhs.cell[3] * self.w,
            rhs.cell[4] * self.x + rhs.cell[5] * self.y + rhs.cell[6] * self.z + rhs.cell[7] * self.w,
            rhs.cell[8] * self.x + rhs.cell[9] * self.y + rhs.cell[10] * self.z + rhs.cell[11] * self.w,
            rhs.cell[12] * self.x + rhs.cell[13] * self.y + rhs.cell[14] * self.z + rhs.cell[15] * self.w
        )
    }
}

impl Mul<&Mat4> for Mat4
{
    type Output = Mat4;

    #[inline(always)]
    fn mul(self, rhs: &Mat4) -> Self::Output {
        let mut r = Mat4::identity_matrix();
        for i in (0..16).step_by(4) {
            for j in 0..4 {
                r.cell[i + j] =
                    (self.cell[i + 0] * rhs.cell[j + 0]) +
                        (self.cell[i + 1] * rhs.cell[j + 4]) +
                        (self.cell[i + 2] * rhs.cell[j + 8]) +
                        (self.cell[i + 3] * rhs.cell[j + 12]);
            }
        }
        return r;
    }
}

impl Mul<Mat4> for Mat4
{
    type Output = Mat4;

    fn mul(self, rhs: Mat4) -> Self::Output {
        return self * &rhs;
    }
}

#[inline(always)]
pub fn transform_position(a: &Float3, m: &Mat4) -> Float3
{
    let f4 = Float4::from_xyzw(a.x, a.y, a.z, 1.0) * m;
    Float3::from_xyz(f4.x, f4.y, f4.z)
}

#[inline(always)]
pub fn transform_vector(a: &Float3, m: &Mat4) -> Float3
{
    let f4 = Float4::from_xyzw(a.x, a.y, a.z, 0.0) * m;
    Float3::from_xyz(f4.x, f4.y, f4.z)
}

#[inline(always)]
pub fn rgbf32_to_rgb8_f4(value: &Float4) -> u32
{
    let r: u32 = (255.0 * value.x.min(1.0)) as u32;
    let g: u32 = (255.0 * value.y.min(1.0)) as u32;
    let b: u32 = (255.0 * value.z.min(1.0)) as u32;
    return (r << 16) + (g << 8) + b;
}

#[inline(always)]
pub fn rgbf32_to_rgb8_f3(value: &Float3) -> u32
{
    let r: u32 = (255.0 * value.x.min(1.0)) as u32;
    let g: u32 = (255.0 * value.y.min(1.0)) as u32;
    let b: u32 = (255.0 * value.z.min(1.0)) as u32;
    return (r << 16) + (g << 8) + b;
}
