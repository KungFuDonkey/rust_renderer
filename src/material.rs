use crate::math::*;

pub fn convert_color_to_u32(color: &Float3) -> u32
{
    let r = (color.x * 255.0) as u32;
    let g = (color.y * 255.0) as u32;
    let b = (color.z * 255.0) as u32;

    return (r << 16) + (g << 8) + b;
}

pub struct Material
{
    pub colors: Vec<Float3>,
    pub reflectiveness: Vec<f32>,
    pub refractive_indices: Vec<f32>
}