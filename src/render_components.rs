use crate::math::*;
use crate::bvh_construction::*;

#[derive(Clone, Copy)]
pub struct Triangle
{
    pub tri_idx: u32,
    pub vertex0: Float3,
    pub vertex1: Float3,
    pub vertex2: Float3,
}

#[derive(Clone, Copy)]
pub struct AABB
{
    pub min_bound: Float3,
    pub max_bound: Float3
}


impl AABB
{
    pub fn from_empty() -> Self
    {
        AABB{
            min_bound: Float3::from_a(1e34),
            max_bound: Float3::from_a(-1e34)
        }
    }

    pub fn from_bounds(min_bound: &Float3, max_bound: &Float3) -> Self
    {
        AABB{
            min_bound: *min_bound,
            max_bound: *max_bound
        }
    }

    pub fn reset(&mut self)
    {
        self.min_bound = Float3::from_a(1e34);
        self.max_bound = Float3::from_a(-1e34);
    }

    pub fn contains(&self, vector: &Float3) -> bool
    {
        let p = *vector;
        let va = p - self.min_bound;
        let vb = self.max_bound - p;
        return va.x >= 0.0 && va.y >= 0.0 && va.z >= 0.0 && vb.x >= 0.0 && vb.y >= 0.0 && vb.z >= 0.0;
    }

    pub fn contains2(&self, vector: &Float3) -> bool
    {
        return self.min_bound.x <= vector.x && self.min_bound.y <= vector.y && self.min_bound.z <= vector.z &&
            self.max_bound.x >= vector.x && self.max_bound.y >= vector.y && self.max_bound.z >= vector.z;
    }

    pub fn grow_aabb(&mut self, bb: &AABB)
    {
        self.min_bound = self.min_bound.min(&bb.min_bound);
        self.max_bound = self.max_bound.max(&bb.max_bound);
    }

    pub fn grow(&mut self, vector: &Float3)
    {
        self.min_bound = self.min_bound.min(vector);
        self.max_bound = self.max_bound.max(vector);
    }

    pub fn union(&self, other: &AABB) -> Self
    {
        let min_bound = self.min_bound.min(&other.min_bound);
        let max_bound = self.max_bound.max(&other.max_bound);

        AABB{
            min_bound,
            max_bound
        }
    }

    pub fn intersection(&self, other: &AABB) -> Self
    {
        let min_bound = self.min_bound.max(&other.min_bound);
        let max_bound = self.max_bound.min(&other.max_bound);

        AABB{
            min_bound,
            max_bound
        }
    }

    pub fn extend_x(&self) -> f32
    {
        return self.max_bound.x - self.min_bound.x;
    }

    pub fn extend_y(&self) -> f32
    {
        return self.max_bound.y - self.min_bound.y;
    }

    pub fn extend_z(&self) -> f32
    {
        return self.max_bound.z - self.min_bound.z;
    }

    pub fn area(&self) -> f32
    {
        let e = self.max_bound - self.min_bound;
        let zero: f32 = 0.0;
        return zero.max(e.x * e.y + e.x * e.z + e.y * e.z);
    }

    pub fn set_bounds(&mut self, max_bound: &Float3, min_bound: &Float3)
    {
        self.min_bound = *min_bound;
        self.max_bound = *max_bound;
    }

    pub fn center(&self, axis: usize) -> f32
    {
        match axis
        {
            0 => (self.min_bound.x + self.max_bound.x) * 0.5,
            1 => (self.min_bound.y + self.max_bound.y) * 0.5,
            2 => (self.min_bound.z + self.max_bound.z) * 0.5,
            _ => panic!("Outside of axis range")
        }
    }

    pub fn minimal(&self, axis: usize) -> f32
    {
        match axis
        {
            0 => self.min_bound.x,
            1 => self.min_bound.y,
            2 => self.min_bound.z,
            _ => panic!("Outside of axis range")
        }
    }

    pub fn maximal(&self, axis: usize) -> f32
    {
        match axis
        {
            0 => self.max_bound.x,
            1 => self.max_bound.y,
            2 => self.max_bound.z,
            _ => panic!("Outside of axis range")
        }
    }
}

pub fn compute_bounds_from_triangles(triangles: &Vec<Triangle>) -> AABB
{
    // calculate bounding box
    let mut bounds = AABB::from_empty();
    for t in triangles
    {
        bounds.grow(&t.vertex0);
        bounds.grow(&t.vertex1);
        bounds.grow(&t.vertex2);
    }
    return bounds;
}

fn test_plane_intersection(normal_axis: f32, e0x: f32, e0y: f32, e1x: f32, e1y: f32, e2x: f32, e2y: f32, v0x: f32, v0y: f32, v1x: f32, v1y: f32, v2x: f32, v2y: f32, dpx: f32, dpy: f32, px: f32, py: f32) -> bool
{
    let mut m = 1.0;
    if normal_axis < 0.0
    {
        m = -1.0;
    }

    let ne0 = Float3::from_xyz(-e0y, e0x, 0.0) * m;
    let ne1 = Float3::from_xyz(-e1y, e1x, 0.0) * m;
    let ne2 = Float3::from_xyz(-e2y, e2x, 0.0) * m;

    let v0 = Float3::from_xyz(v0x, v0y, 0.0);
    let v1 = Float3::from_xyz(v1x, v1y, 0.0);
    let v2 = Float3::from_xyz(v2x, v2y, 0.0);

    let zero: f32 = 0.0;
    let de0 = -dot(&ne0, &v0) + zero.max(dpx * ne0.x) + zero.max(dpy * ne0.y);
    let de1 = -dot(&ne1, &v1) + zero.max(dpx * ne1.x) + zero.max(dpy * ne1.y);
    let de2 = -dot(&ne2, &v2) + zero.max(dpx * ne2.x) + zero.max(dpy * ne2.y);

    let pp = Float3::from_xyz(px, py, 0.0);
    if (dot(&ne0, &pp) + de0) < 0.0 ||
        (dot(&ne1, &pp) + de1) < 0.0 ||
        (dot(&ne1, &pp) + de2) < 0.0
    {
        return false;
    }

    return true;
}

pub fn intersect_aabb_triangle(aabb: &AABB, triangle: &Triangle, triangle_normal: &Float3) -> bool
{
    let n = triangle_normal;
    let p = aabb.min_bound;
    let dp = aabb.max_bound - p;

    let mut c = Float3::zero();
    if n.x > 0.0
    {
        c.x = dp.x;
    }
    if n.y > 0.0
    {
        c.y = dp.y;
    }
    if n.z > 0.0
    {
        c.z = dp.z;
    }

    let d1 = dot(n, &(c - triangle.vertex0));
    let d2 = dot(n, &(dp - c - triangle.vertex0));

    if (dot(n, &p) + d1) * (dot(n,&p) + d2) > 0.0
    {
        return false;
    }

    let edge0 = triangle.vertex1 - triangle.vertex0;
    let edge1 = triangle.vertex2 - triangle.vertex0;
    let edge2 = triangle.vertex2 - triangle.vertex1;

    if !(test_plane_intersection(n.z, edge0.x, edge0.y, edge1.x, edge1.y, edge2.x, edge2.y,
                                 triangle.vertex0.x, triangle.vertex0.y, triangle.vertex1.x, triangle.vertex1.y, triangle.vertex2.x, triangle.vertex2.y,
                                 dp.x, dp.y, p.x, p.y))
    {
        return false;
    }

    if !(test_plane_intersection(n.x, edge0.y, edge0.z, edge1.y, edge1.z, edge2.y, edge2.z,
                                 triangle.vertex0.y, triangle.vertex0.z, triangle.vertex1.y, triangle.vertex1.z, triangle.vertex2.y, triangle.vertex2.z,
                                 dp.y, dp.z, p.y, p.z))
    {
        return false;
    }

    if !(test_plane_intersection(n.y, edge0.z, edge0.x, edge1.z, edge1.x, edge2.z, edge2.x,
                                 triangle.vertex0.z, triangle.vertex0.x, triangle.vertex1.z, triangle.vertex1.x, triangle.vertex2.z, triangle.vertex2.x,
                                 dp.z, dp.x, p.z, p.x))
    {
        return false;
    }

    return true;
}

pub fn triangle_is_in_aabb(aabb: &AABB, triangle: &Triangle, triangle_normal: &Float3) -> bool
{
    aabb.contains2(&triangle.vertex0) || aabb.contains2(&triangle.vertex1) || aabb.contains2(&triangle.vertex2) || intersect_aabb_triangle(aabb, triangle, triangle_normal)
}

#[derive(Clone, Copy)]
pub struct BVHNode
{
    pub bounds: AABB,
    pub tri_count: usize,
    pub left_first: usize,
}

impl BVHNode
{
    pub fn is_leaf(&self) -> bool
    {
        return self.tri_count > 0;
    }
}

pub struct BVH
{
    pub bvh_nodes: Vec<BVHNode>,
    pub triangle_idx: Vec<usize>,
}

impl BVH
{
    pub fn from_mesh(triangles: &Vec<Triangle>, bounds: &AABB, bin_size: usize) -> Self
    {
        let constructor = BVHConstructor::from_mesh(triangles, bounds, bin_size);
        BVH
        {
            bvh_nodes: constructor.bvh_nodes,
            triangle_idx: constructor.triangle_idx
        }
    }
}

pub struct Sphere
{
    pub position: Float3,
    pub radius: f32,
    pub radius2: f32,
    pub inv_radius: f32
}

pub struct Plane
{
    pub normal: Float3,
    pub distance: f32
}

pub struct Mesh
{
    pub triangles: Vec<Triangle>,
    pub triangle_vertex_ids: Vec<Uint3>,
    pub vertex_normals: Vec<Float3>,
    pub vertex_uvs: Vec<Float2>
}