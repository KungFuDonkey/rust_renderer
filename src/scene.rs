
use std::f32::consts::PI;
use crate::material::*;
use crate::math::*;
use crate::render_components::*;
use crate::obj_loader::*;
use crate::opencl::{OpenCL, OpenCLBuffer};

pub struct SceneObject
{
    pub mesh_idx: u32,
    pub mat_idx: u32,
    pub transform: Mat4,
    pub inv_transform: Mat4,
    pub children: Vec<SceneObject>
}

impl SceneObject
{
    pub fn new(mesh_idx: u32, mat_idx: u32, transform: Mat4,children: Vec<SceneObject>) -> Self
    {
        SceneObject{
            mesh_idx,
            mat_idx,
            transform,
            inv_transform: transform.inverted(),
            children
        }
    }
}

pub struct SceneDescription
{
    pub root_objects: Vec<SceneObject>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>
}


impl SceneDescription
{
    pub fn new() -> Self
    {
        let mut materials = vec![];

        let mut meshes: Vec<Mesh> = Vec::new();

        let mut root_objects: Vec<SceneObject> = Vec::new();

        let transform = Mat4::translate( &Float3::from_xyz(2.0, 0.0, 0.5)) * Mat4::scale(0.5);
        let (msh, mts) = load_obj(&std::path::Path::new("./assets/suzanne.obj"));

        root_objects.push(SceneObject::new(1, 0, transform, vec![]));

        for mesh in msh
        {
            meshes.push(mesh);
        }

        for material in mts
        {
            materials.push(material);
        }

        materials.push(Material{
            colors: vec![Float3::from_xyz(1.0,0.0,0.0)],
            reflectiveness: vec![0.0],
            refractive_indices: vec![0.0]
        });

        let mut scene = SceneDescription{
            root_objects,
            meshes,
            materials
        };

        return scene;
    }
}

pub struct Scene
{
    pub obj_mesh_ids: OpenCLBuffer<u32>,
    pub obj_mat_ids: OpenCLBuffer<u32>,
    pub obj_transforms: OpenCLBuffer<Mat4>,
    pub obj_inv_transforms: OpenCLBuffer<Mat4>,

    pub bvh_offsets: OpenCLBuffer<u32>,
    pub mesh_offsets: OpenCLBuffer<u32>,
    pub bvh_min_bounds: OpenCLBuffer<Float3>,
    pub bvh_max_bounds: OpenCLBuffer<Float3>,
    pub bvh_tri_counts: OpenCLBuffer<u32>,
    pub bvh_left_firsts: OpenCLBuffer<u32>,

    pub bvh_triangle_offsets: OpenCLBuffer<u32>,
    pub bvh_triangles: OpenCLBuffer<Triangle>,

    pub mesh_vertex_ids: OpenCLBuffer<Uint3>,
    pub mesh_vertex_normals: OpenCLBuffer<Float3>,

    pub mat_offsets: OpenCLBuffer<u32>,
    pub mat_colors: OpenCLBuffer<u32>,
    pub mat_reflectiveness: OpenCLBuffer<u8>,
    pub mat_refraction_indices: OpenCLBuffer<f32>,

    pub bvhs: Vec<BVH>
}

impl Scene
{
    pub fn new(cl: &OpenCL) -> Self
    {
        let scene = SceneDescription::new();

        return Scene::from_scene_description(cl, &scene);
    }

    // create gpu scene based on scene
    pub fn from_scene_description(cl: &OpenCL, scene: &SceneDescription) -> Self
    {
        let mut obj_mesh_ids: Vec<u32> = Vec::new();
        let mut obj_mat_ids: Vec<u32> = Vec::new();
        let mut obj_transforms: Vec<Mat4> = Vec::new();
        let mut obj_inv_transforms: Vec<Mat4> = Vec::new();

        let mut bvh_offsets: Vec<u32> = Vec::new();
        let mut mesh_offsets: Vec<u32> = Vec::new();

        let mut bvh_min_bounds: Vec<Float3> = Vec::new();
        let mut bvh_max_bounds: Vec<Float3> = Vec::new();
        let mut bvh_tri_counts: Vec<u32> = Vec::new();
        let mut bvh_left_firsts: Vec<u32> = Vec::new();
        let mut bvh_triangle_offsets: Vec<u32> = Vec::new();
        let mut bvh_triangles: Vec<Triangle> = Vec::new();

        let mut mesh_vertex_ids: Vec<Uint3> = Vec::new();
        let mut mesh_vertex_normals: Vec<Float3> = Vec::new();

        let mut mat_offsets: Vec<u32> = Vec::new();
        let mut mat_colors: Vec<u32> = Vec::new();
        let mut mat_reflectiveness: Vec<u8> = Vec::new();
        let mut mat_refraction_indices: Vec<f32> = Vec::new();

        for object in &scene.root_objects
        {
            obj_mesh_ids.push(object.mesh_idx);
            obj_mat_ids.push(object.mat_idx);
            obj_transforms.push(object.transform);
            obj_inv_transforms.push(object.inv_transform);
        }

        let mut bvhs: Vec<BVH> = Vec::new();
        let mut bvh_offset = 0;
        let mut mesh_offset = 0;
        let mut triangle_offset = 0;
        for mesh in &scene.meshes
        {
            let bvh = BVH::from_mesh(&mesh.triangles, &compute_bounds_from_triangles(&mesh.triangles), 4);

            bvh_offsets.push(bvh_offset);
            bvh_offset += bvh.bvh_nodes.len() as u32;

            for bvh_node in &bvh.bvh_nodes
            {
                bvh_min_bounds.push(bvh_node.bounds.min_bound);
                bvh_max_bounds.push(bvh_node.bounds.max_bound);
                bvh_tri_counts.push(bvh_node.tri_count as u32);
                bvh_left_firsts.push(bvh_node.left_first as u32);
            }

            for id in &bvh.triangle_idx
            {
                bvh_triangles.push(mesh.triangles[*id]);
            }

            for ids in &mesh.triangle_vertex_ids
            {
                mesh_vertex_ids.push(*ids);
            }

            for normal in &mesh.vertex_normals
            {
                mesh_vertex_normals.push(*normal);
            }

            mesh_offsets.push(mesh_offset);
            mesh_offset += mesh.triangle_vertex_ids.len() as u32;

            bvh_triangle_offsets.push(triangle_offset);
            triangle_offset += bvh.triangle_idx.len() as u32;
            bvhs.push(bvh);
        }

        let mut mat_offset = 0;
        for material in &scene.materials
        {
            mat_offsets.push(mat_offset);
            mat_offset += material.colors.len() as u32;
            for color in &material.colors
            {
                mat_colors.push(convert_color_to_u32(color));
            }

            for reflectiveness in &material.reflectiveness
            {
                mat_reflectiveness.push((*reflectiveness * 255.0) as u8);
            }

            for refractive_index in &material.refractive_indices
            {
                mat_refraction_indices.push(*refractive_index);
            }
        }


        let obj_mesh_ids = OpenCLBuffer::read_write(cl, obj_mesh_ids);
        let obj_mat_ids = OpenCLBuffer::read_write(cl, obj_mat_ids);
        let obj_transforms = OpenCLBuffer::read_write(cl, obj_transforms);
        let obj_inv_transforms = OpenCLBuffer::read_write(cl, obj_inv_transforms);
        let bvh_offsets = OpenCLBuffer::read_write(cl, bvh_offsets);
        let mesh_offsets = OpenCLBuffer::read_write(cl, mesh_offsets);
        let bvh_min_bounds = OpenCLBuffer::read_write(cl, bvh_min_bounds);
        let bvh_max_bounds = OpenCLBuffer::read_write(cl, bvh_max_bounds);
        let bvh_tri_counts = OpenCLBuffer::read_write(cl, bvh_tri_counts);
        let bvh_left_firsts = OpenCLBuffer::read_write(cl, bvh_left_firsts);
        let bvh_triangle_offsets = OpenCLBuffer::read_write(cl, bvh_triangle_offsets);
        let bvh_triangles = OpenCLBuffer::read_write(cl, bvh_triangles);
        let mesh_vertex_ids = OpenCLBuffer::read_write(cl, mesh_vertex_ids);
        let mesh_vertex_normals = OpenCLBuffer::read_write(cl, mesh_vertex_normals);
        let mat_offsets = OpenCLBuffer::read_write(cl, mat_offsets);
        let mat_colors = OpenCLBuffer::read_write(cl, mat_colors);
        let mat_reflectiveness = OpenCLBuffer::read_write(cl, mat_reflectiveness);
        let mat_refraction_indices = OpenCLBuffer::read_write(cl, mat_refraction_indices);

        obj_mesh_ids.copy_to_device(cl);
        obj_mat_ids.copy_to_device(cl);
        obj_transforms.copy_to_device(cl);
        obj_inv_transforms.copy_to_device(cl);
        bvh_offsets.copy_to_device(cl);
        mesh_offsets.copy_to_device(cl);
        bvh_min_bounds.copy_to_device(cl);
        bvh_max_bounds.copy_to_device(cl);
        bvh_tri_counts.copy_to_device(cl);
        bvh_left_firsts.copy_to_device(cl);
        bvh_triangle_offsets.copy_to_device(cl);
        bvh_triangles.copy_to_device(cl);
        mesh_vertex_ids.copy_to_device(cl);
        mesh_vertex_normals.copy_to_device(cl);
        mat_offsets.copy_to_device(cl);
        mat_colors.copy_to_device(cl);
        mat_reflectiveness.copy_to_device(cl);
        mat_refraction_indices.copy_to_device(cl);

        return Scene
        {
            obj_mesh_ids,
            obj_mat_ids,
            obj_transforms,
            obj_inv_transforms,
            bvh_offsets,
            mesh_offsets,
            bvh_min_bounds,
            bvh_max_bounds,
            bvh_tri_counts,
            bvh_left_firsts,
            bvh_triangle_offsets,
            bvh_triangles,
            mesh_vertex_ids,
            mesh_vertex_normals,
            mat_offsets,
            mat_colors,
            mat_reflectiveness,
            mat_refraction_indices,
            bvhs
        }
    }
}