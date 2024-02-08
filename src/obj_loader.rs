use crate::material::Material;
use crate::math::*;
use crate::render_components::*;

fn compute_triangle_normals(triangles: &Vec<Triangle>) -> Vec<Float3>
{
    let mut triangle_normals: Vec<Float3> = Vec::with_capacity(triangles.len());
    for triangle in triangles
    {
        let v0v1 = triangle.vertex1 - triangle.vertex0;
        let v0v2 = triangle.vertex2 - triangle.vertex0;
        triangle_normals.push(normalize(&cross(&v0v1, &v0v2)));
    }
    return triangle_normals;
}

fn compute_vertex_normals(triangle_normals: &Vec<Float3>, triangle_vertex_ids: &Vec<Uint3>, num_vertices: usize) -> Vec<Float3>
{
    let mut vertex_normals: Vec<Float3> = Vec::with_capacity(num_vertices);
    let mut vertex_triangle_count: Vec<u32> = Vec::with_capacity(num_vertices);

    for i in 0..num_vertices
    {
        vertex_normals.push(Float3::zero());
        vertex_triangle_count.push(0);
    }

    let num_triangles = triangle_normals.len();
    for triangle_idx in 0..num_triangles
    {
        let ids: [u32; 3] = [triangle_vertex_ids[triangle_idx].x, triangle_vertex_ids[triangle_idx].y, triangle_vertex_ids[triangle_idx].z];
        let normal = triangle_normals[triangle_idx];
        for i in ids
        {
            let i = i as usize;
            vertex_normals[i] += normal;
            vertex_triangle_count[i] += 1;
        }
    }

    for i in 0..num_vertices
    {
        vertex_normals[i] = vertex_normals[i] / (vertex_triangle_count[i] as f32);
    }

    return vertex_normals;
}

pub fn load_obj(path: &std::path::Path) -> (Vec<Mesh>, Vec<Material>)
{
    let mut options = tobj::LoadOptions::default();
    options.triangulate = true;

    let (models, _) = tobj::load_obj(path, &options).expect("Failed to load obj file");

    let mut meshes: Vec<Mesh> = Vec::new();
    let mats: Vec<Material> = Vec::new();

    for m in &models
    {
        let mesh = &m.mesh;
        let mat_id = mesh.material_id;

        let mut vertices: Vec<Float3> = Vec::with_capacity(mesh.positions.len() / 3);
        let mut triangles: Vec<Triangle> = Vec::with_capacity(mesh.indices.len() / 3);
        let mut triangle_vertex_ids: Vec<Uint3> = Vec::with_capacity(mesh.indices.len() / 3);

        for vtx in 0..mesh.positions.len() / 3
        {
            vertices.push(Float3::from_xyz(
                mesh.positions[3 * vtx + 0],
                mesh.positions[3 * vtx + 1],
                mesh.positions[3 * vtx + 2]
            ));
        }

        for vtx in 0..mesh.indices.len() / 3
        {
            let vertex_id = Uint3::from_xyz(mesh.indices[3 * vtx + 0], mesh.indices[3 * vtx + 1], mesh.indices[3 * vtx + 2]);

            triangles.push(Triangle {
                tri_idx: vtx as u32,
                vertex0: vertices[vertex_id.x as usize],
                vertex1: vertices[vertex_id.y as usize],
                vertex2: vertices[vertex_id.z as usize],
            });

            triangle_vertex_ids.push(vertex_id);
        }

        let triangle_normals = compute_triangle_normals(&triangles);
        let vertex_normals = compute_vertex_normals(&triangle_normals, &triangle_vertex_ids, vertices.len());

        meshes.push(Mesh{
            triangles,
            triangle_vertex_ids,
            vertex_normals,
            vertex_uvs: vec![]
        });
    }


    return (meshes, mats);
}