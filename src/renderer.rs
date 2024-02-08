use std::path;
use log::info;
use crate::math::{Float3, random_uint_s};
use crate::opencl::*;
use crate::renderer::RenderMode::Albedo;
use crate::surface::{SCRHEIGHT, SCRWIDTH};
use image::GenericImageView;
use crate::camera::Camera;
use crate::scene::Scene;

#[derive(PartialEq, Copy, Clone)]
pub enum RenderMode
{
    PathTracing,
    Normals,
    Albedo,
    AccumulatedLight,
    LightLayer
}

#[derive(PartialEq, Copy, Clone)]
pub struct RenderSettings
{
    pub render_mode: RenderMode,
    pub num_primary_rays: usize,
    pub num_bounces: usize,
}

pub fn load_blue_noise_from_file(cl: &OpenCL, file: std::path::PathBuf) -> OpenCLBuffer<u8>
{
    let img = image::open(file).expect("Blue noise not found");

    let mut pixels: Vec<u8> = vec![0; (img.width() * img.height()) as usize];
    for (x,y, value) in img.pixels()
    {
        let r = value[0] as u8;
        pixels[(y * img.height() + x) as usize] = r;
    }

    return OpenCLBuffer::read_only(cl, pixels);
}

pub struct Renderer
{
    pub settings: RenderSettings,
    pub last_settings: RenderSettings,

    // path tracing kernels
    generate_rays_kernel: OpenCLKernel,
    extend_kernel: OpenCLKernel,
    shade_kernel: OpenCLKernel,
    connect_kernel: OpenCLKernel,

    // final kernels
    albedo_kernel: OpenCLKernel,
    finalize_kernel: OpenCLKernel,

    num_rays: OpenCLBuffer<u32>,
    ray_write_back_ids: OpenCLBuffer<u32>,
    ray_ts: OpenCLBuffer<f32>,
    ray_origins: OpenCLBuffer<Float3>,
    ray_directions: OpenCLBuffer<Float3>,
    ray_normals: OpenCLBuffer<Float3>,

    ray_energies: OpenCLBuffer<Float3>,
    ray_intersection_colors: OpenCLBuffer<Float3>,
    ray_write_back_lights: OpenCLBuffer<Float3>,

    shadow_ray_ts: OpenCLBuffer<f32>,
    shadow_ray_origins: OpenCLBuffer<Float3>,
    shadow_ray_directions: OpenCLBuffer<Float3>,
    shadow_ray_write_back_ids: OpenCLBuffer<u32>,
    shadow_ray_write_back_lights: OpenCLBuffer<Float3>,

    albedo: OpenCLBuffer<Float3>,
    light: OpenCLBuffer<Float3>,
    light_accumulator: OpenCLBuffer<Float3>,

    // render target
    pub output_buffer: OpenCLBuffer<u32>,

    // rng
    seed: u32,
    blue_noise_texture: OpenCLBuffer<u8>,

    // for accumulation
    rendered_frames: u32,
}

impl Renderer
{
    pub fn new(cl: &OpenCL) -> Self
    {
        info!("generating ray kernels");

        let generate_rays_program = OpenCLProgram::from_file(cl, path::Path::new("./src/kernels/generate_rays.cl"));
        let generate_rays_kernel = OpenCLKernel::from_program(cl, &generate_rays_program, "generate_rays");

        let extend_program = OpenCLProgram::from_file(cl, path::Path::new("./src/kernels/extend.cl"));
        let extend_kernel = OpenCLKernel::from_program(cl, &extend_program, "extend");

        let shade_program = OpenCLProgram::from_file(cl, path::Path::new("./src/kernels/shade.cl"));
        let shade_kernel = OpenCLKernel::from_program(cl, &shade_program, "shade");

        let connect_program = OpenCLProgram::from_file(cl, path::Path::new("./src/kernels/connect.cl"));
        let connect_kernel = OpenCLKernel::from_program(cl, &connect_program, "connect");

        let finalize_program = OpenCLProgram::from_file(cl, path::Path::new("./src/kernels/finalize.cl"));
        let finalize_kernel = OpenCLKernel::from_program(cl, &finalize_program, "finalize");
        let albedo_kernel = OpenCLKernel::from_program(cl, &finalize_program, "albedo");

        info!("generating ray kernels -- finished");

        let num_primary_rays = SCRWIDTH * SCRHEIGHT;
        let num_bounces = 10;

        let mut num_rays: Vec<u32> = Vec::with_capacity(num_bounces * 2);
        for _ in 0..(num_bounces + 1)
        {
            num_rays.push(0);
            num_rays.push(0);
        }
        num_rays[0] = num_primary_rays as u32; // set initial rays

        let mut ray_write_back_ids: Vec<u32> = Vec::new();
        let mut ray_ts: Vec<f32> = Vec::new();
        let mut ray_origins: Vec<Float3> = Vec::new();
        let mut ray_directions: Vec<Float3> = Vec::new();
        let mut ray_normals: Vec<Float3> = Vec::new();

        let mut ray_energies: Vec<Float3> = Vec::new();
        let mut ray_intersection_colors: Vec<Float3> = Vec::new();
        let mut ray_write_back_lights: Vec<Float3> = Vec::new();

        let mut shadow_ray_ts: Vec<f32> = Vec::new();
        let mut shadow_ray_origins: Vec<Float3> = Vec::new();
        let mut shadow_ray_directions: Vec<Float3> = Vec::new();
        let mut shadow_ray_write_back_ids: Vec<u32> = Vec::new();
        let mut shadow_ray_write_back_lights: Vec<Float3> = Vec::new();

        let mut albedo: Vec<Float3> = Vec::new();
        let mut light: Vec<Float3> = Vec::new();
        let mut light_accumulator: Vec<Float3> = Vec::new();

        // render target
        let mut output_buffer: Vec<u32> = Vec::new();

        // populate buffers
        for _ in 0..num_primary_rays
        {
            ray_write_back_ids.push(0);
            ray_ts.push(0.0);
            ray_origins.push(Float3::zero());
            ray_directions.push(Float3::zero());
            ray_normals.push(Float3::zero());
            output_buffer.push(0);
            ray_energies.push(Float3::zero());
            ray_intersection_colors.push(Float3::zero());
            ray_write_back_lights.push(Float3::zero());
            shadow_ray_write_back_ids.push(0);
            shadow_ray_ts.push(0.0);
            shadow_ray_origins.push(Float3::zero());
            shadow_ray_directions.push(Float3::zero());
            shadow_ray_write_back_lights.push(Float3::zero());
            albedo.push(Float3::zero());
        }

        for _ in 0..(num_bounces + 1)
        {
            for _ in 0..num_primary_rays
            {
                light.push(Float3::zero());
                light_accumulator.push(Float3::zero())
            }
        }

        let settings = RenderSettings
        {
            render_mode: Albedo,
            num_primary_rays,
            num_bounces
        };

        let num_rays = OpenCLBuffer::read_write(cl, num_rays);
        let ray_write_back_ids = OpenCLBuffer::read_write(cl, ray_write_back_ids);
        let ray_ts = OpenCLBuffer::read_write(cl, ray_ts);
        let ray_origins = OpenCLBuffer::read_write(cl, ray_origins);
        let ray_directions = OpenCLBuffer::read_write(cl, ray_directions);
        let ray_normals = OpenCLBuffer::read_write(cl, ray_normals);
        let ray_energies = OpenCLBuffer::read_write(cl, ray_energies);
        let ray_intersection_colors = OpenCLBuffer::read_write(cl, ray_intersection_colors);
        let ray_write_back_lights = OpenCLBuffer::read_write(cl, ray_write_back_lights);
        let shadow_ray_ts = OpenCLBuffer::read_write(cl, shadow_ray_ts);
        let shadow_ray_origins = OpenCLBuffer::read_write(cl, shadow_ray_origins);
        let shadow_ray_directions = OpenCLBuffer::read_write(cl, shadow_ray_directions);
        let shadow_ray_write_back_ids = OpenCLBuffer::read_write(cl, shadow_ray_write_back_ids);
        let shadow_ray_write_back_lights = OpenCLBuffer::read_write(cl, shadow_ray_write_back_lights);
        let albedo = OpenCLBuffer::read_write(cl, albedo);
        let light = OpenCLBuffer::read_write(cl, light);
        let light_accumulator = OpenCLBuffer::read_write(cl, light_accumulator);
        let output_buffer = OpenCLBuffer::read_write(cl, output_buffer);

        num_rays.copy_to_device(cl);
        ray_write_back_ids.copy_to_device(cl);
        ray_ts.copy_to_device(cl);
        ray_origins.copy_to_device(cl);
        ray_directions.copy_to_device(cl);
        ray_normals.copy_to_device(cl);
        ray_energies.copy_to_device(cl);
        ray_intersection_colors.copy_to_device(cl);
        ray_write_back_lights.copy_to_device(cl);
        shadow_ray_ts.copy_to_device(cl);
        shadow_ray_origins.copy_to_device(cl);
        shadow_ray_directions.copy_to_device(cl);
        shadow_ray_write_back_ids.copy_to_device(cl);
        shadow_ray_write_back_lights.copy_to_device(cl);
        albedo.copy_to_device(cl);
        light.copy_to_device(cl);
        light_accumulator.copy_to_device(cl);
        output_buffer.copy_to_device(cl);

        generate_rays_kernel.set_argument(0, 320340458);
        generate_rays_kernel.set_argument(1, SCRWIDTH as u32);
        generate_rays_kernel.set_argument(2, SCRHEIGHT as u32);
        generate_rays_kernel.set_argument(3, num_bounces as u32);

        generate_rays_kernel.set_argument(8, &ray_write_back_ids);
        generate_rays_kernel.set_argument(9, &ray_ts);
        generate_rays_kernel.set_argument(10, &ray_origins);
        generate_rays_kernel.set_argument(11, &ray_directions);
        generate_rays_kernel.set_argument(12, &ray_normals);
        generate_rays_kernel.set_argument(13, &ray_intersection_colors);
        generate_rays_kernel.set_argument(14, &ray_energies);
        generate_rays_kernel.set_argument(15, &ray_write_back_lights);
        generate_rays_kernel.set_argument(16, &albedo);
        generate_rays_kernel.set_argument(17, &light);

        albedo_kernel.set_argument(0, &albedo);
        albedo_kernel.set_argument(1, &output_buffer);

        return Renderer{
            settings,
            last_settings: settings,
            generate_rays_kernel,
            extend_kernel,
            shade_kernel,
            connect_kernel,
            finalize_kernel,
            albedo_kernel,
            num_rays,
            ray_write_back_ids,
            ray_ts,
            ray_origins,
            ray_directions,
            ray_normals,

            ray_energies,
            ray_intersection_colors,
            ray_write_back_lights,

            shadow_ray_ts,
            shadow_ray_origins,
            shadow_ray_directions,
            shadow_ray_write_back_ids,
            shadow_ray_write_back_lights,

            albedo,
            light,
            light_accumulator,

            // render target
            output_buffer,

            // rng
            seed: 320340458,
            blue_noise_texture: load_blue_noise_from_file(cl, std::path::PathBuf::from("./assets/blue_noise.png")),

            // for accumulation
            rendered_frames: 1,
        }
    }

    pub fn set_camera(&mut self, camera: &Camera)
    {
        self.generate_rays_kernel.set_argument(4, &camera.position);
        self.generate_rays_kernel.set_argument(5, &camera.top_left);
        self.generate_rays_kernel.set_argument(6, &camera.bottom_left);
        self.generate_rays_kernel.set_argument(7, &camera.top_right);
        self.rendered_frames = 1;
    }

    pub fn set_scene(&mut self, scene: &Scene)
    {
        self.generate_rays_kernel.set_argument(18, scene.obj_mesh_ids.host_buffer.len() as u32);
        self.generate_rays_kernel.set_argument(19, &scene.obj_mesh_ids);
        self.generate_rays_kernel.set_argument(20, &scene.obj_mat_ids);
        self.generate_rays_kernel.set_argument(21, &scene.obj_transforms);
        self.generate_rays_kernel.set_argument(22, &scene.obj_inv_transforms);
        self.generate_rays_kernel.set_argument(23, &scene.bvh_offsets);
        self.generate_rays_kernel.set_argument(24, &scene.mesh_offsets);
        self.generate_rays_kernel.set_argument(25, &scene.bvh_min_bounds);
        self.generate_rays_kernel.set_argument(26, &scene.bvh_max_bounds);
        self.generate_rays_kernel.set_argument(27, &scene.bvh_tri_counts);
        self.generate_rays_kernel.set_argument(28, &scene.bvh_left_firsts);
        self.generate_rays_kernel.set_argument(29, &scene.bvh_triangle_offsets);
        self.generate_rays_kernel.set_argument(30, &scene.bvh_triangles);
        self.generate_rays_kernel.set_argument(31, &scene.mesh_vertex_ids);
        self.generate_rays_kernel.set_argument(32, &scene.mesh_vertex_normals);
        self.generate_rays_kernel.set_argument(33, &scene.mat_offsets);
        self.generate_rays_kernel.set_argument(34, &scene.mat_colors);
        self.generate_rays_kernel.set_argument(35, &scene.mat_reflectiveness);
        self.generate_rays_kernel.set_argument(36, &scene.mat_refraction_indices);
    }

    pub fn render(&mut self, cl: &OpenCL, scene: &Scene)
    {

        self.generate_rays_kernel.set_argument(0, self.seed);
        random_uint_s(&mut self.seed);

        self.generate_rays_kernel.run2d(cl, SCRWIDTH, SCRHEIGHT);

        self.albedo_kernel.run(cl, self.settings.num_primary_rays);

        self.output_buffer.copy_from_device(cl);

        cl.flush_queue();
        self.num_rays.copy_to_device(cl);
    }
}

