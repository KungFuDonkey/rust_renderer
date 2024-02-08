use std::ops::{Deref};
use crate::camera::Camera;
use crate::timer::{FrameTimer,Timer};
use imgui_glfw_rs::imgui::Ui;
use imgui_glfw_rs::imgui::ImString;
use log::info;
use crate::input::Input;
use crate::opencl::OpenCL;
use crate::profiler::Profiler;
use crate::renderer::Renderer;
use crate::scene::Scene;

pub struct Application
{
    cl: OpenCL,
    pub renderer: Renderer,
    scene: Scene,
    camera: Camera,
    profiler: Profiler,
    is_rendering: bool,
}

impl Application
{
    pub fn new() -> Self
    {
        let camera = Camera::new();

        let cl = OpenCL::init();
        let mut renderer = Renderer::new(&cl);
        let mut scene = Scene::new(&cl);

        renderer.set_scene(&scene);
        renderer.set_camera(&camera);

        let app = Application {
            cl,
            renderer,
            scene,
            camera,
            profiler: Profiler::new(),
            is_rendering: true,
        };

        info!("Application initialized");

        return app;
    }

    pub fn tick(&mut self, delta_time: f32, input: &Input)
    {
        self.handle_input(input, delta_time);

        if !self.is_rendering
        {
            return;
        }

        self.renderer.render(&self.cl, &self.scene);
    }

    fn handle_input(&mut self, input: &Input, delta_time: f32)
    {
        if self.camera.handle_input(&input, delta_time)
        {
            // actions to perform when the camera moves
            self.renderer.set_camera(&self.camera);
        }
    }

    pub fn ui(&mut self, ui: &mut Ui)
    {
        if self.is_rendering
        {
            let render_string = format!("ms: {}\nfps: {}\n", self.profiler.ms, self.profiler.fps);

            ui.text(ImString::new(render_string).deref());
        }
        else
        {
            ui.text(ImString::new("paused").deref());
        }
    }

    pub fn shutdown(&mut self)
    {
        info!("Application shut down");
    }
}