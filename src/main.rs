use std::ffi::CString;

extern crate cl3;

mod input;
mod math;
mod camera;
mod obj_loader;
mod opencl;
mod opengl;
mod application;
mod timer;
mod surface;
mod profiler;
mod renderer;
mod scene;
mod render_components;
mod material;
mod bvh_construction;

use surface::*;
use crate::opengl::{draw_quad, GLTexture, Shader, TextureType};
use crate::application::Application;
use crate::timer::Timer;
use imgui_glfw_rs::glfw;
use imgui_glfw_rs::glfw::*;
use imgui_glfw_rs::imgui;
use imgui_glfw_rs::ImguiGLFW;
use input::Input;

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();


    let (mut window, events) = glfw.create_window(SCRWIDTH as u32, SCRHEIGHT as u32, "ray tracer", glfw::WindowMode::Windowed)
        .expect("Failed to create window");

    window.set_all_polling(true);
    window.make_current();
    //println!("gl version: {},{},{}", window.get_context_version().major, window.get_context_version().minor, window.get_context_version().patch);

    gl::load_with(|s| glfw.get_proc_address_raw(s));
    gl::Viewport::load_with(|s| glfw.get_proc_address_raw(s));

    let shader: Shader = Shader::compile(
        CString::new("#version 330\nin vec4 p;\nin vec2 t;out vec2 u;void main(){u=t;gl_Position=p;}").unwrap(),
        CString::new("#version 330\nuniform sampler2D c;in vec2 u;out vec4 f;void main(){f=/*sqrt*/(texture(c,u));}").unwrap()
    );
    let mut render_target: GLTexture = GLTexture::new(SCRWIDTH as u32, SCRHEIGHT as u32, TextureType::INTTARGET);
    let mut application: Application = Application::new();

    let mut imgui = imgui::Context::create();
    let mut imgui_glfw = ImguiGLFW::new(&mut imgui, &mut window);

    let mut frame_nr: i64 = -1;

    let mut timer: Timer = Timer::new();
    let mut delta_time: f32;
    let mut input: Input = Input::new();

    while !window.should_close()
    {
        delta_time = ((timer.elapsed() as f32) / 1000.0).min(500.0);
        timer.reset();
        application.tick(delta_time, &input);

        frame_nr = frame_nr + 1;
        if frame_nr == 0
        {
            continue;
        }

        render_target.copy_from_pixels(&application.renderer.output_buffer.host_buffer);

        shader.bind();
        shader.set_input_texture(0, CString::new("c").unwrap(), &render_target);
        draw_quad();
        shader.unbind();
        let mut ui = imgui_glfw.frame(&mut window, &mut imgui);

        application.ui(&mut ui);

        imgui_glfw.draw(ui, &mut window);

        window.swap_buffers();
        input.tick();
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, &event);
            imgui_glfw.handle_event(&mut imgui, &event);

            match event {
                glfw::WindowEvent::Key(key, _, action, _) =>
                    {
                        input.set_key(key as u32, action != Action::Release);
                    }
                glfw::WindowEvent::Focus(focussed) =>
                    {
                        input.set_focus(focussed);
                    }
                _ => {}
            }
        }
    }

    application.shutdown();
}

fn handle_window_event(window: &mut glfw::Window, event: &glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true)
        }
        _ => {}
    }
}