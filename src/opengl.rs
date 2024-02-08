
extern crate gl;

use std::ffi::{c_void, CString};
use std::ptr::null;
use gl::types::*;
use crate::surface::Surface;

#[derive(PartialEq)]
pub enum TextureType {
    DEFAULT,
    FLOAT,
    INTTARGET
}

pub fn create_vbo(data: *const GLfloat, size: usize) -> GLuint
{
    let mut id: GLuint = 0;
    unsafe
        {
            let id_ptr: *mut GLuint = &mut id as *mut GLuint;
            gl::GenBuffers(1, id_ptr);
            gl::BindBuffer(gl::ARRAY_BUFFER, id);
            gl::BufferData(gl::ARRAY_BUFFER, size as GLsizeiptr, data as *const c_void, gl::STATIC_DRAW)
        }

    return id;
}

pub fn bind_vbo(idx: u32, n: u32, id: GLuint)
{
    unsafe
        {
            gl::EnableVertexAttribArray( idx );
            gl::BindBuffer(gl::ARRAY_BUFFER, id);
            gl::VertexAttribPointer(idx, n as GLint, gl::FLOAT, gl::FALSE, 0, null());
        }
}

pub fn draw_quad()
{
    unsafe
        {
            static mut VAO: GLuint = 0;
            if VAO == 0
            {
                let verts: [GLfloat; 18] = [-1.0, 1.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0];
                let uvdata: [GLfloat; 12] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

                // use if it goes wrong?
                //let verts_igp: [GLfloat; 18] = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
                //let uvdata_igp: [GLfloat; 12] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

                let vertex_buffer = create_vbo(verts.as_ptr(), 18 * 4);
                let uv_buffer = create_vbo(uvdata.as_ptr(), 18 * 4);

                let vao_ptr: *mut GLuint = &mut VAO as *mut GLuint;
                gl::GenVertexArrays(1, vao_ptr);
                gl::BindVertexArray(VAO);

                bind_vbo(0, 3, vertex_buffer);
                bind_vbo(1, 2, uv_buffer);
                gl::BindVertexArray( 0 );
            }

            gl::BindVertexArray(VAO);
            gl::DrawArrays(gl::TRIANGLES, 0, 6);
            gl::BindVertexArray( 0 );
        }
}

pub struct GLTexture
{
    pub id: GLuint,
    pub width: u32,
    pub height: u32
}

impl GLTexture
{
    pub fn new(w: u32, h: u32, t: TextureType) -> Self
    {
        let texture: GLTexture;
        unsafe
            {
                let mut texture_id: GLuint = 123;
                let texture_ptr: *mut GLuint = &mut texture_id as *mut GLuint;

                gl::GenTextures(1, texture_ptr);
                gl::BindTexture(gl::TEXTURE_2D, texture_id);

                if t == TextureType::DEFAULT
                {
                    gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGB as GLint, w as GLsizei, h as GLsizei, 0, gl::BGR, gl::UNSIGNED_BYTE, null());
                    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as GLint);
                    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as GLint);
                }
                else if t == TextureType::INTTARGET
                {
                    gl::TexParameteri( gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as GLint );
                    gl::TexParameteri( gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as GLint );
                    gl::TexParameteri( gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as GLint );
                    gl::TexParameteri( gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as GLint );
                    gl::TexImage2D( gl::TEXTURE_2D, 0, gl::RGBA as GLint, w as GLsizei, h as GLsizei, 0, gl::RGBA, gl::UNSIGNED_BYTE, null() );
                }
                else if t == TextureType::FLOAT
                {
                    gl::TexParameteri( gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as GLint );
                    gl::TexParameteri( gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as GLint );
                    gl::TexParameteri( gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as GLint );
                    gl::TexParameteri( gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as GLint );
                    gl::TexImage2D( gl::TEXTURE_2D, 0, gl::RGBA32F as GLint, w as GLsizei, h as GLsizei, 0, gl::RGBA, gl::FLOAT, null() );
                }
                gl::BindTexture( gl::TEXTURE_2D, 0 );

                let error: GLenum = gl::GetError();
                if error != gl::NO_ERROR
                {
                    println!("ERROR ON CREATION OF TEXTURE")
                }

                texture = GLTexture
                {
                    id: texture_id,
                    width: w,
                    height: h
                }
            }
        return texture;
    }

    pub fn copy_from_pixels(&mut self, screen: &Vec<u32>)
    {
        unsafe
            {
                gl::BindTexture(gl::TEXTURE_2D, self.id);
                gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGBA as GLint, self.width as GLint, self.height as GLint, 0, gl::BGRA, gl::UNSIGNED_BYTE, screen.as_ptr() as *const c_void);
                let error: GLenum = gl::GetError();
                if error != gl::NO_ERROR
                {
                    println!("ERROR ON COPY OF SCREEN TEXTURE")
                }
            }
    }
}

#[derive(Copy, Clone)]
pub struct Shader
{
    vertex: GLuint,
    pixel: GLuint,
    id: GLuint
}

impl Shader
{
    pub fn compile(vert_code: CString, pixel_code: CString) -> Self
    {
        let shader: Shader;
        unsafe
            {
                shader = Shader
                {
                    vertex: gl::CreateShader(gl::VERTEX_SHADER),
                    pixel: gl::CreateShader(gl::FRAGMENT_SHADER),
                    id: gl::CreateProgram()
                };

                gl::ShaderSource(shader.vertex, 1, &vert_code.as_ptr(), null());
                gl::CompileShader( shader.vertex );

                gl::ShaderSource(shader.pixel, 1, &pixel_code.as_ptr(), null());
                gl::CompileShader( shader.pixel );

                gl::AttachShader(shader.id, shader.vertex);
                gl::AttachShader(shader.id, shader.pixel);

                let pos = CString::new("pos").unwrap();
                gl::BindAttribLocation(shader.id, 0, pos.as_ptr());

                let tuv = CString::new("tuv").unwrap();
                gl::BindAttribLocation(shader.id, 1, tuv.as_ptr());

                gl::LinkProgram(shader.id);
            }

        return shader;
    }

    pub fn bind(&self)
    {
        unsafe
            {
                gl::UseProgram( self.id);
            }
    }

    pub fn unbind(&self)
    {
        unsafe
            {
                gl::UseProgram( 0);
            }
    }

    pub fn set_input_texture(&self, slot: u32, name: CString, texture: &GLTexture)
    {
        unsafe
            {
                gl::ActiveTexture(gl::TEXTURE0 + slot);
                gl::BindTexture(gl::TEXTURE_2D, texture.id);
                gl::Uniform1i( gl::GetUniformLocation(self.id, name.as_ptr()), slot as GLint);
            }
    }
}