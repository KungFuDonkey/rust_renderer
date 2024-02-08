// Copyright (c) 2021 Via Technology Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::ffi::{c_void, CString};
use std::io::{stdout, Write};
use std::ptr::{null, null_mut};
use cl3::command_queue::{create_command_queue, enqueue_nd_range_kernel, enqueue_read_buffer, enqueue_write_buffer, flush, release_command_queue};
use cl3::device::*;
use cl3::platform::*;
use cl3::types::*;
use cl3::context::*;
use cl3::ext::{CL_PROGRAM_BINARIES, CL_QUEUE_PROFILING_ENABLE};
use cl3::kernel::{create_kernel, set_kernel_arg};
use cl3::memory::*;
use cl3::program::{CL_PROGRAM_BUILD_LOG, create_program_with_source, get_program_build_info};
use crate::math::*;
use crate::opencl::OpenCLVendor::Nvidia;
use crate::timer::Timer;

/// Finds all the OpenCL platforms and devices on a system.
///
/// It displays OpenCL platform information from `clGetPlatformInfo` and
/// OpenCL device information from `clGetDeviceInfo` for all the platforms and
/// devices.
pub fn get_cl_platform_info() -> Result<(), cl_int> {
    let platforms = get_platform_ids()?;
    println!("Number of platforms: {}", platforms.len());

    for platform_id in platforms {
        println!(
            "CL_PLATFORM_VENDOR: {}",
            String::from(get_platform_info(platform_id, CL_PLATFORM_VENDOR)?)
        );
        println!(
            "CL_PLATFORM_NAME: {}",
            String::from(get_platform_info(platform_id, CL_PLATFORM_NAME)?)
        );
        println!(
            "CL_PLATFORM_VERSION: {}",
            String::from(get_platform_info(platform_id, CL_PLATFORM_VERSION)?)
        );
        println!(
            "CL_PLATFORM_PROFILE: {}",
            String::from(get_platform_info(platform_id, CL_PLATFORM_PROFILE)?)
        );
        println!(
            "CL_PLATFORM_EXTENSIONS: {}",
            String::from(get_platform_info(platform_id, CL_PLATFORM_EXTENSIONS)?)
        );

        let devices = get_device_ids(platform_id, CL_DEVICE_TYPE_ALL)?;
        println!("Number of devices: {}", devices.len());
        println!();
        for device_id in devices {
            println!(
                "\tCL_DEVICE_VENDOR: {}",
                String::from(get_device_info(device_id, CL_DEVICE_VENDOR)?)
            );
            let vendor_id: cl_uint = get_device_info(device_id, CL_DEVICE_VENDOR_ID)?.into();
            println!(
                "\tCL_DEVICE_VENDOR_ID: {:X}, {}",
                vendor_id,
                vendor_id_text(vendor_id)
            );
            println!(
                "\tCL_DEVICE_NAME: {}",
                String::from(get_device_info(device_id, CL_DEVICE_NAME)?)
            );
            println!(
                "\tCL_DEVICE_VERSION: {}",
                String::from(get_device_info(device_id, CL_DEVICE_VERSION)?)
            );
            let device_type: cl_ulong = get_device_info(device_id, CL_DEVICE_TYPE)?.into();
            println!(
                "\tCL_DEVICE_TYPE: {:X}, {}",
                device_type,
                device_type_text(device_type)
            );
            println!(
                "\tCL_DEVICE_PROFILE: {}",
                String::from(get_device_info(device_id, CL_DEVICE_PROFILE)?)
            );
            println!(
                "\tCL_DEVICE_EXTENSIONS: {}",
                String::from(get_device_info(device_id, CL_DEVICE_EXTENSIONS)?)
            );
            println!(
                "\tCL_DEVICE_OPENCL_C_VERSION: {:?}",
                String::from(get_device_info(device_id, CL_DEVICE_OPENCL_C_VERSION)?)
            );

            println!(
                "\tCL_DEVICE_BUILT_IN_KERNELS: {}",
                String::from(get_device_info(device_id, CL_DEVICE_BUILT_IN_KERNELS)?)
            );
            println!(
                "\tCL_DEVICE_SVM_CAPABILITIES: {:X}",
                cl_ulong::from(get_device_info(device_id, CL_DEVICE_SVM_CAPABILITIES)?)
            );

            println!();
        }
    }

    Ok(())
}

#[derive(PartialEq)]
pub enum OpenCLVendor
{
    Nvidia,
    AMD,
    Intel,
    Other
}

#[derive(PartialEq)]
pub enum OpenCLArch
{
    Ampere,
    Turing,
    Pascal,
    Unknown
}

pub struct OpenCLDevice
{
    device_id: cl_device_id,
    vendor: OpenCLVendor,
    architecture: OpenCLArch
}


pub struct OpenCL
{
    platform: cl_platform_id,
    device: OpenCLDevice,
    context: cl_context,
    queue: cl_command_queue
}

fn get_platform_id() -> cl_platform_id
{
    let platforms = get_platform_ids()
        .expect("Failed to find platforms");

    if platforms.len() == 0
    {
        panic!("No platforms were found");
    }

    let device_types = [CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU];
    let device_order = [["NVIDIA", "AMD", ""], ["", "", ""]];
    println!("Available OpenCL platforms:");

    for platform in &platforms
    {
        println!(
            "CL_PLATFORM_NAME: {}",
            String::from(get_platform_info(*platform, CL_PLATFORM_NAME)
                .expect("Failed to get platform name"))
        );
    }

    let mut i = 0;
    for device_type in &device_types
    {
        for vendor in device_order[i]
        {
            for platform in &platforms
            {
                let device_ids = get_device_ids(*platform, *device_type).expect("Failed to get device ids from platform");
                if device_ids.len() == 0
                {
                    continue;
                }
                let platform_name = String::from(get_platform_info(*platform, CL_PLATFORM_NAME).expect("Failed to get platform name"));

                if vendor != "" && !platform_name.contains(vendor)
                {
                    continue;
                }

                return *platform;
            }
        }
        i += 1;
    }

    panic!("Failed to find good platform ID");
}

fn get_platform_context(platform: &cl_platform_id) -> (cl_context, cl_device_id)
{
    let device_ids = get_device_ids(*platform, CL_DEVICE_TYPE_ALL)
        .expect("Failed to find any devices in platform");

    let must_haves = ["cl_khr_gl_sharing", "cl_khr_global_int32_base_atomics"];

    for device in device_ids
    {
        let extensions = String::from(get_device_info(device, CL_DEVICE_EXTENSIONS)
            .expect("Failed to get device extensions"));

        let mut has_all = true;
        for must_have in must_haves
        {
            if !extensions.contains(must_have)
            {
                has_all = false;
            }
        }

        if !has_all
        {
            continue;
        }

        let platform_prop = (*platform) as cl_context_properties;
        let devices = [device];
        let props = [
            CL_CONTEXT_PLATFORM, platform_prop, 0
        ];
        let context = create_context(&devices, props.as_ptr(), None, null_mut())
            .expect("Failed to create context");

        println!("Created cl_context!");
        return (context, device);
    }

    panic!("Failed to get platform context")
}

impl OpenCLDevice
{
    pub fn from_id(device_id: cl_device_id) -> Self
    {
        let device_name = String::from(get_device_info(device_id, CL_DEVICE_NAME)
            .expect("Failed to get device name"));

        let device_version = String::from(get_device_info(device_id, CL_DEVICE_VERSION)
            .expect("Failed to get device version"));

        println!("Device {} ({})", device_name, device_version);

        let device_name = device_name.to_lowercase();

        let mut vendor: OpenCLVendor;
        let mut arch: OpenCLArch = OpenCLArch::Unknown;;
        if device_name.contains("nvidia")
        {
            vendor = OpenCLVendor::Nvidia;
            if device_name.contains("rtx")
            {
                if  device_name.contains("3050") ||
                    device_name.contains("3060") ||
                    device_name.contains("3070") ||
                    device_name.contains("3080") ||
                    device_name.contains("3090") ||
                    device_name.contains("a2000") ||
                    device_name.contains("a3000") ||
                    device_name.contains("a4000") ||
                    device_name.contains("a5000") ||
                    device_name.contains("a6000")
                {
                    arch = OpenCLArch::Ampere;
                }
                if  device_name.contains("2060") ||
                    device_name.contains("2070") ||
                    device_name.contains("2080") ||
                    device_name.contains("titan rtx")
                {
                    arch = OpenCLArch::Turing;
                }
                if  device_name.contains("quadro") &&
                    (device_name.contains("3000") ||
                        device_name.contains("4000") ||
                        device_name.contains("5000") ||
                        device_name.contains("6000") ||
                        device_name.contains("8000"))
                {
                    arch = OpenCLArch::Turing;
                }
            }
            else if device_name.contains("gtx")
            {
                if  device_name.contains("1650") ||
                    device_name.contains("1660")
                {
                    arch = OpenCLArch::Turing;
                }
                if  device_name.contains("1010") ||
                    device_name.contains("1030") ||
                    device_name.contains("1050") ||
                    device_name.contains("1060") ||
                    device_name.contains("1070") ||
                    device_name.contains("1080")
                {
                    arch = OpenCLArch::Pascal;
                }
            }
            else if device_name.contains("quadro")
            {
                if  device_name.contains("p2000") ||
                    device_name.contains("p1000") ||
                    device_name.contains("p600")  ||
                    device_name.contains("p400")  ||
                    device_name.contains("p5000") ||
                    device_name.contains("p100")
                {
                    arch = OpenCLArch::Pascal;
                }
            }
            else
            {
                if device_name.contains("titan x")
                {
                    arch = OpenCLArch::Pascal;
                }
            }
        }
        else if device_name.contains("amd") || device_name.contains("ellesmere")
        {
            vendor = OpenCLVendor::AMD;
        }
        else if device_name.contains("intel")
        {
            vendor = OpenCLVendor::Intel;
        }
        else
        {
            vendor = OpenCLVendor::Other;
        }

        println!("hardware detected: ");
        match vendor
        {
            OpenCLVendor::Nvidia =>
                {
                    print!("NVIDIA, ");
                    match arch
                    {
                        OpenCLArch::Ampere => println!("AMPERE class"),
                        OpenCLArch::Turing => println!("TURING class"),
                        OpenCLArch::Pascal => println!("PASCAL class"),
                        OpenCLArch::Unknown => println!("PRE-PASCAL hardware (warning: slow)."),
                    }
                },
            OpenCLVendor::AMD => println!("AMD"),
            OpenCLVendor::Intel => println!("INTEL"),
            OpenCLVendor::Other => println!("Identification failed")
        }

        return OpenCLDevice
        {
            architecture: arch,
            vendor,
            device_id
        }
    }
}

impl OpenCL
{
    pub fn init() -> Self
    {
        let platform = get_platform_id();
        let (context, device_id) = get_platform_context(&platform);
        let device = OpenCLDevice::from_id(device_id);

        unsafe
            {
                let queue = create_command_queue(context, device.device_id, CL_QUEUE_PROFILING_ENABLE).expect("Failed to create command queue");
                return OpenCL
                {
                    platform,
                    context,
                    device,
                    queue
                }
            }
    }

    pub fn flush_queue(&self)
    {
        flush(self.queue)
            .expect("Failed to flush queue");
    }
}

impl Drop for OpenCL
{
    fn drop(&mut self) {
        unsafe
            {
                release_command_queue(self.queue).expect("Failed to drop queue");
                release_context(self.context).expect("Failed to drop context");
            }

    }
}

pub struct OpenCLProgram
{
    program: cl_program
}

impl OpenCLProgram
{
    pub fn from_file(cl: &OpenCL, file_path: &std::path::Path) -> Self
    {
        print!("  {}", file_path.to_str().unwrap());
        stdout().flush();
        let mut timer = Timer::new();

        let text = std::fs::read_to_string(file_path)
            .expect(format!("Failed to read file at {}", file_path.to_str().unwrap()).as_str());

        let mut vendor_lines = String::new();
        if cl.device.vendor == OpenCLVendor::Nvidia
        {
            vendor_lines += "#define ISNVIDIA\n";
        }
        if cl.device.vendor == OpenCLVendor::AMD
        {
            vendor_lines += "#define ISAMD\n";
        }
        if cl.device.vendor == OpenCLVendor::Intel
        {
            vendor_lines += "#define ISINTEL\n";
        }
        if cl.device.vendor == OpenCLVendor::Other
        {
            vendor_lines += "#define ISOTHER\n";
        }
        if cl.device.architecture == OpenCLArch::Ampere
        {
            vendor_lines += "#define ISAMPERE\n"
        }
        if cl.device.architecture == OpenCLArch::Turing
        {
            vendor_lines += "#define ISTURING\n"
        }
        if cl.device.architecture == OpenCLArch::Pascal
        {
            vendor_lines += "#define ISPASCAL\n"
        }

        let source = vendor_lines + text.as_str();
        let sources = [source.as_str()];
        let program = create_program_with_source(cl.context, &sources)
            .expect("Failed to create program");

        let options = CString::new("-cl-fast-relaxed-math -cl-mad-enable -cl-single-precision-constant")
            .expect("Failed to create c string");

        match cl3::program::build_program(program,&[cl.device.device_id], options.as_c_str(), None, null_mut())
        {
            Err(_) =>
                {
                    let build_info = String::from(get_program_build_info(program, cl.device.device_id, CL_PROGRAM_BUILD_LOG)
                        .expect("Failed to get build info"));

                    println!("{}", build_info);
                    panic!("Failed to build");
                }
            _ => {}
        }

        println!(" - finished {} seconds", (timer.elapsed() as f32) / 1000.0);

        OpenCLProgram
        {
            program
        }
    }
}

pub struct OpenCLKernel
{
    kernel: cl_kernel
}

pub trait SetArgument<T>
{
    fn set_argument(&self, idx: u32, value: T);
}

impl OpenCLKernel
{
    pub fn from_program(cl: &OpenCL, program: &OpenCLProgram, entry_point: &str) -> Self
    {
        let entry_point = CString::new(entry_point)
            .expect("Failed to convert entry point");

        let kernel = create_kernel(program.program, entry_point.as_c_str())
            .expect("Failed to create kernel");

        OpenCLKernel
        {
            kernel
        }
    }

    pub fn run(&self, cl: &OpenCL, count: usize)
    {
        unsafe
            {
                let global_work_dims: *const usize = &count;
                enqueue_nd_range_kernel(cl.queue, self.kernel, 1, null(), global_work_dims, null() /*local*/, 0, null())
                    .expect("Failed to execute kernel");
            }
    }

    pub fn run2d(&self, cl: &OpenCL, count_x: usize, count_y: usize)
    {
        unsafe
            {
                let dims = [count_x, count_y];
                let local_dims: [usize; 2] = [32, 4];
                let global_work_dims: *const usize = &dims[0];
                let local_work_dims: *const usize = &local_dims[0];
                enqueue_nd_range_kernel(cl.queue, self.kernel, 2, null(), global_work_dims, local_work_dims, 0, null())
                    .expect("Failed to execute kernel");
            }
    }
}

impl SetArgument<f32> for OpenCLKernel
{
    fn set_argument(&self, idx: u32, value: f32) {
        unsafe
            {
                let pointer: *const f32 = &value;
                set_kernel_arg(self.kernel, idx, std::mem::size_of::<f32>(), pointer as *const c_void)
                    .expect("Failed to set kernel float")
            }
    }
}

impl SetArgument<u32> for OpenCLKernel
{
    fn set_argument(&self, idx: u32, value: u32) {
        unsafe
            {
                let pointer: *const u32 = &value;
                set_kernel_arg(self.kernel, idx, std::mem::size_of::<u32>(), pointer as *const c_void)
                    .expect("Failed to set kernel uint")
            }
    }
}

impl SetArgument<i32> for OpenCLKernel
{
    fn set_argument(&self, idx: u32, value: i32) {
        unsafe
            {
                let pointer: *const i32 = &value;
                set_kernel_arg(self.kernel, idx, std::mem::size_of::<i32>(), pointer as *const c_void)
                    .expect("Failed to set kernel int")
            }
    }
}

impl SetArgument<&Float2> for OpenCLKernel
{
    fn set_argument(&self, idx: u32, value: &Float2) {
        unsafe
            {
                let pointer: *const Float2 = value;
                set_kernel_arg(self.kernel, idx, std::mem::size_of::<Float2>(), pointer as *const c_void)
                    .expect("Failed to set kernel float2")
            }
    }
}

impl SetArgument<&Float3> for OpenCLKernel
{
    fn set_argument(&self, idx: u32, value: &Float3) {
        unsafe
            {
                let pointer: *const Float3 = value;
                set_kernel_arg(self.kernel, idx, std::mem::size_of::<Float3>(), pointer as *const c_void)
                    .expect("Failed to set kernel float3")
            }
    }
}

impl SetArgument<&Float4> for OpenCLKernel
{
    fn set_argument(&self, idx: u32, value: &Float4) {
        unsafe
            {
                let pointer: *const Float4 = value;
                set_kernel_arg(self.kernel, idx, std::mem::size_of::<Float4>(), pointer as *const c_void)
                    .expect("Failed to set kernel float4")
            }
    }
}

impl<T> SetArgument<&OpenCLBuffer<T>> for OpenCLKernel
{
    fn set_argument(&self, idx: u32, value: &OpenCLBuffer<T>) {
        unsafe
            {
                let pointer: *const cl_mem = &value.buffer;
                set_kernel_arg(self.kernel, idx, std::mem::size_of::<cl_mem>(), pointer as *const c_void)
                    .expect("Failed to set kernel buffer")
            }
    }
}

impl SetArgument<&Mat4> for OpenCLKernel
{
    fn set_argument(&self, idx: u32, value: &Mat4) {
        unsafe
            {
                let pointer: *const Mat4 = value;
                set_kernel_arg(self.kernel, idx, std::mem::size_of::<Mat4>(), pointer as *const c_void)
                    .expect("Failed to set kernel Mat4")
            }
    }
}

pub struct OpenCLBuffer<T>
{
    pub host_buffer: Vec<T>,
    buffer: cl_mem,
    size: usize
}

impl<T: Clone> OpenCLBuffer<T>
{
    fn create_buffer(cl: &OpenCL, host_data: Vec<T>, flags: cl_mem_flags) -> Self
    {
        unsafe
            {
                let size = host_data.len() * std::mem::size_of::<T>();

                return OpenCLBuffer
                {
                    host_buffer: host_data,
                    buffer: create_buffer(cl.context, flags, size, null_mut())
                        .expect("Failed to create buffer"),
                    size
                };;
            }
    }

    pub fn read_write(cl: &OpenCL, host_data: Vec<T>) -> Self
    {
        return OpenCLBuffer::create_buffer(cl, host_data, CL_MEM_READ_WRITE);
    }

    pub fn read_only(cl: &OpenCL, host_data: Vec<T>) -> Self
    {
        return OpenCLBuffer::create_buffer(cl, host_data, CL_MEM_READ_ONLY);
    }

    pub fn write_only(cl: &OpenCL, host_data: Vec<T>) -> Self
    {
        return OpenCLBuffer::create_buffer(cl, host_data, CL_MEM_WRITE_ONLY);
    }

    pub fn copy_to_device(&self, cl: &OpenCL)
    {
        unsafe
            {
                enqueue_write_buffer(cl.queue, self.buffer, 1, 0, self.size, self.host_buffer.as_ptr() as *const c_void, 0, null())
                    .expect("Failed to write buffer to device");
            }
    }

    pub fn copy_from_device(&mut self, cl: &OpenCL)
    {
        unsafe
            {
                enqueue_read_buffer(cl.queue, self.buffer, 1, 0, self.size, self.host_buffer.as_mut_ptr() as *mut c_void, 0, null())
                    .expect("Failed to read buffer from device");
            }
    }

}

