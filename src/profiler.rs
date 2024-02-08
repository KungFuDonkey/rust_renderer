use crate::timer::{FrameTimer, Timer};

pub struct Profiler
{
    render_timer: FrameTimer,
    pub ms: f32,
    pub fps: f32,
}

impl Profiler {
    pub fn new() -> Self
    {
        Profiler{
            render_timer: FrameTimer::new(),
            ms: 0.0,
            fps: 0.0
        }
    }

}