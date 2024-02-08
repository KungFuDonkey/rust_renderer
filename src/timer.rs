pub struct Timer
{
    start: std::time::Instant
}

impl Timer
{
    pub fn new() -> Self
    {
        Timer { start: std::time::Instant::now() }
    }

    pub fn elapsed(&self) -> u128
    {
        return self.start.elapsed().as_millis();
    }

    pub fn elapsed_seconds(&self) -> u64
    {
        return self.start.elapsed().as_secs();
    }

    pub fn reset(&mut self)
    {
        self.start = std::time::Instant::now();
    }
}

pub struct FrameTimer
{
    internal_timer: Timer,
    avg: f32,
    alpha: f32
}

impl FrameTimer
{
    pub fn new() -> Self
    {
        FrameTimer {
            internal_timer: Timer::new(),
            avg: 10.0,
            alpha: 1.0
        }
    }

    pub fn reset(&mut self)
    {
        self.internal_timer.reset();
    }

    pub fn get_frame_time(&mut self) -> (f32, f32)
    {
        self.avg = (1.0 - self.alpha) * self.avg + self.alpha * (self.internal_timer.elapsed() as f32);
        if self.alpha > 0.05
        {
            self.alpha *= 0.5;
        }
        let fps = 1000.0 / self.avg;
        return (self.avg, fps);
    }

    pub fn print_frame_time(&mut self)
    {
        let (ms, fps) = self.get_frame_time();
        println!("{}ms ({}fps)", ms, fps);
    }
}
