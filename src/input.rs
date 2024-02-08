
pub struct Input
{
    has_focus: bool,
    key_state: u128,
    prev_key_state: u128
}

impl Input
{
    pub fn new() -> Self
    {
        Input{
            has_focus: true,
            key_state: 0,
            prev_key_state: 0
        }
    }

    pub fn is_key_down(&self, key: glfw::Key) -> bool
    {
        let key = (key as u128) & 255;
        let bit = 1u128 << key;
        return self.key_state & bit > 0;
    }

    pub fn is_key_released(&self, key: glfw::Key) -> bool
    {
        let key = (key as u128) & 255;
        let bit = 1u128 << key;
        return self.key_state & bit == 0 && self.prev_key_state & bit > 0;
    }

    pub fn is_key_pressed(&self, key: glfw::Key) -> bool
    {
        let key = (key as u128) & 255;
        let bit = 1u128 << key;
        return self.key_state & bit > 0 && self.prev_key_state & bit == 0;
    }

    pub fn window_has_focus(&self) -> bool
    {
        return self.has_focus;
    }

    pub fn set_key(&mut self, key: u32, pressed: bool)
    {
        let key = (key as u128) & 255;
        let bit = 1u128 << key;
        if pressed
        {
            self.key_state |= bit;
        }
        else
        {
            let mask = u128::MAX ^ bit;
            self.key_state &= mask;
        }
    }

    pub fn set_focus(&mut self, focus: bool)
    {
        self.has_focus = focus;
    }

    pub fn tick(&mut self)
    {
        self.prev_key_state = self.key_state;
    }

}