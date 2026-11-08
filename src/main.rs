// 4D Tesseract Console Renderer in Rust.

use std::io::{self, Write};
use std::thread;
use std::time::{Duration, Instant};

const FPS: u64 = 100;
const FRAME_TIME: Duration = Duration::from_micros(1_000_000 / FPS);
const EDGE_CHARS: &[u8] = b"@&Xx+;:,. ";
const VERTEX_COUNT: usize = 16; // 16 vertices (2^4)
const HALF_SIZE: f32 = 1.0;
const MAX_DIM: usize = 510;
const DEPTH_FAR: u8 = 9;
const CHAR_ASPECT: f32 = 0.5;

#[repr(C)]
struct Winsize {
    ws_row: u16,
    ws_col: u16,
    ws_xpixel: u16,
    ws_ypixel: u16,
}

unsafe extern "C" {
    fn ioctl(fd: i32, request: u64, arg: *mut Winsize) -> i32;
}

const TIOCGWINSZ: u64 = 0x5413;

#[derive(Clone, Copy, Debug, Default)]
struct Vec4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Point2D {
    x: f32,
    y: f32,
    depth: u8,
}

struct Screen {
    width: usize,
    height: usize,
    depth: Vec<u8>,
}

impl Screen {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            depth: vec![DEPTH_FAR; width * height],
        }
    }

    fn resize(&mut self, width: usize, height: usize) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        self.depth = vec![DEPTH_FAR; width * height];
    }

    fn clear(&mut self) {
        self.depth.fill(DEPTH_FAR);
    }

    fn plot(&mut self, x: usize, y: usize, depth: u8) {
        if x >= self.width || y >= self.height {
            return;
        }
        let index = y * self.width + x;
        if depth < self.depth[index] {
            self.depth[index] = depth;
        }
    }

    fn render_to_string(&self) -> String {
        let mut s = String::with_capacity(self.width * self.height + self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let depth = self.depth[y * self.width + x] as usize;
                s.push(EDGE_CHARS[depth] as char);
            }
            s.push('\n');
        }
        s
    }
}

// current terminal dimensions
fn terminal_size() -> (usize, usize) {
    let mut ws = Winsize {
        ws_row: 0,
        ws_col: 0,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    for fd in [1, 0] {
        if unsafe { ioctl(fd, TIOCGWINSZ, &mut ws) } == 0 {
            if ws.ws_row > 0 && ws.ws_col > 0 {
                return (ws.ws_col as usize, ws.ws_row as usize);
            }
        }
    }
    // fallback
    let cols = std::env::var("COLUMNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(80);
    let rows = std::env::var("LINES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    (cols, rows)
}

fn fit_screen(cols: usize, rows: usize, header: bool) -> (usize, usize) {
    let reserved_rows = if header { 3 } else { 1 };
    let available_rows = rows.saturating_sub(reserved_rows);
    let available_cols = cols.saturating_sub(1);

    let height = (available_rows as f32)
        .min((available_cols as f32 * CHAR_ASPECT).round())
        .min(MAX_DIM as f32) as usize;
    let width = (available_cols as f32)
        .min((height as f32 / CHAR_ASPECT).round())
        .min(MAX_DIM as f32) as usize;

    (width, height)
}

// generate 16 vertices
fn tesseract_vertices() -> [Vec4; VERTEX_COUNT] {
    let mut vertices = [Vec4::default(); VERTEX_COUNT];

    for i in 0..VERTEX_COUNT {
        vertices[i].x = if (i & (1 << 0)) != 0 {
            HALF_SIZE
        } else {
            -HALF_SIZE
        };
        vertices[i].y = if (i & (1 << 1)) != 0 {
            HALF_SIZE
        } else {
            -HALF_SIZE
        };
        vertices[i].z = if (i & (1 << 2)) != 0 {
            HALF_SIZE
        } else {
            -HALF_SIZE
        };
        vertices[i].w = if (i & (1 << 3)) != 0 {
            HALF_SIZE
        } else {
            -HALF_SIZE
        };
    }

    vertices
}

fn rotate_4d(v: Vec4, angle: f32) -> Vec4 {
    let c = angle.cos();
    let s = angle.sin();

    // XZ plane
    let x_4d = v.x * c - v.z * s;
    let z_4d = v.x * s + v.z * c;

    // YW plane
    let y_4d = v.y * c - v.w * s;
    let w_4d = v.y * s + v.w * c;

    Vec4 {
        x: x_4d,
        y: y_4d,
        z: z_4d,
        w: w_4d,
    }
}

// projection from 4D to 3D
fn project_4d_to_3d(v: Vec4) -> Vec3 {
    let distance_4d = 3.0; // distance of the 4D camera
    let factor = distance_4d - v.w;

    Vec3 {
        x: v.x / factor,
        y: v.y / factor,
        z: v.z / factor,
    }
}

// projection from 3D to 2D
fn project_3d_to_2d(v: Vec3, width: usize, height: usize) -> Point2D {
    let distance_3d = 2.5; // distance of the 3D camera
    let scale = 2.2;

    let z_factor = distance_3d - v.z;
    let x_2d = (v.x / z_factor) * scale;
    let y_2d = (v.y / z_factor) * scale;

    let half_w = width as f32 / 2.0;
    let half_h = height as f32 / 2.0;
    let x_screen = half_w + x_2d * half_w;
    let y_screen = half_h + y_2d * half_h;

    let norm_z = ((z_factor - 1.7) / 1.6).clamp(0.0, 1.0);
    let max_visible_depth = (DEPTH_FAR - 1) as f32;
    let depth = (norm_z * max_visible_depth).round() as u8;

    Point2D {
        x: x_screen,
        y: y_screen,
        depth,
    }
}

// Bresenham's line algorithm
fn draw_line(screen: &mut Screen, a: Point2D, b: Point2D) {
    let x0 = a.x.round() as i32;
    let y0 = a.y.round() as i32;
    let x1 = b.x.round() as i32;
    let y1 = b.y.round() as i32;

    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;

    let mut x = x0;
    let mut y = y0;
    let line_depth = (a.depth + b.depth) / 2;

    loop {
        if x >= 0 && y >= 0 {
            screen.plot(x as usize, y as usize, line_depth);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

// transform vertices and draw 32 tesseract edges
fn render_frame(screen: &mut Screen, angle: f32, width: usize, height: usize) {
    screen.clear();

    let vertices = tesseract_vertices();
    let mut projected_points = [Point2D::default(); VERTEX_COUNT];

    // rotate and project
    for i in 0..VERTEX_COUNT {
        let v_rotated = rotate_4d(vertices[i], angle);
        let v_3d = project_4d_to_3d(v_rotated);
        projected_points[i] = project_3d_to_2d(v_3d, width, height);
    }

    // connect 1bit different vertices
    for i in 0..VERTEX_COUNT {
        for bit in 0..4 {
            let neighbor_idx = i ^ (1 << bit);
            if neighbor_idx > i {
                draw_line(screen, projected_points[i], projected_points[neighbor_idx]);
            }
        }
    }
}

fn main() {
    let mut screen = Screen::new(0, 0);
    let mut stdout = io::stdout();

    // clear
    print!("\x1b[2J\x1b[H");
    let _ = stdout.flush();

    let mut previous_frame = Instant::now();
    let mut angle = 0.0_f32;

    // main render
    loop {
        let frame_start = Instant::now();

        // update angle
        angle += 0.78 * frame_start.duration_since(previous_frame).as_secs_f32();
        previous_frame = frame_start;

        let (cols, rows) = terminal_size();
        let show_header = cols >= 45 && rows >= 6;
        let (width, height) = fit_screen(cols, rows, show_header);

        screen.resize(width, height);
        render_frame(&mut screen, angle, width, height);

        print!("\x1b[H");
        if show_header {
            print!("4D Tesseract in console. (Ctrl + C to quit)\n");
            print!("Written in Rust by nazar-ponomarenko, 2026.\n");
        }
        print!("{}", screen.render_to_string());
        let _ = stdout.flush();

        let elapsed = frame_start.elapsed();
        if elapsed < FRAME_TIME {
            thread::sleep(FRAME_TIME - elapsed);
        }
    }
}
