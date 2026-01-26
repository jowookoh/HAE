import numpy as np
from PIL import Image, ImageDraw
import os
import argparse
from pathlib import Path
import random
from scipy.interpolate import splprep, splev


def random_line(draw, size):
    x1 = random.randint(0, size - 1)
    y1 = random.randint(0, size - 1)
    x2 = random.randint(0, size - 1)
    y2 = random.randint(0, size - 1)
    width = random.randint(1, max(1, size // 16))
    draw.line([(x1, y1), (x2, y2)], fill=255, width=width)


def random_circle(draw, size):
    radius = random.randint(size // 16, size // 3)
    cx = random.randint(radius, size - radius - 1)
    cy = random.randint(radius, size - radius - 1)
    width = random.randint(1, max(1, size // 16))
    filled = random.random() < 0.3
    if filled:
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
    else:
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=255, width=width)


def random_rectangle(draw, size):
    x1 = random.randint(0, size - 2)
    y1 = random.randint(0, size - 2)
    x2 = random.randint(x1 + 1, size - 1)
    y2 = random.randint(y1 + 1, size - 1)
    width = random.randint(1, max(1, size // 16))
    filled = random.random() < 0.3
    if filled:
        draw.rectangle([x1, y1, x2, y2], fill=255)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=255, width=width)


def random_triangle(draw, size):
    points = [(random.randint(0, size - 1), random.randint(0, size - 1)) for _ in range(3)]
    width = random.randint(1, max(1, size // 16))
    filled = random.random() < 0.3
    if filled:
        draw.polygon(points, fill=255)
    else:
        draw.polygon(points, outline=255)


def random_arc(draw, size):
    radius = random.randint(size // 8, size // 2)
    cx = random.randint(0, size - 1)
    cy = random.randint(0, size - 1)
    start_angle = random.randint(0, 360)
    end_angle = start_angle + random.randint(30, 270)
    width = random.randint(1, max(1, size // 16))
    draw.arc([cx - radius, cy - radius, cx + radius, cy + radius], start_angle, end_angle, fill=255, width=width)


def random_spline(draw, size, n_points=None):
    if n_points is None:
        n_points = random.randint(3, 7)
    
    points_x = [random.randint(0, size - 1) for _ in range(n_points)]
    points_y = [random.randint(0, size - 1) for _ in range(n_points)]
    
    try:
        tck, u = splprep([points_x, points_y], s=0, k=min(3, n_points - 1))
        u_new = np.linspace(0, 1, 100)
        x_new, y_new = splev(u_new, tck)
        
        coords = list(zip(x_new.astype(int), y_new.astype(int)))
        coords = [(max(0, min(size - 1, x)), max(0, min(size - 1, y))) for x, y in coords]
        
        width = random.randint(1, max(1, size // 16))
        for i in range(len(coords) - 1):
            draw.line([coords[i], coords[i + 1]], fill=255, width=width)
    except Exception:
        random_line(draw, size)


def bezier_point(t, points):
    n = len(points) - 1
    x, y = 0.0, 0.0
    for i, (px, py) in enumerate(points):
        binom = 1
        for j in range(i):
            binom = binom * (n - j) // (j + 1)
        factor = binom * ((1 - t) ** (n - i)) * (t ** i)
        x += factor * px
        y += factor * py
    return int(x), int(y)


def random_bezier(draw, size, n_points=None):
    if n_points is None:
        n_points = random.randint(3, 5)
    
    control_points = [(random.randint(0, size - 1), random.randint(0, size - 1)) for _ in range(n_points)]
    curve_points = [bezier_point(t / 50.0, control_points) for t in range(51)]
    curve_points = [(max(0, min(size - 1, x)), max(0, min(size - 1, y))) for x, y in curve_points]
    
    width = random.randint(1, max(1, size // 16))
    for i in range(len(curve_points) - 1):
        draw.line([curve_points[i], curve_points[i + 1]], fill=255, width=width)


def random_ellipse(draw, size):
    rx = random.randint(size // 16, size // 3)
    ry = random.randint(size // 16, size // 3)
    cx = random.randint(rx, size - rx - 1)
    cy = random.randint(ry, size - ry - 1)
    width = random.randint(1, max(1, size // 16))
    filled = random.random() < 0.3
    if filled:
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    else:
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=255, width=width)


def random_polyline(draw, size, n_points=None):
    if n_points is None:
        n_points = random.randint(3, 8)
    
    points = [(random.randint(0, size - 1), random.randint(0, size - 1)) for _ in range(n_points)]
    width = random.randint(1, max(1, size // 16))
    
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=255, width=width)


def random_spiral(draw, size):
    cx, cy = random.randint(size // 4, 3 * size // 4), random.randint(size // 4, 3 * size // 4)
    max_radius = random.randint(size // 6, size // 3)
    turns = random.uniform(1.5, 4)
    points = []
    
    for i in range(100):
        t = i / 100.0
        angle = t * turns * 2 * np.pi
        radius = t * max_radius
        x = int(cx + radius * np.cos(angle))
        y = int(cy + radius * np.sin(angle))
        points.append((max(0, min(size - 1, x)), max(0, min(size - 1, y))))
    
    width = random.randint(1, max(1, size // 16))
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=255, width=width)


def random_cross(draw, size):
    cx = random.randint(size // 4, 3 * size // 4)
    cy = random.randint(size // 4, 3 * size // 4)
    arm_len = random.randint(size // 8, size // 3)
    width = random.randint(1, max(1, size // 16))
    
    draw.line([(cx - arm_len, cy), (cx + arm_len, cy)], fill=255, width=width)
    draw.line([(cx, cy - arm_len), (cx, cy + arm_len)], fill=255, width=width)


def random_star(draw, size):
    cx = random.randint(size // 4, 3 * size // 4)
    cy = random.randint(size // 4, 3 * size // 4)
    outer_r = random.randint(size // 8, size // 3)
    inner_r = outer_r // 2
    n_points = random.randint(4, 8)
    
    points = []
    for i in range(n_points * 2):
        angle = i * np.pi / n_points - np.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))
        points.append((x, y))
    points.append(points[0])
    
    width = random.randint(1, max(1, size // 16))
    filled = random.random() < 0.3
    if filled:
        draw.polygon(points, fill=255)
    else:
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=255, width=width)


SHAPE_FUNCTIONS = [
    random_line,
    random_circle,
    random_rectangle,
    random_triangle,
    random_arc,
    random_spline,
    random_bezier,
    random_ellipse,
    random_polyline,
    random_spiral,
    random_cross,
    random_star,
]


def generate_doodle(size, min_shapes=1, max_shapes=5):
    img = Image.new('L', (size, size), color=0)
    draw = ImageDraw.Draw(img)
    
    n_shapes = random.randint(min_shapes, max_shapes)
    
    for _ in range(n_shapes):
        shape_fn = random.choice(SHAPE_FUNCTIONS)
        shape_fn(draw, size)
    
    return img


def generate_dataset(output_dir, size, num_images, min_shapes=1, max_shapes=5, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_images} doodle images at {size}x{size}...")
    
    for i in range(num_images):
        img = generate_doodle(size, min_shapes, max_shapes)
        img.save(output_path / f"{i:06d}.png")
        
        if (i + 1) % 100 == 0 or (i + 1) == num_images:
            print(f"  Generated {i + 1}/{num_images} images")
    
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic doodle dataset")
    parser.add_argument("--output", "-o", type=str, default="./images", help="Output directory")
    parser.add_argument("--size", "-s", type=int, default=64, help="Image size (e.g., 32, 64, 128, 256)")
    parser.add_argument("--num", "-n", type=int, default=1000, help="Number of images to generate")
    parser.add_argument("--min-shapes", type=int, default=1, help="Minimum shapes per image")
    parser.add_argument("--max-shapes", type=int, default=5, help="Maximum shapes per image")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    generate_dataset(
        output_dir=args.output,
        size=args.size,
        num_images=args.num,
        min_shapes=args.min_shapes,
        max_shapes=args.max_shapes,
        seed=args.seed,
    )
