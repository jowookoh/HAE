import argparse
from pathlib import Path
from quickdraw import QuickDrawData, QuickDrawDataGroup
from PIL import Image, ImageDraw
import random


POPULAR_CATEGORIES = [
    "cat", "dog", "bird", "fish", "rabbit", "bear", "lion", "tiger", "elephant", "monkey",
    "house", "tree", "flower", "sun", "moon", "star", "cloud", "rainbow", "mountain", "river",
    "car", "bus", "train", "airplane", "bicycle", "boat", "helicopter",
    "apple", "banana", "pizza", "cake", "ice cream", "cookie", "bread",
    "face", "eye", "hand", "foot",
    "chair", "table", "bed", "door", "window", "clock", "lamp", "book",
    "guitar", "piano", "drums", "trumpet",
    "heart", "smiley face", "skull", "lightning",
    "circle", "square", "triangle", "hexagon",
]


def render_drawing(drawing, size, stroke_width=None):
    img = Image.new('L', (size, size), color=0)
    draw = ImageDraw.Draw(img)
    
    if stroke_width is None:
        stroke_width = max(1, size // 32)
    
    orig_size = 256
    scale = size / orig_size
    
    for stroke in drawing.strokes:
        points = [(int(x * scale), int(y * scale)) for x, y in stroke]
        if len(points) >= 2:
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=255, width=stroke_width)
    
    return img


def download_category(category, output_dir, size, num_images, stroke_width=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading '{category}'...")
    group = QuickDrawDataGroup(category, recognized=True, max_drawings=num_images)
    
    count = 0
    for i, drawing in enumerate(group.drawings):
        if count >= num_images:
            break
        img = render_drawing(drawing, size, stroke_width)
        img.save(output_path / f"{category.replace(' ', '_')}_{i:06d}.png")
        count += 1
        
        if (count) % 100 == 0:
            print(f"  Saved {count}/{num_images}")
    
    print(f"  Saved {count} images for '{category}'")
    return count


def download_mixed(categories, output_dir, size, num_images, stroke_width=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_per_category = max(1, num_images // len(categories))
    extra = num_images % len(categories)
    
    all_images = []
    
    for i, category in enumerate(categories):
        n = images_per_category + (1 if i < extra else 0)
        print(f"Downloading '{category}' ({n} images)...")
        
        try:
            group = QuickDrawDataGroup(category, recognized=True, max_drawings=n)
            count = 0
            for drawing in group.drawings:
                if count >= n:
                    break
                all_images.append((category, drawing))
                count += 1
        except Exception as e:
            print(f"  Warning: Failed to load '{category}': {e}")
    
    random.shuffle(all_images)
    
    print(f"\nSaving {len(all_images)} mixed images...")
    for i, (category, drawing) in enumerate(all_images):
        img = render_drawing(drawing, size, stroke_width)
        img.save(output_path / f"{i:06d}.png")
        
        if (i + 1) % 500 == 0:
            print(f"  Saved {i + 1}/{len(all_images)}")
    
    print(f"Dataset saved to {output_path}")
    return len(all_images)


def list_categories():
    qd = QuickDrawData()
    return qd.drawing_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Quick Draw dataset")
    parser.add_argument("--output", "-o", type=str, default="./images", help="Output directory")
    parser.add_argument("--size", "-s", type=int, default=64, help="Image size")
    parser.add_argument("--num", "-n", type=int, default=1000, help="Number of images (total or per category)")
    parser.add_argument("--categories", "-c", nargs="+", default=None, help="Categories to download (default: popular mix)")
    parser.add_argument("--single", action="store_true", help="Download single category (uses first category)")
    parser.add_argument("--stroke-width", type=int, default=None, help="Stroke width (default: size/32)")
    parser.add_argument("--list", action="store_true", help="List all available categories")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available categories:")
        for name in sorted(list_categories()):
            print(f"  {name}")
        exit(0)
    
    if args.seed is not None:
        random.seed(args.seed)
    
    categories = args.categories if args.categories else POPULAR_CATEGORIES
    
    if args.single:
        category = categories[0]
        download_category(category, args.output, args.size, args.num, args.stroke_width)
    else:
        download_mixed(categories, args.output, args.size, args.num, args.stroke_width)
