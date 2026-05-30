"""
Trash Identifier
Analyzes trash images for sorting
Runs locally
No internet required
"""

import cv2
import numpy as np
import os
import json
import argparse
import sys
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Trash Categories
CATEGORIES = {
    "recyclable": {
        "label": "recyclable",
        "bin": "Recycling Bin",
        "color": (41, 128, 185),
        "tip": "Clean before recycling. Remove caps/lids.",
    },
    "organic": {
        "label": "organic",
        "bin": "Compost Bin",
        "color": (39, 174, 96),
        "tip": "No liquids. Remove packaging first.",
    },
    "landfill": {
        "label": "landfill",
        "bin": "Waste Bin",
        "color": (127, 140, 141),
        "tip": "Last resort. Try to reduce this waste.",
    },
    "hazardous": {
        "label": "harzardous",
        "bin": "Special Disposal",
        "color": (192, 57, 43),
        "tip": "Do NOT put in regular bins. Find a drop-off depot.",
    },
    "uncertain": {
        "label": "uncertain",
        "bin": "Check locally",
        "color": (243, 156, 18),
        "tip": "Check your local municipality's waste guide.",
    },
}


# analysis functions
def load_image(image_path: str):
    # Load image with OpenCV and PIL
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    img_cv = cv2.imread(str(path))
    if img_cv is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_pil = Image.open(str(path)).convert("RGB")
    return img_cv, img_pil


def detect_greens_browns(img_cv) -> dict:
    # detect greens and browns for organic waste
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # green range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # brown range
    lower_brown = np.array([10, 40, 40])
    upper_brown = np.array([25, 200, 180])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    combined_mask = cv2.bitwise_or(green_mask, brown_mask)
    total_pixels = img_cv.shape[0] * img_cv.shape[1]
    organic_pixels = cv2.countNonZero(combined_mask)
    organic_ratio = organic_pixels / total_pixels

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = [c for c in contours if cv2.contourArea(c) > 300]

    return {
        "organic_ratio": organic_ratio,
        "mask": combined_mask,
        "contours": significant,
        "region_count": len(significant),
    }


def detect_metallic_shiny(img_cv) -> dict:
    # detect silver for recyclable waste
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # metallic/white range
    shiny_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))

    total_pixels = img_cv.shape[0] * img_cv.shape[1]
    shiny_pixels = cv2.countNonZero(shiny_mask)
    shiny_ratio = shiny_pixels / total_pixels

    # specularity
    v_float = v.astype(np.float32)
    local_var = cv2.Laplacian(v_float, cv2.CV_32F).var()

    contours, _ = cv2.findContours(shiny_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = [c for c in contours if cv2.contourArea(c) > 500]

    return {
        "shiny_ratio": shiny_ratio,
        "specularity": float(local_var),
        "mask": shiny_mask,
        "contours": significant,
        "region_count": len(significant),
    }


def detect_clear_transparent(img_cv) -> dict:
    # detect transparency
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    clear_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))

    total_pixels = img_cv.shape[0] * img_cv.shape[1]
    clear_ratio = cv2.countNonZero(clear_mask) / total_pixels

    return {"clear_ratio": clear_ratio, "mask": clear_mask}


def detect_dark_matte(img_cv) -> dict:
    # detect black for waste
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))

    kernel = np.ones((5, 5), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

    total_pixels = img_cv.shape[0] * img_cv.shape[1]
    dark_ratio = cv2.countNonZero(dark_mask) / total_pixels

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = [c for c in contours if cv2.contourArea(c) > 400]

    return {
        "dark_ratio": dark_ratio,
        "mask": dark_mask,
        "contours": significant,
        "region_count": len(significant),
    }


def detect_texture_complexity(img_cv) -> dict:
    # surface texture
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    total_pixels = img_cv.shape[0] * img_cv.shape[1]
    edge_ratio = cv2.countNonZero(edges) / total_pixels

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = float(laplacian.var())

    return {
        "edge_ratio": edge_ratio,
        "texture_variance": texture_variance,
        "edges": edges,
    }


def detect_dominant_color_profile(img_cv) -> dict:
    # detect saturation as a tie breaker
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_hue = float(np.mean(h))
    mean_sat = float(np.mean(s))
    mean_val = float(np.mean(v))
    hue_std  = float(np.std(h))

    return {
        "mean_hue": mean_hue,
        "mean_sat": mean_sat,
        "mean_val": mean_val,
        "hue_std": hue_std,
    }


# classifier
def classify(organic, metallic, transparent, dark_matte, texture, color_profile) -> dict:
    """
    calculate scores
    select highest score
    """
    scores = {cat: 0 for cat in CATEGORIES}
    indicators = []

    if organic["organic_ratio"] > 0.30:
        scores["organic"] += 40
        indicators.append(
            f"Strong green/brown tones ({organic['organic_ratio']*100:.1f}%) — likely food/plant waste"
        )
    elif organic["organic_ratio"] > 0.15:
        scores["organic"] += 20
        indicators.append(
            f"Moderate organic colour ({organic['organic_ratio']*100:.1f}%) — possible food/garden waste"
        )

    if texture["texture_variance"] > 800 and organic["organic_ratio"] > 0.10:
        scores["organic"] += 15
        indicators.append("High surface complexity consistent with organic material")

    if metallic["shiny_ratio"] > 0.25:
        scores["recyclable"] += 35
        indicators.append(
            f"Shiny/metallic surface ({metallic['shiny_ratio']*100:.1f}%) — likely metal, glass or plastic"
        )
    elif metallic["shiny_ratio"] > 0.10:
        scores["recyclable"] += 15
        indicators.append(f"Partial sheen ({metallic['shiny_ratio']*100:.1f}%)")

    if transparent["clear_ratio"] > 0.20:
        scores["recyclable"] += 25
        indicators.append(
            f"Transparent regions ({transparent['clear_ratio']*100:.1f}%) — possible glass or PET plastic"
        )

    if metallic["specularity"] > 1500:
        scores["recyclable"] += 10
        indicators.append("High specularity — smooth reflective surface detected")

    if texture["edge_ratio"] < 0.08 and metallic["shiny_ratio"] > 0.10:
        scores["recyclable"] += 10
        indicators.append("Smooth, uniform surface — consistent with recyclable packaging")

    if dark_matte["dark_ratio"] > 0.35:
        scores["landfill"] += 35
        indicators.append(
            f"Dark matte coverage ({dark_matte['dark_ratio']*100:.1f}%) — possible black bag or dirty packaging"
        )
    elif dark_matte["dark_ratio"] > 0.15:
        scores["landfill"] += 15
        indicators.append(f"Significant dark matte area ({dark_matte['dark_ratio']*100:.1f}%)")

    if texture["texture_variance"] > 1200 and organic["organic_ratio"] < 0.10:
        scores["landfill"] += 10
        indicators.append("High texture irregularity on non-organic material — possible mixed/composite waste")

    if color_profile["mean_sat"] > 160 and organic["organic_ratio"] < 0.10:
        scores["hazardous"] += 20
        indicators.append(
            "High colour saturation with low organic signal — possible chemical container"
        )

    if dark_matte["dark_ratio"] > 0.20 and metallic["shiny_ratio"] > 0.20:
        scores["hazardous"] += 15
        indicators.append("Dark AND shiny regions — possible battery or electronic component")

    # determine winner
    best_category = max(scores, key=lambda k: scores[k])
    best_score    = scores[best_category]
    total_score   = sum(scores.values()) or 1

    # confidence
    confidence = int(min(95, (best_score / total_score) * 100 + 10))

    if best_score < 15:
        best_category = "uncertain"
        confidence = 30

    if not indicators:
        indicators = ["No strong sorting signals detected — manual inspection recommended"]

    return {
        "category": best_category,
        "confidence": confidence,
        "scores": scores,
        "indicators": indicators,
    }


# summary
def build_summary(category, organic, metallic, transparent, dark_matte, texture) -> str:
    cat_info = CATEGORIES[category]
    parts = [
        f"Best match: {cat_info['label']} → {cat_info['bin']}.",
        f"Tip: {cat_info['tip']}",
        f"Organic colour coverage: {organic['organic_ratio']*100:.1f}% ({organic['region_count']} region(s)).",
        f"Shiny/metallic coverage: {metallic['shiny_ratio']*100:.1f}%.",
        f"Transparent coverage: {transparent['clear_ratio']*100:.1f}%.",
        f"Dark matte coverage: {dark_matte['dark_ratio']*100:.1f}%.",
        f"Texture variance: {texture['texture_variance']:.1f} (edge density {texture['edge_ratio']*100:.1f}%).",
    ]
    return " ".join(parts)


# output image drawing
def draw_annotated_image(img_pil, img_cv, organic, metallic, dark_matte, category, confidence) -> Image.Image:
    "Draw image"
    annotated = img_pil.copy()
    draw = ImageDraw.Draw(annotated)

    cat_info  = CATEGORIES[category]
    box_color = cat_info["color"]

    # Organic region boxes
    for contour in organic["contours"]:
        x, y, w, h = cv2.boundingRect(contour)
        draw.rectangle([x, y, x + w, y + h], outline=(39, 174, 96), width=2)
        draw.text((x + 2, max(0, y - 14)), "Organic?", fill=(39, 174, 96))

    # Metallic region boxes
    for contour in metallic["contours"]:
        x, y, w, h = cv2.boundingRect(contour)
        draw.rectangle([x, y, x + w, y + h], outline=(41, 128, 185), width=2)
        draw.text((x + 2, max(0, y - 14)), "Shiny/Metal?", fill=(41, 128, 185))

    # Dark matte region boxes
    for contour in dark_matte["contours"]:
        x, y, w, h = cv2.boundingRect(contour)
        draw.rectangle([x, y, x + w, y + h], outline=(127, 140, 141), width=2)
        draw.text((x + 2, max(0, y - 14)), "Dark/Matte", fill=(127, 140, 141))

    # Banner
    banner_h = 42
    draw.rectangle([0, 0, annotated.width, banner_h], fill=(*box_color, 220))
    label = f"{cat_info['label']}  |  {cat_info['bin']}  |  Confidence: {confidence}%"
    draw.text((10, 12), label, fill=(255, 255, 255))

    return annotated


# output
def format_result(result: dict, image_path: str) -> str:
    total = sum(result["scores"].values()) or 1
    lines = []
    for cat, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
        pct = round(score / total * 100, 1)
        lines.append(f"{cat}: {pct}%")
    return "\n".join(lines)

# main
def analyze(image_path: str, output_path: str = None, save_json: bool = False) -> dict:
    # outputs results
    img_cv, img_pil = load_image(image_path)

    print("Analyzing")
    organic = detect_greens_browns(img_cv)
    color_profile = detect_dominant_color_profile(img_cv)
    metallic = detect_metallic_shiny(img_cv)
    transparent = detect_clear_transparent(img_cv)
    dark_matte = detect_dark_matte(img_cv)
    texture = detect_texture_complexity(img_cv)
    classification = classify(organic, metallic, transparent, dark_matte, texture, color_profile)

    summary = build_summary(
        classification["category"], organic, metallic, transparent, dark_matte, texture
    )

    # output image
    stem = Path(image_path).stem
    if output_path is None:
        output_path = f"detected_{stem}.jpg"

    annotated = draw_annotated_image(
        img_pil, img_cv,
        organic, metallic, dark_matte,
        classification["category"],
        classification["confidence"],
    )
    annotated.save(output_path)

    result = {
        "category":   classification["category"],
        "bin":        CATEGORIES[classification["category"]]["bin"],
        "confidence": classification["confidence"],
        "scores":     classification["scores"],
        "summary":    summary,
        "indicators": classification["indicators"],
        "output_image": output_path,
        "metrics": {
            "organic_ratio":     round(organic["organic_ratio"], 4),
            "organic_regions":   organic["region_count"],
            "shiny_ratio":       round(metallic["shiny_ratio"], 4),
            "specularity":       round(metallic["specularity"], 2),
            "clear_ratio":       round(transparent["clear_ratio"], 4),
            "dark_ratio":        round(dark_matte["dark_ratio"], 4),
            "dark_regions":      dark_matte["region_count"],
            "edge_ratio":        round(texture["edge_ratio"], 4),
            "texture_variance":  round(texture["texture_variance"], 2),
        },
    }

    if save_json:
        json_path = f"{stem}_trash_analysis.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        result["json_output"] = json_path

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Identify trash"
    )
    parser.add_argument(
        "image", nargs="?", default="trash.jpg",
        help="Path to the trash image (default: trash.jpg)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output annotated image path (default: detected_<name>.jpg)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save analysis as a JSON file"
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_only",
        help="Print raw JSON output only"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open the annotated output image after analysis"
    )

    args = parser.parse_args()

    try:
        result = analyze(args.image, output_path=args.output, save_json=args.save)

        if args.json_only:
            print(json.dumps(result, indent=2))
        else:
            print(format_result(result, args.image))

        if args.show:
            Image.open(result["output_image"]).show()

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()