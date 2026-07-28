#!/usr/bin/env python3
"""Materialize the reviewed 60-image COCO extension at 384x384."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO


SIZE = 384
RHO = 1.2
DEFAULT_ANNOTATIONS = Path(
    "/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/coco_inpaint_15_20260717/"
    "source/instances_val2017.json"
)
DEFAULT_SELECTION = Path("/tmp/coco60_review/provisional_selection_8.json")
DEFAULT_ROOT = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/coco_inpaint_60_20260721")

REPLACEMENTS = {
    "person": ["an astronaut", "a firefighter", "a medieval knight", "a humanoid robot"],
    "bird": ["a colorful parrot", "a white owl", "a raven", "a butterfly"],
    "cat": ["a red fox", "a rabbit", "a raccoon", "a small puppy"],
    "dog": ["a red fox", "a gray wolf", "a tiger cub", "a white rabbit"],
    "horse": ["a zebra", "a camel", "a donkey", "a cow"],
    "sheep": ["a goat", "an alpaca", "a calf", "a deer"],
    "cow": ["a bison", "a zebra", "a horse", "a rhinoceros"],
    "elephant": ["a rhinoceros", "a giraffe", "a hippopotamus", "a woolly mammoth"],
    "bear": ["a lion", "a tiger", "a gray wolf", "a giant panda"],
    "bicycle": ["a red motorcycle", "a blue scooter", "a small moped", "a white horse"],
    "car": ["a pickup truck", "a city bus", "a yellow taxi", "a horse-drawn carriage"],
    "motorcycle": ["a vintage bicycle", "a blue scooter", "a red quad bike", "a small sports car"],
    "airplane": ["a helicopter", "a spaceship", "a glider", "a hot-air balloon"],
    "bus": ["a red tram", "a fire truck", "a camper van", "a steam locomotive"],
    "train": ["a red tram", "a city bus", "a steam locomotive", "a futuristic monorail"],
    "truck": ["a city bus", "a white delivery van", "an armored vehicle", "a tractor"],
    "boat": ["a swan", "a submarine", "a sea turtle", "a floating sofa"],
    "banana": ["a cucumber", "a carrot", "a baguette", "a yellow feather"],
    "apple": ["an orange", "a pear", "a peach", "a tennis ball"],
    "sandwich": ["a hamburger", "a slice of cake", "an open book", "a handbag"],
    "orange": ["an apple", "a peach", "a tennis ball", "a small pumpkin"],
    "broccoli": ["a cauliflower", "a green cactus", "a flower bouquet", "a small tree"],
    "carrot": ["a cucumber", "a paintbrush", "a red chili pepper", "a yellow pencil"],
    "hot dog": ["a baguette", "a taco", "a banana", "a rolled newspaper"],
    "pizza": ["a chocolate cake", "a fruit tart", "a sushi platter", "a round clock"],
    "donut": ["a bagel", "a round clock", "a flower wreath", "a bracelet"],
    "cake": ["a fruit basket", "a flower vase", "a toy castle", "a gift box"],
    "chair": ["a small table", "a suitcase", "a potted plant", "a floor lamp"],
    "bed": ["a couch", "a dining table", "a bathtub", "a grand piano"],
    "dining table": ["a pool table", "a bed", "a fountain", "a large suitcase"],
    "toilet": ["a blue armchair", "a washing machine", "a flower pot", "a small fountain"],
    "tv": ["a framed painting", "a window", "a bookshelf", "a large mirror"],
    "cell phone": ["a calculator", "a compact camera", "a wallet", "a bar of soap"],
    "book": ["a tablet computer", "a framed photograph", "a gift box", "a small mirror"],
    "vase": ["a table lamp", "a glass bottle", "a small statue", "a flower pot"],
    "scissors": ["a wrench", "a paintbrush", "a kitchen knife", "a pair of pliers"],
    "clock": ["a framed painting", "a round mirror", "a ceramic plate", "a wall calendar"],
    "bottle": ["a table lamp", "a flower vase", "a candle", "a pepper grinder"],
    "cup": ["a glass vase", "a small bowl", "a candle holder", "a toy bucket"],
    "umbrella": ["a palm tree", "a street lamp", "a colorful kite", "a large flower"],
    "suitcase": ["a toolbox", "a picnic basket", "a backpack", "a treasure chest"],
    "remote": ["a cell phone", "a calculator", "a hairbrush", "a chocolate bar"],
}

OUTDOOR_IDS = {
    "41", "42", "44", "52", "53", "56", "57", "58", "59", "60",
    "61", "62", "64", "65", "66", "67", "68", "69", "70", "72",
    "78", "79", "82", "86", "88", "95", "98",
}
SCENE_CONTEXT = {
    "41": "outdoor_recreational", "42": "outdoor_recreational",
    "43": "indoor_workplace_or_public", "44": "outdoor_recreational",
    "45": "indoor_home", "46": "indoor_workplace_or_public",
    "47": "indoor_workplace_or_public", "48": "indoor_workplace_or_public",
    "49": "indoor_dining", "50": "indoor_workplace_or_public",
    "51": "indoor_home", "52": "outdoor_natural_or_rural",
    "53": "outdoor_natural_or_rural", "54": "indoor_workplace_or_public",
    "55": "indoor_home", "56": "outdoor_natural_or_rural",
    "57": "outdoor_natural_or_rural", "58": "outdoor_natural_or_rural",
    "59": "outdoor_urban_or_built", "60": "outdoor_natural_or_rural",
    "61": "outdoor_urban_or_built", "62": "outdoor_urban_or_built",
    "63": "indoor_workplace_or_public", "64": "outdoor_recreational",
    "65": "outdoor_urban_or_built", "66": "outdoor_urban_or_built",
    "67": "outdoor_urban_or_built", "68": "outdoor_urban_or_built",
    "69": "outdoor_urban_or_built", "70": "outdoor_natural_or_rural",
    "71": "indoor_home", "72": "outdoor_natural_or_rural",
    "73": "indoor_dining", "74": "indoor_dining", "75": "indoor_dining",
    "76": "indoor_home", "77": "indoor_home",
    "78": "outdoor_natural_or_rural", "79": "outdoor_dining",
    "80": "indoor_dining", "81": "indoor_dining",
    "82": "outdoor_urban_or_built", "83": "indoor_dining",
    "84": "indoor_home", "85": "indoor_home",
    "86": "outdoor_recreational", "87": "indoor_home",
    "88": "outdoor_natural_or_rural", "89": "indoor_home",
    "90": "indoor_home", "91": "indoor_home", "92": "indoor_home",
    "93": "indoor_home", "94": "indoor_home",
    "95": "outdoor_natural_or_rural", "96": "indoor_dining",
    "97": "indoor_dining", "98": "outdoor_urban_or_built",
    "99": "indoor_workplace_or_public", "100": "indoor_home",
}


def article(subject: str) -> str:
    return ("an " if subject[0].lower() in "aeiou" else "a ") + subject


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def tight_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty instance mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def scale_box(box: list[int], factor: float) -> list[int]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = (x1 - x0) * factor / 2, (y1 - y0) * factor / 2
    return [
        max(0, math.floor(cx - half_w)), max(0, math.floor(cy - half_h)),
        min(SIZE, math.ceil(cx + half_w)), min(SIZE, math.ceil(cy + half_h)),
    ]


def rectangle(box: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    result = np.zeros((SIZE, SIZE), dtype=bool)
    result[y0:y1, x0:x1] = True
    return result


def save_binary(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def download(url: str, path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url.replace("https://", "http://"), headers={"User-Agent": "Mozilla/5.0"})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
    temporary.replace(path)


def background_metrics(image: np.ndarray, foreground: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    background = ~foreground
    edges = cv2.Canny(gray, 50, 150) > 0
    edge_density = float(edges[background].mean())
    histogram = np.bincount(gray[background], minlength=256).astype(float)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return edge_density, entropy


def save_overlay(path: Path, image: Image.Image, segmentation: np.ndarray, boxes: list[list[int]]) -> None:
    array = np.asarray(image, dtype=np.float32)
    red = np.zeros_like(array)
    red[..., 0] = 255
    array[segmentation] = 0.55 * array[segmentation] + 0.45 * red[segmentation]
    overlay = Image.fromarray(np.uint8(np.clip(array, 0, 255)))
    draw = ImageDraw.Draw(overlay)
    for box, color in zip(boxes, ("lime", "yellow", "cyan")):
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(path)


def make_review_pages(root: Path, items: list[dict]) -> None:
    font = ImageFont.load_default()
    tile, caption = 300, 54
    for page in range(6):
        chunk = items[page * 10:(page + 1) * 10]
        canvas = Image.new("RGB", (5 * tile, 2 * (tile + caption)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, item in enumerate(chunk):
            overlay = Image.open(root / "overlays" / f"{item['id']}_overlay.png").convert("RGB")
            overlay.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            column, row = index % 5, index // 5
            x = column * tile + (tile - overlay.width) // 2
            y = row * (tile + caption) + (tile - overlay.height) // 2
            canvas.paste(overlay, (x, y))
            text_y = row * (tile + caption) + tile + 3
            draw.text((column * tile + 4, text_y), f"{item['id']} {item['subject']}", fill="black", font=font)
            draw.text(
                (column * tile + 4, text_y + 17),
                f"{item['scene_type']} {item['size_bin']}/{item['position_bin']}",
                fill="black", font=font,
            )
        canvas.save(root / f"overview_selected_60_page_{page + 1}.jpg", quality=94)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root
    selection_payload = json.loads(args.selection.read_text(encoding="utf-8"))
    selections = selection_payload["items"]
    if [entry["id"] for entry in selections] != [str(index) for index in range(41, 101)]:
        raise ValueError("selection IDs must be 41..100")
    root.mkdir(parents=True, exist_ok=True)
    annotations_target = root / "source" / "instances_val2017.json"
    annotations_target.parent.mkdir(parents=True, exist_ok=True)
    if not annotations_target.exists():
        shutil.copy2(args.annotations, annotations_target)
    shutil.copy2(args.selection, root / "selection_reviewed_60.json")

    coco = COCO(str(annotations_target))
    category_names = {entry["id"]: entry["name"] for entry in coco.dataset["categories"]}
    license_by_id = {entry["id"]: entry for entry in coco.dataset.get("licenses", [])}
    items, metadata_rows = [], []
    for selection in selections:
        sample_id = selection["id"]
        image_info = coco.imgs[selection["image_id"]]
        annotation = coco.anns[selection["annotation_id"]]
        category = category_names[annotation["category_id"]]
        if category != selection["category"] or annotation["image_id"] != selection["image_id"]:
            raise ValueError(f"annotation mismatch for {sample_id}")
        original_name = f"{sample_id}_coco_{selection['image_id']:012d}.jpg"
        original_path = root / "original_images" / original_name
        download(f"http://images.cocodataset.org/val2017/{image_info['file_name']}", original_path)
        original = Image.open(original_path).convert("RGB")
        if original.size != (image_info["width"], image_info["height"]):
            raise ValueError(f"source size mismatch for {sample_id}")
        original_mask = coco.annToMask(annotation).astype(bool)
        original_mask_path = root / "original_masks" / f"{sample_id}_instance.png"
        save_binary(original_mask_path, original_mask)

        resized_image = original.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        image_name = f"{sample_id}_{category.replace(' ', '_')}.png"
        image_path = root / "images_384" / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        resized_image.save(image_path)
        segmentation = np.asarray(
            Image.fromarray(original_mask.astype(np.uint8) * 255, mode="L").resize(
                (SIZE, SIZE), Image.Resampling.NEAREST
            )
        ) > 127
        bbox = tight_box(segmentation)
        enlarged = scale_box(bbox, RHO)
        doubled = scale_box(enlarged, RHO)
        masks = [segmentation, rectangle(bbox), rectangle(enlarged), rectangle(doubled)]
        mask_dir = root / "masks_384" / sample_id
        for name, mask in zip(
            ("segmentation.png", "bbox.png", "enlarged_bbox_rho_1.2.png", "double_enlarged_bbox_rho_1.44.png"),
            masks,
        ):
            save_binary(mask_dir / name, mask)
        save_binary(mask_dir / "attack_two_stage" / "01_positive_enlarged_bbox_rho_1.2.png", masks[2])
        save_binary(mask_dir / "attack_two_stage" / "02_negative_enlarged_bbox_rho_1.2.png", ~masks[2])
        save_overlay(root / "overlays" / f"{sample_id}_overlay.png", resized_image, segmentation, [bbox, enlarged, doubled])

        image_array = np.asarray(resized_image)
        edge_density, entropy = background_metrics(image_array, segmentation)
        metadata = {
            "id": sample_id,
            "source": "COCO 2017 val instance annotation",
            "coco_image_id": selection["image_id"],
            "coco_annotation_id": selection["annotation_id"],
            "coco_category": category,
            "coco_url": image_info["coco_url"],
            "flickr_url": image_info.get("flickr_url"),
            "license": license_by_id.get(image_info.get("license")),
            "original_size": [image_info["width"], image_info["height"]],
            "native_resolution": [SIZE, SIZE],
            "original_image_sha256": sha256(original_path),
            "native_image_sha256": sha256(image_path),
            "original_instance_mask_sha256": sha256(original_mask_path),
            "bbox_xyxy_half_open": bbox,
            "enlarged_bbox_rho_1.2_xyxy_half_open": enlarged,
            "double_enlarged_bbox_repeated_rho_1.2_xyxy_half_open": doubled,
            "segmentation_area_fraction": float(masks[0].mean()),
            "bbox_area_fraction": float(masks[1].mean()),
            "enlarged_bbox_area_fraction": float(masks[2].mean()),
            "double_enlarged_bbox_area_fraction": float(masks[3].mean()),
            "mask_fill_ratio": float(masks[0].sum() / masks[1].sum()),
            "background_edge_density": edge_density,
            "background_intensity_entropy_bits": entropy,
            "domain": selection["domain"],
            "scene_type": "outdoor" if sample_id in OUTDOOR_IDS else "indoor",
            "scene_context": SCENE_CONTEXT[sample_id],
            "size_bin": selection["size_bin"],
            "position_bin": selection["position_bin"],
            "instance_complexity_bin": selection["complexity_bin"],
            "occlusion_proxy_bin": selection["occlusion_proxy_bin"],
            "visual_reviewed": True,
            "white_mask_semantics": "foreground region to inpaint",
        }
        write_json(mask_dir / "metadata.json", metadata)
        metadata_rows.append(metadata)
        items.append({
            "id": sample_id,
            "file": image_name,
            "subject": category,
            "attack_prompt": article(category),
            "inpaint_prompts": REPLACEMENTS[category],
            "source_group": "coco_val2017_stratified_extension_60",
            "coco_image_id": selection["image_id"],
            "coco_annotation_id": selection["annotation_id"],
            "domain": selection["domain"],
            "scene_type": metadata["scene_type"],
            "scene_context": metadata["scene_context"],
            "size_bin": selection["size_bin"],
            "position_bin": selection["position_bin"],
            "instance_complexity_bin": selection["complexity_bin"],
            "occlusion_proxy_bin": selection["occlusion_proxy_bin"],
        })

    scores = np.asarray([
        row["background_edge_density"] + 0.04 * row["background_intensity_entropy_bits"]
        for row in metadata_rows
    ])
    order = np.argsort(scores)
    visual_bins = [None] * len(items)
    for rank, index in enumerate(order):
        visual_bins[index] = "low" if rank < 20 else ("medium" if rank < 40 else "high")
    for item, metadata, visual_bin in zip(items, metadata_rows, visual_bins):
        item["visual_background_complexity_bin"] = visual_bin
        metadata["visual_background_complexity_bin"] = visual_bin
        write_json(root / "masks_384" / item["id"] / "metadata.json", metadata)

    manifest = {
        "schema_version": 5,
        "dataset_name": "coco_inpaint_60_20260721",
        "image_size": SIZE,
        "bbox_scale": RHO,
        "prompt_protocol": {
            "attack": "short grammatical source-subject noun phrase",
            "inpaint_prompts_per_image": 4,
            "inpaint": "four held-out replacement subjects distinct from the source",
        },
        "selection": {
            "source": "COCO 2017 validation instances",
            "annotation_sha256": sha256(annotations_target),
            "method": selection_payload["method"],
            "quotas": selection_payload["quotas"],
            "visual_review_exclusions": selection_payload["review_exclusions"],
        },
        "items": items,
    }
    write_json(root / "dataset_extension_60.json", manifest)
    write_json(root / "masks_384" / "manifest.json", {"count": 60, "items": metadata_rows})
    audit = {
        "status": "PASS",
        "images": 60,
        "resolution": [SIZE, SIZE],
        "domain_counts": Counter(item["domain"] for item in items),
        "scene_type_counts": Counter(item["scene_type"] for item in items),
        "scene_context_counts": Counter(item["scene_context"] for item in items),
        "size_counts": Counter(item["size_bin"] for item in items),
        "position_counts": Counter(item["position_bin"] for item in items),
        "instance_complexity_counts": Counter(item["instance_complexity_bin"] for item in items),
        "visual_background_complexity_counts": Counter(item["visual_background_complexity_bin"] for item in items),
        "occlusion_proxy_counts": Counter(item["occlusion_proxy_bin"] for item in items),
        "review_exclusion_count": len(selection_payload["review_exclusions"]),
    }
    write_json(root / "audit.json", audit)
    make_review_pages(root, items)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
