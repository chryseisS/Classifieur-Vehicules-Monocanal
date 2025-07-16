# -*- coding: utf-8 -*-
"""
Extrait les crops des différentes classes du dataset dota2
Parcourt toutes les images, récupère les crops des objets de taille 15-300
en rajoutant un peu de contexte,et enregistre par classe les objets en convertissant
en niveaux de gris
Pour un certain nombre de classes, ne récupère que les objets isolés
"""
import os
import json
from PIL import Image
from tqdm import tqdm

image_dir = r'dota\train\img'
annotation_dir = r'dota\train\ann'
output_dir = r'dota\crop'

crop_factor = 1.5
max_size = 300
dataset = "dota"

isolated_classes = {
    "large vehicle", "small vehicle", "bus", "pickup truck", "truck", "cement mixer", "dump truck",
    "engineering vehicle", "excavator", "ferry", "fishing vessel", "front loader",
    "ground grader", "haul truck", "maritime vessel", "mobile crane", "motorboat",
    "oil tanker", "yacht", "tugboat", "utility truck", "storage tank", "salilboat",
    "reach stacker", "trailer", "ship"
}

os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

for image_file in tqdm(image_files, desc="Processing DOTA images"):
    base_name = os.path.splitext(image_file)[0]
    image_path = os.path.join(image_dir, image_file)
    json_path = os.path.join(annotation_dir, image_file + ".json")

    if not os.path.exists(json_path):
        continue

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Erreur image {image_file} : {e}")
        continue

    width, height = image.size

    with open(json_path, "r") as f:
        data = json.load(f)

    objects = data.get("objects", [])

    for i, obj in enumerate(objects):
        class_title = obj.get("classTitle", "unknown")
        exterior = obj.get("points", {}).get("exterior", [])

        if len(exterior) < 3:
            continue

        xs = [pt[0] for pt in exterior]
        ys = [pt[1] for pt in exterior]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        w = xmax - xmin
        h = ymax - ymin

        if max(w, h) <= 15 or max(w, h) * crop_factor >= max_size:
            continue

        side = int(crop_factor * max(w, h))
        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)
        crop_x1 = max(center_x - side // 2, 0)
        crop_y1 = max(center_y - side // 2, 0)
        crop_x2 = min(center_x + side // 2, width)
        crop_y2 = min(center_y + side // 2, height)

        if class_title in isolated_classes:
            overlap = False
            for j, other in enumerate(objects):
                if i == j:
                    continue
                other_ext = other.get("points", {}).get("exterior", [])
                if len(other_ext) < 3:
                    continue
                oxs = [pt[0] for pt in other_ext]
                oys = [pt[1] for pt in other_ext]
                oxmin, oxmax = min(oxs), max(oxs)
                oymin, oymax = min(oys), max(oys)

                if not (oxmax < crop_x1 or oxmin > crop_x2 or oymax < crop_y1 or oymin > crop_y2):
                    overlap = True
                    break

            if overlap:
                continue

        crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2)).convert("L")

        class_dir = os.path.join(output_dir, class_title)
        os.makedirs(class_dir, exist_ok=True)

        crop_name = f"{dataset}_{base_name}_{i}.png"
        crop.save(os.path.join(class_dir, crop_name))
