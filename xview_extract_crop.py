# -*- coding: utf-8 -*-
"""
Extrait les crops des différentes classes du dataset xView
Parcourt toutes les images, récupère les crops des objets de taille 15-300
en rajoutant un peu de contexte,et enregistre par classe les objets en convertissant
en niveaux de gris
Pour un certain nombre de classes, ne récupère que les objets isolés

"""
import tifffile
import os
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

image_dir = r'xView\train\img'
annotation_dir = r'xView\train\ann'
output_dir = r'xview\crop'

crop_factor = 1.5
max_size = 300

dataset = "xview"

os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(( ".tiff"))]
for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(image_dir, image_file)
    base_name = os.path.splitext(image_file)[0]
    json_path = os.path.join(annotation_dir, image_file + ".json")

    if not os.path.exists(json_path):
        continue

    try:
        if image_file.lower().endswith((".tif", ".tiff")):
            tiff_data = tifffile.imread(image_path)
            if tiff_data.ndim == 2:
                image = Image.fromarray(tiff_data).convert("L")
            elif tiff_data.ndim == 3:
                if tiff_data.shape[0] in [3, 4] and tiff_data.shape[2] != 3:
                    tiff_data = np.transpose(tiff_data, (1, 2, 0))
                image = Image.fromarray(tiff_data[:, :, :3]).convert("RGB")
            else:
                raise ValueError("Format TIFF non pris en charge")
        else:
            image = Image.open(image_path)
    except Exception as e:
        print(f"Erreur lecture image : {image_file} - {e}")
        continue

    width, height = image.size

    with open(json_path, "r") as f:
        data = json.load(f)

    objects = data.get("objects", [])

    for i, obj in enumerate(objects):
        class_title = obj.get("classTitle", "unknown")
        points = obj.get("points", {}).get("exterior", [])

        if len(points) != 2:
            continue

        xmin, ymin = points[0]
        xmax, ymax = points[1]

        w = xmax - xmin
        h = ymax - ymin

        if max(w, h) <= 15 or w >= max_size or h >= max_size:
            continue

        side = int(crop_factor * max(w, h))
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        crop_x1 = max(center_x - side // 2, 0)
        crop_y1 = max(center_y - side // 2, 0)
        crop_x2 = min(center_x + side // 2, width)
        crop_y2 = min(center_y + side // 2, height)

        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
        crop = image.crop(crop_box)

        isolated_classes = {"building", "facility", "bus", "pickup truck", "truck", "cement mixer", "dump truck", "engineering vehicle", "excavator", "ferry", "fishing vessel", "front loader", "ground grader", "haul truck", "maritime vessel", "mobile crane", "motorboat", "oil tanker", "yacht", "tugboat", "utility truck", "storage tank", "salilboat", "reach stacker", "trailer", "truck tractor"}
        if class_title in isolated_classes:
            overlap = False
            for j, other in enumerate(objects):
                if i == j:
                    continue

                other_points = other.get("points", {}).get("exterior", [])
                if len(other_points) != 2:
                    continue

                oxmin, oymin = other_points[0]
                oxmax, oymax = other_points[1]

                if not (oxmax < crop_x1 or oxmin > crop_x2 or oymax < crop_y1 or oymin > crop_y2):
                    overlap = True
                    break

            if overlap:
                continue

        crop = crop.convert("L")

        class_dir = os.path.join(output_dir, class_title)
        os.makedirs(class_dir, exist_ok=True)

        crop_name = f"{dataset}_{i}.png"

        crop.save(os.path.join(class_dir, crop_name))
