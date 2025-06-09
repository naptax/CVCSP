import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
import re
from collections import defaultdict
from tqdm import tqdm

# --- Utilitaires ---
def find_svg_files(directory):
    """Trouve tous les fichiers SVG dans un dossier (récursif)."""
    svg_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.svg'):
                svg_files.append(os.path.join(root, file))
    return svg_files

def parse_svg(svg_path):
    """Extrait les polygones et classes d'un fichier SVG."""
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        # Extraire toutes les classes (sans namespace)
        classes = [c.text for c in root.findall('.//class')]
        # Extraire les polygones (avec namespace SVG)
        polygons = []
        for poly in root.findall('.//svg:polygon', ns):
            poly_class = poly.attrib.get('class', 'Unknown')
            points_str = poly.attrib.get('points', '')
            # Extraction robuste des points
            points = []
            for pair in re.findall(r"[\d.]+[ ,][\d.]+", points_str):
                nums = re.split(r"[ ,]", pair.strip())
                if len(nums) == 2:
                    try:
                        points.append((float(nums[0]), float(nums[1])))
                    except ValueError:
                        continue
            polygons.append({
                'class': poly_class,
                'points': points
            })
        return classes, polygons
    except Exception as e:
        print(f"[ERREUR] Fichier {svg_path}: {e}")
        return [], []

def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def polygon_to_coco_segmentation(points):
    # COCO attend une liste de listes de coordonnées [x1, y1, x2, y2, ...]
    return [sum(points, ())]

# --- Main conversion ---
def main():
    parser = argparse.ArgumentParser(description="Convertit des SVG en annotations COCO (Detectron2, segmentation)")
    parser.add_argument('svg_dir', type=str, help='Dossier contenant les SVG à convertir')
    parser.add_argument('--output', type=str, default='annotations.json', help='Fichier de sortie COCO')
    args = parser.parse_args()

    svg_files = find_svg_files(args.svg_dir)
    if not svg_files:
        print(f"Aucun fichier SVG trouvé dans {args.svg_dir}")
        sys.exit(1)
    print(f"{len(svg_files)} fichiers SVG trouvés.")

    # Extraction des classes globales et polygones
    all_classes = set()
    all_polygons = []
    image_id = 1
    annotation_id = 1
    images = []
    annotations = []
    class_count = defaultdict(int)
    class_name_to_id = {}

    for svg_path in tqdm(svg_files, desc="Traitement SVG"):
        filename = os.path.basename(svg_path)
        classes, polygons = parse_svg(svg_path)
        all_classes.update(classes)
        # Image info (taille non extraite ici, peut être ajoutée si connue)
        images.append({
            'id': image_id,
            'file_name': filename
        })
        for poly in polygons:
            if len(poly['points']) < 3:
                print(f"[WARN] Polygone ignoré (moins de 3 points) dans {filename}")
                continue
            class_name = poly['class']
            class_count[class_name] += 1
            all_classes.add(class_name)
            # On mappe la classe à un id plus tard
            annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': None,  # Rempli après
                'segmentation': polygon_to_coco_segmentation(poly['points']),
                'bbox': polygon_to_bbox(poly['points']),
                'iscrowd': 0
            })
            annotation_id += 1
        image_id += 1

    # Générer la liste exhaustive des classes
    all_classes = sorted(list(all_classes))
    for idx, cname in enumerate(all_classes):
        class_name_to_id[cname] = idx + 1
    print(f"{len(all_classes)} classes trouvées : {all_classes}")
    print("Nombre d'instances par classe :")
    for cname, count in class_count.items():
        print(f"  {cname}: {count}")

    # Remplir les category_id
    for ann in annotations:
        # Si la classe n'existe pas, on met à 0 (fond)
        # (Ici, on suppose que la classe est dans all_classes)
        # Pour robustesse, on peut lever une erreur sinon
        seg = ann['segmentation'][0]
        # On retrouve la classe à partir du polygone (pas optimal mais robuste)
        for cname in class_name_to_id:
            if class_count[cname] > 0:
                ann['category_id'] = class_name_to_id[cname]
                break
        if ann['category_id'] is None:
            ann['category_id'] = 0

    # Générer la liste des catégories COCO
    categories = [
        {'id': class_name_to_id[cname], 'name': cname}
        for cname in all_classes
    ]

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(coco, f, indent=2)
    print(f"Annotations COCO sauvegardées dans {args.output}")

if __name__ == '__main__':
    main()
