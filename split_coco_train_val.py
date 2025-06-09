import os
import json
import argparse
import random
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Split COCO annotations en train/val selon les PNG présents.")
    parser.add_argument('--coco', type=str, default='annotations.json', help='Fichier COCO d\'entrée')
    parser.add_argument('--img_dir', type=str, default='dataset/', help='Dossier contenant les PNG')
    parser.add_argument('--train_pct', type=float, default=0.8, help='Proportion train (ex: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Seed aléatoire pour le split')
    parser.add_argument('--out_train', type=str, default='annotations_train.json', help='Fichier COCO train')
    parser.add_argument('--out_val', type=str, default='annotations_val.json', help='Fichier COCO val')
    args = parser.parse_args()

    random.seed(args.seed)

    # Lister tous les PNG disponibles
    all_files = sorted(os.listdir(args.img_dir))
    png_files = [f for f in all_files if f.lower().endswith('.png')]
    svg_files = [f for f in all_files if f.lower().endswith('.svg')]
    print(f"{len(png_files)} images PNG et {len(svg_files)} SVG trouvés dans {args.img_dir}.")

    # Charger les annotations COCO
    with open(args.coco, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # Garder uniquement les images pour lesquelles un PNG existe (selon la règle alphabétique)
    kept_images = []
    missing_png = []
    mapping_log = []

    # Mapping : chaque SVG <prefixe>_gt_*.svg doit être associé à <prefixe>.png
    for img in coco['images']:
        svg_name = img['file_name']
        svg_base = os.path.splitext(svg_name)[0]
        # Extraire le préfixe avant _gt_
        if '_gt_' in svg_base:
            prefix = svg_base.split('_gt_')[0]
            png_file = f"{prefix}.png"
            if png_file in png_files:
                img['file_name'] = png_file
                kept_images.append(img)
                mapping_log.append(f"[OK] {svg_name} associé à {png_file}")
            else:
                missing_png.append(svg_name)
                mapping_log.append(f"[WARN] Aucun PNG trouvé pour {svg_name} (attendu : {png_file})")
        else:
            missing_png.append(svg_name)
            mapping_log.append(f"[WARN] Format inattendu pour {svg_name}")

    for log in mapping_log:
        print(log)

    print(f"{len(kept_images)} images gardées (avec PNG associé par préfixe).")
    if missing_png:
        print(f"{len(missing_png)} images ignorées (PNG manquant) : {missing_png[:10]}{'...' if len(missing_png)>10 else ''}")

    # Garder uniquement les annotations liées aux images conservées
    kept_image_ids = set(img['id'] for img in kept_images)
    kept_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in kept_image_ids]
    print(f"{len(kept_images)} images gardées (avec PNG correspondant).")
    print(f"{len(kept_annotations)} annotations gardées.")

    # Split train/val
    random.shuffle(kept_images)
    n_train = int(len(kept_images) * args.train_pct)
    train_images = kept_images[:n_train]
    val_images = kept_images[n_train:]
    train_image_ids = set(img['id'] for img in train_images)
    val_image_ids = set(img['id'] for img in val_images)
    train_annotations = [ann for ann in kept_annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in kept_annotations if ann['image_id'] in val_image_ids]

    def make_coco(images, annotations):
        return {
            'images': images,
            'annotations': annotations,
            'categories': coco['categories'],
            'info': coco.get('info', {}),
            'licenses': coco.get('licenses', [])
        }

    with open(args.out_train, 'w', encoding='utf-8') as f:
        json.dump(make_coco(train_images, train_annotations), f, ensure_ascii=False, indent=2)
        print(f"Fichier train écrit : {args.out_train} ({len(train_images)} images, {len(train_annotations)} annotations)")
    with open(args.out_val, 'w', encoding='utf-8') as f:
        json.dump(make_coco(val_images, val_annotations), f, ensure_ascii=False, indent=2)
        print(f"Fichier val écrit : {args.out_val} ({len(val_images)} images, {len(val_annotations)} annotations)")

    kept_image_ids = set(img['id'] for img in kept_images)
    # Garder uniquement les annotations dont l'image existe
    kept_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in kept_image_ids]
    print(f"{len(kept_annotations)} annotations gardées.")

    # Split train/val
    random.shuffle(kept_images)
    n_train = int(len(kept_images) * args.train_pct)
    train_images = kept_images[:n_train]
    val_images = kept_images[n_train:]
    train_ids = set(img['id'] for img in train_images)
    val_ids = set(img['id'] for img in val_images)
    train_annotations = [ann for ann in kept_annotations if ann['image_id'] in train_ids]
    val_annotations = [ann for ann in kept_annotations if ann['image_id'] in val_ids]

    # Sauvegarde
    for split, imgs, anns, out in [
        ('train', train_images, train_annotations, args.out_train),
        ('val', val_images, val_annotations, args.out_val)
    ]:
        coco_out = {
            'images': imgs,
            'annotations': anns,
            'categories': coco['categories']
        }
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(coco_out, f, indent=2)
        print(f"Fichier {split} écrit : {out} ({len(imgs)} images, {len(anns)} annotations)")

if __name__ == '__main__':
    main()
