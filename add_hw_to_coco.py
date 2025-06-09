import json
import os
from PIL import Image

def add_hw_to_coco(coco_path, img_dir, out_path):
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    missing = []
    for img in coco['images']:
        fn = img['file_name']
        img_path = os.path.join(img_dir, fn)
        try:
            with Image.open(img_path) as im:
                img['width'], img['height'] = im.size
        except Exception as e:
            missing.append((fn, str(e)))
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print(f"Missing: {missing}, Total: {len(coco['images'])} images")

if __name__ == "__main__":
    add_hw_to_coco('annotations_train.json', 'dataset', 'annotations_train_with_hw.json')
    add_hw_to_coco('annotations_val.json', 'dataset', 'annotations_val_with_hw.json')
