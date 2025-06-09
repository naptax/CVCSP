import os
import json
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger

setup_logger()

def main():
    # Configurations
    data_dir = "dataset"
    train_json = "annotations_train_with_hw.json"
    val_json = "annotations_val_with_hw.json"
    train_imgs = data_dir
    val_imgs = data_dir
    output_dir = "output_detectron2"
    num_classes = 7  # 7 catégories dans le COCO
    batch_size = 2
    max_iter = 3000  # adjust for your dataset

    # --- Correction automatique des fichiers COCO pour éviter KeyError: 'info' ---
    def ensure_coco_info_and_licenses(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        changed = False
        if 'info' not in coco_data:
            coco_data['info'] = {
                "description": "Auto-added info for COCO file",
                "version": "1.0",
                "year": 2025
            }
            changed = True
        if 'licenses' not in coco_data:
            coco_data['licenses'] = []
            changed = True
        if changed:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, indent=2, ensure_ascii=False)

    ensure_coco_info_and_licenses(train_json)
    ensure_coco_info_and_licenses(val_json)

    # Calcul et affichage du nombre d'epochs
    with open(train_json, 'r') as f:
        coco = json.load(f)
    nb_images_train = len(coco['images'])
    iters_per_epoch = nb_images_train // batch_size
    nb_epochs = max_iter / iters_per_epoch if iters_per_epoch else 0
    print(f"Nombre d'images d'entraînement : {nb_images_train}")
    print(f"Batch size : {batch_size}")
    print(f"MAX_ITER : {max_iter}")
    print(f"Itérations par epoch : {iters_per_epoch}")
    print(f"Nombre d'epochs estimé : {nb_epochs:.2f}")

    # Register datasets
    register_coco_instances("plan_train", {}, train_json, train_imgs)
    register_coco_instances("plan_val", {}, val_json, val_imgs)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("plan_train",)
    cfg.DATASETS.TEST = ("plan_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # --- Évaluation automatique détaillée ---
    print("\nÉvaluation du modèle sur le jeu de validation...")
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    evaluator = COCOEvaluator("plan_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "plan_val")
    eval_results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print("\n===== Résultats de l'évaluation COCO =====")
    print(eval_results)
    # Sauvegarde dans un fichier pour consultation ultérieure
    with open(os.path.join(cfg.OUTPUT_DIR, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
