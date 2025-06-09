# Entraînement Mask R-CNN avec Detectron2

## Prérequis
- PyTorch >=2.7.0 (installé avec CUDA 12.8)
- Detectron2 (installé depuis GitHub)
- opencv-python, tqdm, matplotlib

## Préparation des données
Les fichiers COCO `annotations_train_with_hw.json` et `annotations_val_with_hw.json` contiennent width/height pour chaque image.

## Lancement de l'entraînement
```bash
python train_detectron2_maskrcnn.py
```

Les checkpoints et logs seront dans `output_detectron2/`.

## Notes
- Le script ajoute automatiquement width/height à chaque image lors de la génération des fichiers COCO.
- Pour inférer sur de nouveaux plans PNG, demander un script d'inférence !
