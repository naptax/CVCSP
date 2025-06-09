<div align="center">

<img src="https://img.icons8.com/color/96/000000/artificial-intelligence.png" width="80"/>
<h1>Detectron2 Mask R-CNN Training on CVSP 2D Floor Plan Dataset</h1>

<img src="https://img.icons8.com/color/96/000000/circuit.png" width="60"/>

</div>

---

## 🚀 Présentation du projet

Ce projet vise la **détection et la segmentation sémantique d’éléments architecturaux** sur des plans 2D (floor plans) à partir du dataset CVSP, un jeu de données de plans d’architecture annotés.

- **Origine des données** :
    - Plans d’architecture (images PNG) extraits automatiquement à partir de fichiers SVG vectoriels.
    - Annotations SVG décrivant chaque objet d’intérêt (portes, murs, pièces, fenêtres, etc.) converties en masques COCO (polygones).
- **Objectif** : Permettre l’entraînement d’un modèle Mask R-CNN Detectron2 pour la segmentation d’objets sur des plans 2D.
- **Chaîne de traitement** :
    1. SVG (annotations vectorielles) →
    2. PNG (images) →
    3. COCO JSON (annotations segmentation polygonale)

### 🏷️ Classes (catégories) détectées
- Door
- Parking
- Room
- Separation
- Text
- Wall
- Window

Chaque instance annotée est associée à une catégorie ci-dessus.

---

## 📂 Détail du dataset

- **Images** : plans 2D en PNG, résolution variable (A4, A3, etc.)
- **Annotations** : fichiers COCO JSON générés automatiquement, contenant pour chaque image :
    - `file_name`, `width`, `height`
    - Pour chaque annotation : polygone (`segmentation`), bbox, catégorie
- **Structure** :
    - `dataset/` : images PNG
    - `*.svg` : annotations SVG originales
    - `annotations_train_with_hw.json`, `annotations_val_with_hw.json` : splits COCO prêts pour Detectron2

---

## 📂 Structure du projet

```
CVCSP/
├── dataset/                     # Images PNG
├── *.svg                        # Annotations SVG
├── split_coco_train_val.py      # Split et mapping COCO
├── add_hw_to_coco.py            # Ajout width/height (optionnel)
├── train_detectron2_maskrcnn.py # Entraînement Mask R-CNN
├── annotations_train_with_hw.json
├── annotations_val_with_hw.json
├── README_detectron2_train.md   # Guide entraînement
└── README.md                    # Présentation générale
```

---

## 🛠️ Installation & Environnement

- Python 3.10+
- PyTorch >=2.7.0 (CUDA 12.8 recommandé)
- Detectron2 (installé depuis GitHub)
- opencv-python, tqdm, matplotlib, Pillow

```bash
pip install -r requirements.txt
```

---

## 🏁 Lancer l’entraînement

```bash
python train_detectron2_maskrcnn.py
```

Les checkpoints sont sauvegardés dans `output_detectron2/`.

---

## 📊 Visualisation & inférence

- Pour visualiser les masques ou inférer sur de nouvelles images, demande un script d’inférence !

---

## 🤝 Contribuer

Pull requests et suggestions bienvenues !

---

## 📄 Licence

MIT

---

<div align="center">
  <img src="https://img.icons8.com/color/96/000000/deep-learning.png" width="60"/>
</div>
