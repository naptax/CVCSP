import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import argparse

# Fichier SVG à lire
SVG_FILE = "dataset/1_gt_14.svg"

# Couleurs par défaut pour la légende si non trouvées dans le SVG
DEFAULT_COLORS = {
    "Wall": "#AFD8F8",
    "Door": "#F6BD0F",
    "Window": "#8BBA00",
    "Room": "#FF8E46",
    "Separation": "#008E8E"
}

def parse_svg_polygons(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Namespace SVG (si présent)
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Extraire toutes les classes (sans namespace)
    classes = [c.text for c in root.findall('.//class')]

    # Extraire tous les polygones (avec namespace SVG)
    polygons = []
    for poly in root.findall('.//svg:polygon', ns):
        poly_class = poly.attrib.get('class', 'Unknown')
        fill = poly.attrib.get('fill', DEFAULT_COLORS.get(poly_class, '#CCCCCC'))
        points_str = poly.attrib.get('points', '')
        # Extraire les points, robustement (supporte espace ou virgule comme séparateur)
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
            'fill': fill,
            'points': points
        })
    return classes, polygons

def plot_svg_polygons(classes, polygons):
    fig, ax = plt.subplots(figsize=(12, 10))
    legend_handles = {}
    for poly in polygons:
        pts = poly["points"]
        if len(pts) < 3:
            continue  # Un polygone doit avoir au moins 3 points
        polygon_patch = patches.Polygon(pts, closed=True, facecolor=poly["fill"], edgecolor='black', linewidth=1, alpha=0.7, label=poly["class"])
        ax.add_patch(polygon_patch)
        # Préparer la légende
        if poly["class"] not in legend_handles:
            legend_handles[poly["class"]] = polygon_patch
    ax.autoscale()
    ax.set_aspect('equal')
    plt.title("Plan SVG avec légende")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(legend_handles.values(), legend_handles.keys(), loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrait et affiche les polygones d'un fichier SVG.")
    parser.add_argument("svg_file", type=str, help="Chemin du fichier SVG à traiter")
    args = parser.parse_args()

    classes, polygons = parse_svg_polygons(args.svg_file)
    print("Classes trouvées:", classes)
    print(f"{len(polygons)} polygones extraits.")
    if len(polygons) == 0:
        print("Aucun polygone extrait. Vérifiez la structure du SVG ou le parsing des points.")
    else:
        plot_svg_polygons(classes, polygons)
