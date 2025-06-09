import os

svg_dir = 'dataset'
png_dir = 'dataset'
missing_svg = []
missing_png = []
mapping = []

with open('mapping_full.txt') as f:
    lines = f.readlines()
    mapping = [l for l in lines if l.startswith('[OK]')]

for l in mapping:
    parts = l.strip().split()
    svg = parts[1]
    png = parts[-1]
    if not os.path.isfile(os.path.join(svg_dir, svg)):
        missing_svg.append(svg)
    if not os.path.isfile(os.path.join(png_dir, png)):
        missing_png.append(png)

print(f'Missing SVG: {missing_svg}')
print(f'Missing PNG: {missing_png}')
print(f'OK: {len(missing_svg)==0 and len(missing_png)==0}')
