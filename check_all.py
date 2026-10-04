import xml.etree.ElementTree as ET
from pathlib import Path
import sys

# Check all XML files in the dataset
dataset_path = Path(r'C:\Users\vishw\Downloads\hachathon at ramco\dataset')

# Recursively find all XML files
xml_files = list(dataset_path.rglob('*.xml'))
print(f'Total XML files: {len(xml_files)}')

classes = set()
for xml_file in xml_files[:200]:
    try:
        tree = ET.parse(str(xml_file))
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            classes.add(name)
    except Exception as e:
        print(f'Error: {e}')

print('All classes found:', classes)

# Also check the road_damage_dataset
road_damage_path = Path(r'C:\Users\vishw\Downloads\hachathon at ramco\road_damage_dataset')
if road_damage_path.exists():
    xml_files2 = list(road_damage_path.rglob('*.xml'))
    print(f'\nRoad damage dataset XML files: {len(xml_files2)}')
    if xml_files2:
        classes2 = set()
        for xml_file in xml_files2[:50]:
            try:
                tree = ET.parse(str(xml_file))
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    classes2.add(name)
            except Exception as e:
                print(f'Error: {e}')
        print('Road damage classes:', classes2)