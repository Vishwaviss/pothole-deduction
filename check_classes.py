import xml.etree.ElementTree as ET
from pathlib import Path
import sys

dataset_path = Path(r'C:\Users\vishw\Downloads\hachathon at ramco\dataset')
dirs = list(dataset_path.iterdir())
print('Number of dirs:', len(dirs))
for d in dirs:
    try:
        print('Dir:', d.name.encode('ascii', 'replace').decode())
    except:
        print('Dir: (special chars)')
    if 'pothole' in d.name.lower():
        print('Found potholes dir')
        xml_files = list(d.glob('*.xml'))
        print(f'XML files found: {len(xml_files)}')
        classes = set()
        for xml_file in xml_files[:50]:
            try:
                tree = ET.parse(str(xml_file))
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    classes.add(name)
            except Exception as e:
                print(f'Error: {e}')
        print('Classes found:', classes)
        break