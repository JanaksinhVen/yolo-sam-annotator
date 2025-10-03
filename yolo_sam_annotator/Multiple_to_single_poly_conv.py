# input: output of the yolo_sam_annotator/SAM_out_to_cvat_image.py file
# multiple polygon for one object converted into single polygon
# import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        poly = np.array(polygon).reshape((1, -1, 2)).astype(np.int32)
        cv2.fillPoly(mask, poly, 1)
    return mask

def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.flatten().tolist() for contour in contours]
    return polygons

def convert_coco_annotations_to_masks(coco_annotations):
    height = coco_annotations['images'][0]['height']
    width = coco_annotations['images'][0]['width']
    annotations = coco_annotations['annotations']
    
    processed_annotations = []
    combined_polygons2 = []
    combined_polygons3 = []
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        if ann['category_id'] == 1:
            polygon = ann['segmentation']
            mask = polygons_to_mask(polygon, height, width)
            combined_mask = np.maximum(combined_mask, mask)
        elif ann['category_id'] == 2:
            combined_polygons2.append(ann['segmentation'])
        elif ann['category_id'] == 3:
            combined_polygons3.append(ann['segmentation'])
        else:
            pass
    combined_polygons = mask_to_polygons(combined_mask)

    annotation_id = 1
    for i in range(len(combined_polygons)):
        polygon = combined_polygons[i]
        x_coords = list(filter(lambda x: (x % 2 == 0), polygon))
        y_coords = list(filter(lambda x: (x % 2 != 0), polygon))
        bbox = [min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)]
        area = bbox[2] * bbox[3]
        processed_annotations.append({
                    "id": annotation_id,
                    "image_id": 1,
                    "category_id": 1,  # Adjust category_id as necessary
                    "segmentation": [combined_polygons[i]],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {"occluded": False}
                })
        annotation_id += 1

    for i in range(len(combined_polygons2)):
        polygon = combined_polygons2[i]
        x_coords = list(filter(lambda x: (x % 2 == 0), polygon[0]))
        y_coords = list(filter(lambda x: (x % 2 != 0), polygon[0]))
        bbox = [min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)]
        area = bbox[2] * bbox[3]
        processed_annotations.append({
                    "id": annotation_id,
                    "image_id": 1,
                    "category_id": 2,  # Adjust category_id as necessary
                    "segmentation": combined_polygons2[i],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {"occluded": False}
                })
        annotation_id += 1
    
    for i in range(len(combined_polygons3)):
        polygon = combined_polygons3[i]
        x_coords = list(filter(lambda x: (x % 2 == 0), polygon[0]))
        y_coords = list(filter(lambda x: (x % 2 != 0), polygon[0]))
        bbox = [min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)]
        area = bbox[2] * bbox[3]
        processed_annotations.append({
                    "id": annotation_id,
                    "image_id": 1,
                    "category_id": 3,  # Adjust category_id as necessary
                    "segmentation": combined_polygons3[i],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {"occluded": False}
                })
        annotation_id += 1
    return processed_annotations

def main():

    files = [f.name for f in Path("masks_json_files").iterdir() if f.is_file()]

    files.sort()
    for fil in files:
        if fil.startswith('coco_'):
            print(f'file:{fil} to {fil.split('_',1)[1]}')
            input_json_path = f'masks_json_files/{fil}'
            output_path=Path(f'masks_json_files_single_poly')
            output_path.mkdir(parents=True, exist_ok=True)
            output_json_path = output_path/f'{fil.split('_',1)[1]}'
        
            with open(input_json_path, 'r') as f:
                coco_annotations = json.load(f)
            
            processed_annotations = convert_coco_annotations_to_masks(coco_annotations)
            coco_annotations['annotations'] = processed_annotations
            
            with open(output_json_path, 'w') as f:
                json.dump(coco_annotations, f, indent=4)

if __name__ == "__main__":
    main()
# run command
# python yolo_sam_annotator/Multiple_to_single_poly_conv.py