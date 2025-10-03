# merge all polygon of all patches for one image in the one json file based on the requirement of the CVAT tool coco 1.0 format
import json
import os
import argparse
def convert_polygons_to_segmentation(polygons):
    segmentation = []
    for polygon in polygons:
        flattened = [coord for point in polygon for coord in point]
        segmentation.append(flattened)
    return segmentation

def adjust_coordinates(polygons, patch_index, patch_size, cols):
    """
    Adjust the coordinates of the polygons from the patch to the original image.
    """
    # Determine the row and column of the patch
    row = patch_index // cols
    col = patch_index % cols
    
    # Calculate the offset for the patch
    x_offset = col * patch_size[0]
    y_offset = row * patch_size[1]

    # Adjust the coordinates of the polygons
    adjusted_polygons = []
    for polygon in polygons:
        adjusted_polygon = [[x + x_offset, y + y_offset] for x, y in polygon]
        adjusted_polygons.append(adjusted_polygon)
    
    return adjusted_polygons

def merge_annotations(input_json, patch_size, image_size, image_name):
    """
    Merge the annotations from all patches to create the annotation for the original image.
    """
    coco_json = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
        "categories": [
            {"id": 1, "name": "flower", "supercategory": ""},
            {"id": 2, "name": "fruitlet", "supercategory": ""},
            {"id": 3, "name": "fruit", "supercategory": ""}
        ],
        "images": [],
        "annotations": []
    }
    coco_json["images"].append({
            "id": 1,
            "width": image_size[0],
            "height": image_size[1],
            "file_name": image_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })

    # Calculate the number of columns
    cols = image_size[0] // patch_size[0]
    annotation_id = 1
    for patch_name, annotation in input_json.items():
        patch_index = int(patch_name.replace("image", ""))-10
        adjusted_polygons = adjust_coordinates(annotation["polygons"], patch_index, patch_size, cols)
        
        for polygon, label in zip(adjusted_polygons, annotation["labels"]):
            segmentation = convert_polygons_to_segmentation([polygon])
            x_coords = [point[0] for point in polygon]
            y_coords = [point[1] for point in polygon]
            bbox = [min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)]
            area = bbox[2] * bbox[3]

            coco_json["annotations"].append({
                    "id": annotation_id,
                    "image_id": 1,
                    "category_id": int(label)+1,  # Adjust category_id as necessary
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {"occluded": False}
                })
            annotation_id += 1
    return coco_json


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="A script that takes three input arguments(i/p image path, o/p image path,splited img size).")
    parser.add_argument("arg1", type=str, help="i/p image name: ex. 01_01_01.jpg")
    # parser.add_argument("-c","--combined", type=str, default="", help="just write the 'combined' if you have used that file")
    parser.add_argument("-c", "--com", action='store_true', help="Use this flag if you have used the 'combined' file")

    args = parser.parse_args()
    image_name = args.arg1
    c=args.com
    if c==True:
        # Load patch annotations from the JSON file
        with open(f'polygons_combined.json') as f:
            input_json = json.load(f)
        print('combined file readed')
    else:
        with open(f'polygons.json') as f:
            input_json = json.load(f)

    # Define patch size and image size
    patch_size = (1000, 1000)
    image_size = (6000, 8000) #width, hight

    # Merge the annotations
    coco_json = merge_annotations(input_json, patch_size, image_size, image_name)

    os.makedirs('masks_json_files', exist_ok=True)
    path_j = os.path.join('masks_json_files', f'coco_{image_name.split('.')[0]}.json')
    # Save the merged annotations to a new JSON file
    with open(path_j, 'w') as f:
        json.dump(coco_json, f)

    print(f"Annotations for the image {image_name} have been merged and saved in coco_{image_name.split('.')[0]}.json")

    # ! python yolo_sam_annotator/SAM_out_to_cvat_image.py
