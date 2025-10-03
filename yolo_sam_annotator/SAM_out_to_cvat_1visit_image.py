# this file combine the 96 annotation taken in one particular visit and create one json for cvat 
import os
import json
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script that takes one input arguments(visit_number)')
    parser.add_argument('arg1',type=str,help="pass the visit number ex.01")
    args = parser.parse_args()
    visit_n = args.arg1

    dir_json = 'masks_json_files_single_poly'

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
    annotation_id = 0
    var_annotation_id = 0
    files = os.listdir(dir_json)
    files.sort()
    img_id = 0
    for file_j in files:
        img_id += 1 

        with open(f'{dir_json}/{file_j}') as f:
            input_json = json.load(f)

        
        coco_json["images"].append({
                "id": img_id,
                "width": input_json['images'][0]['width'],
                "height": input_json['images'][0]['height'],
                "file_name": input_json['images'][0]['file_name'],
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0
            })
        
        
        for i in range(len(input_json["annotations"])):
            annotation_id += 1

            coco_json["annotations"].append({
                            "id": annotation_id,
                            "image_id": img_id,
                            "category_id": input_json['annotations'][i]['category_id'],  # Adjust category_id as necessary
                            "segmentation": input_json['annotations'][i]['segmentation'],
                            "area": input_json["annotations"][i]['area'],
                            "bbox": input_json["annotations"][i]['bbox'],
                            "iscrowd": 0,
                            "attributes": {"occluded": False}
                        })
        if input_json['annotations']!=[]:
            var_annotation_id += input_json['annotations'][i]['id']
    # print('annotation_id',annotation_id)
    # print('var_annotation_id',var_annotation_id)
    if var_annotation_id==annotation_id:
        print('Total number of annotation id\'s are varified!!!!!')
    else:
        print('Total number of annotation id\'s are incorrect!!!!!')

    with open(f'combined_mask_visit{visit_n}.json', 'w') as f:
        json.dump(coco_json,f)

    
    print(f"annotation combination of {img_id} images is done and saved in combined_mask_visit{visit_n}.json")

    # python yolo_sam_annotator/SAM_out_to_cvat_1visit_image.py '01'