# python file to convert the combined annotation file from the cvat to individual  
# image wise annotations
import json
import argparse
from pathlib import Path 

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="annotaton json file")
    parser.add_argument("arg1", type=str, help="i/p annotaton json path ex. 'instances_default_visit_04.json'")
    parser.add_argument("arg2", type=str, help="i/p annotaton json path ex. 'image_wise_json_after_cvat_manual_annotation'")
    # parser.add_argument("arg2", type=int, help="o/p image path ex. 'split_ratio' ")
    
    args = parser.parse_args()
    all_annotation_part = args.arg1
    # split_ratio = args.arg2/100
    # all_annotation_part = '4_tree_visit01_annotated_outof_12/annotations/instances_default.json' # change the path
    # all_annotation_part = 'instances_default_visit_04.json'
    save_dir = Path(args.arg2)
    save_dir.mkdir(parents=True, exist_ok=True)
    print("Saving JSON files to:", save_dir)

    with open(all_annotation_part, 'r') as f:
        all_annotation_polys = json.load(f)

    for i in range(len(all_annotation_polys['images'])):
        single_image_polys = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
        "categories": [
            {"id": 1, "name": "flower", "supercategory": ""},
            {"id": 2, "name": "fruitlet", "supercategory": ""},
            # {"id": 3, "name": "fruit", "supercategory": ""}
        ],
        "images": [],
        "annotations": []
        }
        single_image_polys["images"].append({
            "id": all_annotation_polys["images"][i]["id"],
            "width": all_annotation_polys["images"][i]["width"],
            "height": all_annotation_polys["images"][i]["height"],
            "file_name": all_annotation_polys["images"][i]["file_name"],
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })
        img_annotaions = []
        for poly in all_annotation_polys["annotations"]:
            if poly["image_id"] == i+1:
                img_annotaions.append(poly)
        single_image_polys["annotations"] = img_annotaions
        f_name = all_annotation_polys["images"][i]["file_name"].split('.')[0] + ".json"
        path_j = save_dir/f_name
        # path_j = os.path.join(args.arg2, f'{all_annotation_polys["images"][i]["file_name"].split('.')[0]}.json')
        with open(path_j, 'w') as f:
            json.dump(single_image_polys, f, indent=1)

            print(f"{all_annotation_polys["images"][i]["file_name"].split('.')[0]}.json file created successfully!")
            
        # image_wise_json_after_cvat_manual_annotation


