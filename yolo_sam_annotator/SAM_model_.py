import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamPredictor, sam_model_registry
# from collections import defaultdict
import torch
from segment_anything.utils.transforms import ResizeLongestSide

from statistics import mean
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
import os


# Data preprocess
def get_bbox_from_mask(mask_path):
    im = cv2.imread(mask_path)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # print('contours:',contours)
    # print('contours:',len(contours))
    if len(contours) >= 1:
      x,y,w,h = cv2.boundingRect(contours[0])
      # print('x,y,w,h:',x,y,w,h)
    #   height, width, _ = im.shape
      return np.array([x, y, x + w, y + h])
    print('Error: b_box not found!')
    return 0

def find_files_with_prefix(folder_path, prefix):
    # List all files in the given folder
    files = os.listdir(folder_path)
    
    # Filter files that start with the specified prefix
    matching_files = [f for f in files if f.startswith(prefix)]
    
    return matching_files

def data_preprocess(folder_path, patch_name,sam_model, device):
    img_path = f'{folder_path}/{patch_name}.png'
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    
    input_image = sam_model.preprocess(transformed_image)
    original_image_size = image.shape[:2]
    input_size = tuple(transformed_image.shape[-2:])

    # n_mask = len(find_files_with_prefix(folder_path, f'{patch_name}_'))
    
    return input_image,input_size,original_image_size

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    polygon_points = contour.reshape(-1, 2).tolist()

    # polygon= [contour.flatten().tolist() for contour in contours]
    return polygon_points

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="A script that takes three input arguments(i/p image path, o/p image path,splited img size).")
    parser.add_argument("arg1", type=str, help="splited image path ex. '/home2/janakv/MS_project/split_images' ")
    # parser.add_argument("arg2", type=str, help="01_01_01")
    parser.add_argument("-n","--name", type=str, default="", help="image_name '01_01_01'")
    parser.add_argument("-m","--m_name", type=str, default="sam_vit_h_4b8939.pth", help="SAM_model_path 'sam_vit_h_4b8939.pth'")
   
    args = parser.parse_args()

    patches_folder_path = args.arg1
    # img_name = args.arg2
    img_name = args.name
    sam_checkpoint = args.m_name
 
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)


    # Run inference with bboxes prompt
    


    #   Load the data from the JSON file
    with open('yolo_out_for_image_bbox_data.json', 'r') as f:
        loaded_data = json.load(f)
    n_split = 48
    polygons_json = {}
    for i in range(10, n_split+10):
        # image = cv2.imread(f'{path}/{i}.png')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = loaded_data[f'image{i}']['bboxes']
        labels = loaded_data[f'image{i}']['labels']

        if len(boxes)>0:
            # result = model(f'{path}/{i}.png', bboxes=boxes)

            image = cv2.imread(f'{patches_folder_path}/{i}.png')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            input_boxes = torch.tensor(boxes).to(device=predictor.device)

            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            polygons = []
            for j in range(len(masks)):
                mask = masks[j].cpu().numpy().reshape(1000,1000)
                mask = (mask > 0).astype(np.uint8) * 255
                polygons.append(mask_to_polygon(mask))

            polygons_json[f'image{i}'] = {'polygons': polygons, 'labels': labels}

          
        else:
            polygons_json[f'image{i}'] = {'polygons': [], 'labels': []}
            
    with open(f'polygons.json', 'w') as f:
        json.dump(polygons_json, f)
        # json.dump(polygons_json, f, indent=1)
        print(f"SAM generated the mask for all patches and saved to polygons.json")


# !python yolo_sam_annotator/SAM_model.py 'split_images'



























# #model takes bboxes in xyxy format
# import torch
# import torchvision
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import cv2
# from segment_anything import sam_model_registry, SamPredictor
# import json
# import argparse
# import cv2

# def masks_to_polygons(masks):
#     numpy_array = masks.cpu().numpy()
#     if numpy_array.dtype != np.uint8:
#             numpy_array = (numpy_array * 255).astype(np.uint8)
#     polygons = []
#     for i in range(numpy_array.shape[0]):
#         # Find contours in the mask
#         contours, _ = cv2.findContours(numpy_array[i, 0, :, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             # Approximate contour to polygon
#             epsilon = 0.01 * cv2.arcLength(contour, True)
#             polygon = cv2.approxPolyDP(contour, epsilon, True)
#             # Convert polygon to a list of coordinates
#             polygon_coords = polygon.reshape(-1, 2).tolist()
#             polygons.append(polygon_coords)
#     return polygons

# if __name__=="__main__":
#     parser = argparse.ArgumentParser(description="A script that takes three input arguments(i/p image path, o/p image path,splited img size).")
#     parser.add_argument("arg1", type=str, help="splited image path ex. '/home2/janakv/MS_project/split_images' ")
#     # parser.add_argument("arg2", type=str, help="Model weights file path")
#     # parser.add_argument("--optional_arg", type=str, default="yolov8n.pt", help="Model weights file path")
   
#     args = parser.parse_args()

#     path = args.arg1
#     # model_weights_file_path = args.optional_arg
#     sam_checkpoint = "/home2/janakv/MS_project/sam_vit_h_4b8939.pth"
#     model_type = "vit_h"

#     device = "cuda"

#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)

#     predictor = SamPredictor(sam)

#     # Load the data from the JSON file
#     with open('yolo_out_for_image_bbox_data.json', 'r') as f:
#         loaded_data = json.load(f)
#     n_split = 48
#     # list_of_img_paths = []
#     # for i in range(n_split):
#     #     list_of_img_paths.append(f'{path}/{i}.png')
#     polygons_json = {}
#     for i in range(n_split):
#         image = cv2.imread(f'{path}/{i}.png')
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         boxes = loaded_data[f'image{i}']['bboxes']
#         labels = loaded_data[f'image{i}']['labels']

#         if len(boxes)>0:
#             input_boxes = torch.tensor(boxes, device=predictor.device)
#             transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
#             predictor.set_image(image)
#             masks, _, _ = predictor.predict_torch(
#                 point_coords=None,
#                 point_labels=None,
#                 boxes=transformed_boxes,
#                 multimask_output=False,
#             )
#             polygons = masks_to_polygons(masks)

#             # Convert to JSON for easier handling/storage
#             # polygons = json.dumps(polygons)
#             polygons_json[f'image{i}'] = {'polygons': polygons, 'labels': labels}

#             # Save polygons to a JSON file
            
#                 # f.write(polygons_json)
#             print('split number:',i)
#         else:
#             polygons_json[f'image{i}'] = {'polygons': [], 'labels': []}
            
#     with open('polygons.json', 'w') as f:
#         json.dump(polygons_json, f, indent=1)
#         print("Polygons have been converted and saved to polygons.json")
