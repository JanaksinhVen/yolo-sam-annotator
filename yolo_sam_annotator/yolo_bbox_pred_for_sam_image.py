# file to predict the bbox for image (in our case 48 patches 0f 1000x1000)
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import os
import ast
import argparse
import json


def predict_yolo(list_of_img_paths,model_weights_file_path='yolov8n.pt'):
    model = YOLO(model_weights_file_path)
    results = model(list_of_img_paths)  # return a list of Results objects
    return results

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="A script that takes three input arguments(i/p image path, o/p image path,splited img size).")
    parser.add_argument("arg1", type=str, help="splited image path ex. '/home2/janakv/MS_project/split_images' ")
    # parser.add_argument("arg2", type=str, help="Model weights file path")
    parser.add_argument("--optional_arg", type=str, default="yolov8n.pt", help="Model weights file path")
   
    args = parser.parse_args()

    path = args.arg1
    model_weights_file_path = args.optional_arg
    n_split = 48
    list_of_img_paths = []
    for i in range(10, n_split+10):
        list_of_img_paths.append(f'{path}/{i}.png')
    # model_weights_file_path = 'yolov8n.pt'
    results = predict_yolo(list_of_img_paths, model_weights_file_path)
    data = {}
    data['class_names'] = {'names': results[0].names}

    for i,result in enumerate(results):
        boxes = result.boxes.xyxy  # Boxes object for bounding box outputs
 
        labels = result.boxes.cls
        data[f'image{i+10}'] = {'bboxes': boxes.tolist(), 'labels': labels.tolist()}

    with open('yolo_out_for_image_bbox_data.json', 'w') as f:
        json.dump(data, f, indent=4)
        print('YOLO predicted BBOX for all patches and saved successfully in yolo_out_for_image_bbox_data.json.')


    # !python yolo_sam_annotator/yolo_bbox_pred_for_sam_image.py "split_images" --optional_arg 'runs/detect/train12/weights/best.pt'

    '''reason for use of the .png over .jpg: 
        In general, for most object detection tasks, where the 
        visual quality of the image is not critical and file size is a consideration, JPG is commonly used.
        However, if you need transparency or the image quality is paramount, PNG may be preferred despite 
        its larger file sizes.
        Note: In our case image quality is more importent'''