
#1 json file read and convert the  image polygons to patch polygons
#2 Get the bbox and store in txt file for all the patches
#3 Convert the polygon into the mask and store in png file for all the patches

import json
import cv2
import os
from shapely.geometry import Polygon, box
from shapely.validation import explain_validity
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def list_files_in_directory(directory):
    try:
        # Get a list of all files and directories in the specified directory
        files_and_dirs = os.listdir(directory)
        
        # Filter out directories, leaving only files
        files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory, f))]
        files_without_extension = [os.path.splitext(f)[0] for f in files]
        return files_without_extension
    except FileNotFoundError:
        return "The specified directory does not exist."
    except PermissionError:
        return "You do not have permission to access this directory."
def validate_polygon(polygon):
    """
    Validate and correct a polygon if it's invalid.
    """
    if not polygon.is_valid:
        # print(f"Invalid Polygon: {explain_validity(polygon)}")
        polygon = polygon.buffer(0)
        if not polygon.is_valid:
            raise ValueError(f"Polygon could not be corrected: {explain_validity(polygon)}")
    return polygon

def convert_to_yolo_format(polygons,labels, image_width, image_height):
    yolo_format = []
    
    for polygon,label in zip(polygons,labels):
        # Find bounding box coordinates
        x_values = [p[0] for p in polygon]
        y_values = [p[1] for p in polygon]
        xmin = min(x_values)
        xmax = max(x_values)
        ymin = min(y_values)
        ymax = max(y_values)
        
        # Calculate bounding box center and dimensions
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        bbox_center_x = xmin + bbox_width / 2
        bbox_center_y = ymin + bbox_height / 2
        
        # Normalize coordinates
        bbox_center_x /= image_width
        bbox_center_y /= image_height
        bbox_width /= image_width
        bbox_height /= image_height
        
        # Append to yolo_format list
        yolo_format.append(f"{label-1} {bbox_center_x:.6f} {bbox_center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
    
    return yolo_format

def main():
    chunk_size=(1000, 1000)

    file_names = list_files_in_directory(input_json_dir)
    file_names.sort()
    # patch_n = 0
    for k,file_name in tqdm(enumerate(file_names)):
    # for file in files_without_extension:
        with open(f'{input_json_dir}/{file_name}.json','r') as f:
            image_json = json.load(f)

        # width = 4000
    # height = 6000
    # patch_size = chunk_size
    
        image = cv2.imread(f'{input_img_dir}/{file_name}.jpg')
       
        width, height =image.shape[1], image.shape[0]
        print(f"Image:{file_name}  Image size:",image.shape)

        # Calculate number of rows and columns required
        rows = height // chunk_size[1]
        cols = width // chunk_size[0]
        print("#Rows:",rows," #cols:",cols)
        # Create output folder if it doesn't exist
   
        if not os.path.exists(output_patches_dir):
            os.makedirs(output_patches_dir)

        # Split the image into smaller images
        for i in range(rows):
            for j in range(cols):
                left = j * chunk_size[0]
                upper = i * chunk_size[1]
                right = left + chunk_size[0]
                lower = upper + chunk_size[1]
                
                cropped_image = image[upper:lower, left:right]

                cv2.imwrite(f"{output_patches_dir}/{rows*cols*k + i * cols + j+10}.png", cropped_image)
                
                patch_polygons = []
                labels = []
                patch = box(left, upper, left + chunk_size[0], upper + chunk_size[1])
                for annotation in image_json['annotations']:
                    flat_list =annotation['segmentation'][0]
                    label=annotation['category_id']
                    p_coordinates=[[flat_list[i], flat_list[i+1]] for i in range(0, len(flat_list), 2)]

                    poly = Polygon(p_coordinates)
                    poly = validate_polygon(poly) 
                    if poly.intersects(patch):
                        intersection = poly.intersection(patch)
                        intersection = validate_polygon(intersection) 
                        if isinstance(intersection, Polygon):
                            patch_polygons.append(list(intersection.exterior.coords))
                            labels.append(label)
                            # save_annotation(annotation, intersection, x, y, output_folder)
                        elif intersection.geom_type == 'MultiPolygon':
                            for part in intersection.geoms:
                                part = validate_polygon(part) 
                                patch_polygons.append(list(part.exterior.coords))
                                labels.append(label)

                # print('Done',patch_polygons)
                # print('Done',len(patch_polygons))
                

                # if not os.path.exists(output_sam_train_dir):
                    # os.makedirs(output_sam_train_dir)

                # sam_mask = np.zeros((chunk_size[1], chunk_size[0]), dtype=np.uint8)
                # for polygon in patch_polygons:
                #     polygon = [[coord[0]- j*chunk_size[0],coord[1]- i*chunk_size[1]] for coord in polygon]
                #     # polygon = [[coord[0]+ i*chunk_size[1],coord[1]+ j*chunk_size[0]] for coord in polygon]
                #     polygon = np.array(polygon, dtype=np.int32)
                #     cv2.fillPoly(sam_mask, [polygon], color=(255))
                patch_polygons_n = []
                for n, polygon in enumerate(patch_polygons):
                    # sam_mask = np.zeros((chunk_size[1], chunk_size[0]), dtype=np.uint8)
                    polygon = [[coord[0]- j*chunk_size[0],coord[1]- i*chunk_size[1]] for coord in polygon]
                    patch_polygons_n.append(polygon)
                    
                    # polygon = np.array(polygon, dtype=np.int32)
                    # cv2.fillPoly(sam_mask, [polygon], color=(255))
                    # cv2.imwrite(f'{output_sam_train_dir}/{rows*cols*k + i * cols + j+10}_{n}.png', sam_mask)

                yolo_data = convert_to_yolo_format(patch_polygons_n,labels, chunk_size[1], chunk_size[0])

                # Save to a text file
                # output_file = "yolo_format.txt"
                if not os.path.exists(output_yolo_train_dir):
                    os.makedirs(output_yolo_train_dir)

                with open(f'{output_yolo_train_dir}/{rows*cols*k + i * cols + j+10}.txt', 'w') as f:
                    for bbox in yolo_data:
                        f.write(bbox + "\n")

                
                # output_file = "output_sam_train_dir
                # plt.imshow(sam_mask)
                # plt.show()
                # cv2.imwrite(f'{output_sam_train_dir}/{rows*cols*k + i * cols + j+10}.png', sam_mask)
                # break
            # break
        # break

    return rows*cols
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script that takes one path of images and train data")
    parser.add_argument("arg1", type=str, help="i/p input image dir path ex. 'visit_04_03192024'")
    parser.add_argument("arg2", type=str, help="o/p image path ex. 'yolo_train_visit_04'")
    
    args = parser.parse_args()
    input_img_dir = args.arg1
    p = args.arg2

    input_json_dir = 'image_wise_json_after_cvat_manual_annotation'
    # input_img_dir='visit_04_03192024'
    output_patches_dir = f'{p}/images'
    output_yolo_train_dir = f'{p}/labels'
    # output_sam_train_dir = 'yolo_sam_fine_tune_data/sam_mask'
    main()