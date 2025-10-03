from PIL import Image
import os
import cv2
import numpy as np
import argparse
import ast

def split_image(image_path, output_folder, chunk_size=(1000, 1000)):
    """
    Split the given image into smaller images of specified size.

    Args:
    image_path (str): Path to the input image.
    output_folder (str): Folder where the smaller images will be saved.
    chunk_size (tuple): Size of each smaller image in pixels (width, height).
    """
    # Open the image
    image = cv2.imread(image_path)


    width, height =image.shape[1], image.shape[0]
    print("Image size:",image.shape)

    # Calculate number of rows and columns required
    rows = height // chunk_size[1]
    cols = width // chunk_size[0]
    print("#Rows:",rows," #cols:",cols)
    # Create output folder if it doesn't exist
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Split the image into smaller images
    for i in range(rows):
        for j in range(cols):
            left = j * chunk_size[0]
            upper = i * chunk_size[1]
            right = left + chunk_size[0]
            lower = upper + chunk_size[1]
            # Crop the image
            cropped_image = image[upper:lower, left:right]

            cv2.imwrite(f"{output_folder}/{i * cols + j+10}.png", cropped_image)
            # cropped_image.save(f"{output_folder}/{i * cols + j+10}.png")
    return rows*cols


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="A script that takes three input arguments(i/p image path, o/p image path,splited img size).")
    parser.add_argument("arg1", type=str, help="i/p image path ex. 'raw_images_data/01_04_06.jpg'")
    parser.add_argument("arg2", type=str, help="o/p image path ex. 'split_images' ")
    parser.add_argument("arg3", type=str, help="splited image size ex. '(1000,1000)' ")
    # parser.add_argument("--arg4", action="store_true", help="An optional boolean flag (default is False)")

    args = parser.parse_args()
    image_path = args.arg1
    split_image_path=args.arg2
    #  arg3_tuple = ast.literal_eval(args.arg3)
    img_split_size=ast.literal_eval(args.arg3)

    n_splits = split_image(image_path, split_image_path, img_split_size)
    print(f'image splited in {n_splits} patches and saved in {split_image_path}.')
    


# Run commands
# python data_split_merge.py value1 value2 value3
# python yolo_sam_annotator/data_split_merge.py 'raw_images_data/01_04_06.jpg' 'split_images' '(1000,1000)'