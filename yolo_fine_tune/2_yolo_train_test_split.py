# to split in train and val
import os
import shutil
import random
import argparse

def train_test_split_yolo(data_dir, split_ratio=0.8):
    # Create necessary folders for train and val splits
    img_train_dir = os.path.join(data_dir, 'images', 'train')
    img_val_dir = os.path.join(data_dir, 'images', 'val')
    label_train_dir = os.path.join(data_dir, 'labels', 'train')
    label_val_dir = os.path.join(data_dir, 'labels', 'val')
    
    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)

    # List all images and labels
    images = [f for f in os.listdir(os.path.join(data_dir, 'images')) if f.endswith('.png')]
    
    # Shuffle the images for random splitting
    random.shuffle(images)
    
    # Calculate split index
    split_index = int(len(images) * split_ratio)
    
    # Split into train and validation sets
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Move images and corresponding labels
    for img in train_images:
        label = img.replace('.png', '.txt')
        shutil.move(os.path.join(data_dir, 'images', img), os.path.join(img_train_dir, img))
        shutil.move(os.path.join(data_dir, 'labels', label), os.path.join(label_train_dir, label))
    
    for img in val_images:
        label = img.replace('.png', '.txt')
        shutil.move(os.path.join(data_dir, 'images', img), os.path.join(img_val_dir, img))
        shutil.move(os.path.join(data_dir, 'labels', label), os.path.join(label_val_dir, label))

    print(f"Dataset split complete. Train images: {len(train_images)}, Validation images: {len(val_images)}")

# Example usage:
if __name__=='__main__':

    parser = argparse.ArgumentParser(description="A script that takes one path of the yolo train data")
    parser.add_argument("arg1", type=str, help="i/p image path ex. 'yolo_train_visit_03'")
    parser.add_argument("arg2", type=int, help="o/p image path ex. 'split_ratio' ")
    
    args = parser.parse_args()
    data_path = args.arg1
    split_ratio = args.arg2/100
    train_test_split_yolo(data_path, split_ratio=split_ratio)
