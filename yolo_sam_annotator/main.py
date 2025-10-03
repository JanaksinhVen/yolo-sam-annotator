# this script run the python files to generate the Instance segmentation file based on CVAT coco 1.0 format
import subprocess
import sys
import os
import argparse 
def run_script(command):
    try:
        result = subprocess.run(
            command,
            check=True,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred while running command: {command}", file=sys.stderr)
        print("Return code:", e.returncode, file=sys.stderr)
        print("STDOUT:", e.stdout, file=sys.stderr)
        print("STDERR:", e.stderr, file=sys.stderr)  # <- actual Python traceback
        sys.exit(1)

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
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script that takes three input arguments(input img path, visit number).")
    parser.add_argument("arg1", type=str, help="input imgs path")
    parser.add_argument("arg2", type=str, help="visit number")
    parser.add_argument("arg3", type=str, help="yolo model path")
    # parser.add_argument("--optional_arg", type=str, default="yolov8n.pt", help="Model weights file path")
   
    args = parser.parse_args()

    visit_n = args.arg1
    # # list_of_img_paths = ast.literal_eval(args.arg1)
    n = args.arg2
    yolo_model_path = args.arg3


    # Define the commands to run
    # visit_n = 'visit_03_03082024'


    imgs = list_files_in_directory(visit_n)
    imgs.sort()
    c=0
    for img in imgs:
        c+=1
        if c>2:
            continue
        print(f'{img} started............')

        commands = [
        f"python yolo_sam_annotator/data_split_merge.py '{visit_n}/{img}.jpg' 'split_images' '(1000,1000)'",
        f"python yolo_sam_annotator/yolo_bbox_pred_for_sam_image.py 'split_images' --optional_arg {yolo_model_path}",
        "python yolo_sam_annotator/yolo_out_duplicate_remove.py",
        # f"python yolo_sam_annotator/SAM_model_fine_tuned.py 'split_images' -n '{img}' -m 'sam_vit_h_4b8939_fine_tuned.pth'",
        f"python yolo_sam_annotator/SAM_model_.py 'split_images' -n '{img}'",
        f"python yolo_sam_annotator/combine_patch_edge_polygons.py",
        f"python yolo_sam_annotator/SAM_out_to_cvat_image.py '{img}.jpg' -c",
        # "python yolo_sam_annotator/Multiple_to_single_poly_conv.py"
    ]
        
        # Run each command in sequence
        for command in commands:
            run_script(command)
        print(f'{img} ended............')

    run_script("python yolo_sam_annotator/Multiple_to_single_poly_conv.py")
    run_script(f"python yolo_sam_annotator/SAM_out_to_cvat_1visit_image.py '{n}'")
    
        
