# 🥭 YOLO-SAM Annotator for High-Resolution Imagery

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20Ultralytics%20%7C%20SAM-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An automated annotation pipeline that leverages the **YOLO (You Only Look Once)** object detection model and the **SAM (Segment Anything Model)** to generate precise segmentation masks for objects in high-resolution images.

This repository was developed for annotating the [MangoSense dataset](https://drive.google.com/drive/folders/1yaXyAVl0defyBm4LeuzsaEt_hthCR9x7?usp=sharing), which involves segmenting small objects (flowers and fruitlets) from very large images of mango trees (e.g., 8000x6000 pixels).

---

## 🚀 How It Works

Working with high-resolution images directly is computationally expensive. This pipeline uses a patch-based approach to make the process efficient:

1.  **Slice**: The high-resolution source image is divided into smaller, manageable patches (e.g., 1000x1000 pixels).
2.  **Detect**: A fine-tuned YOLO model runs on each patch to detect objects and generate bounding boxes.
3.  **Segment**: The bounding boxes are fed as prompts to the Segment Anything Model (SAM), which produces highly accurate segmentation masks for each detection.
4.  **Merge**: The annotated patches are seamlessly stitched back together to reconstruct the full-resolution image with complete segmentation masks.



---

## 🔧 Environment Setup

To get started, clone the repository and set up the environment using Conda.

1.  **Create and Activate Conda Environment:**
    The provided `environment_mangosense.yml` file contains all the necessary dependencies.

    ```bash
    # Create a new conda environment from the file
    conda env create -f environment_mangosense.yml

    # Activate the newly created environment
    conda activate mangosense
    ```
    Note: pip dependency will fail sometimes when we are using conda to install the environment directly from the .yml file, so 2nd step will manually install all pip dependencies

2.  **Manually Install Pip Dependencies:**
    Install YOLO and Segment Anything.

    ```bash
    # Install YOLO (Ultralytics)
    pip install ultralytics

    # Clone the official Segment Anything repository
    git clone [https://github.com/facebookresearch/segment-anything.git](https://github.com/facebookresearch/segment-anything.git)
    cd segment-anything

    # Install Segment Anything in editable mode
    pip install -e .

    # Return to the project root
    cd ..
    ```

3.  **Install Additional Dependencies:**

    ```bash
    pip install shapely tqdm
    ```

---

## 📊 Dataset

This project uses the **MangoSense** dataset. You can download it from the following link:

* **Download URL:** [Google Drive](https://drive.google.com/drive/folders/1yaXyAVl0defyBm4LeuzsaEt_hthCR9x7?usp=sharing)

---

## ⚙️ Usage Workflow

The process is divided into two main stages: fine-tuning the YOLO model on your custom data and then using the trained model to generate annotations.

### Part 1: Fine-Tuning the YOLO Model

First, you need to train a YOLO model to accurately detect the objects of interest in the image patches.

#### Step 1: Prepare Annotations
Convert your main COCO-formatted JSON annotation file into individual JSON files for each image.

```bash
python yolo_fine_tune/1_all_annotation_json_to_imagewise_json.py 'instances_default_visit_07.json' 'image_wise_json_after_cvat_manual_annotation'
```

#### Step 2: Prepare Training Data (Images and Labels)
Slice the high-resolution images into patches and convert the corresponding annotations into the YOLO format (bounding boxes).
```bash
# Arguments: <source_images_folder> <output_patches_folder>
python yolo_fine_tune/2_yolo_train_data_preparation.py 'visit_07_04232024' 'yolo_train_visit_07'
```

#### Step 3: Split Data
Divide the generated patches into training and validation sets.
```bash
# Arguments: <patches_folder> <train_percentage>
python yolo_fine_tune/2_yolo_train_test_split.py 'yolo_train_visit_07' '80'
```

#### Step 4: Create Data Configuration File
Create a data_config.yaml file that tells YOLO where to find the training and validation data. You can generate it using the following Python script. Remember to update the path variable to your absolute path.
```bash
import yaml
from pathlib import Path

# Data configuration details
data_config = {
    # IMPORTANT: Update this to the absolute path of your dataset folder
    "path": "/path/to/your/project/yolo_train_visit_07",
    "train": "images/train",
    "val": "images/val",
    "nc": 2,  # Number of classes
    "names": {
        0: "flower",
        1: "fruitlet"
    }
}

# Directory to save the config file
save_dir = Path("yolo_train_visit_07")
save_dir.mkdir(parents=True, exist_ok=True)
yaml_path = save_dir / "data_config.yaml"

# Write the YAML file
with open(yaml_path, "w") as f:
    yaml.dump(data_config, f, sort_keys=False)

print(f"✅ data_config.yaml created at: {yaml_path}")
```

#### Step 5: Run Fine-Tuning
Start the YOLOv8 fine-tuning process using the configuration file and a pre-trained model.
```bash
python yolo_fine_tune/3_yolo_fine_tune.py --config yolo_train_visit_07/data_config.yaml --weights yolov8n.pt
```
The best-trained model weights will be saved in the runs/detect/ directory.

### Part 2: Generating Annotations with YOLO+SAM
Once the YOLO model is fine-tuned, you can use it to automatically generate segmentation masks for new, unseen images.
```bash
# Arguments: <images_to_annotate_folder> <visit_name> <path_to_best_yolo_weights>
python yolo_sam_annotator/main.py 'visit_08_04302024' '08' 'runs/detect/train/weights/best.pt'
```

## 📝 Citation
This work is based on the research published in the MangoSense paper. If you use this dataset or code in your research, please cite:
```
@article{ven2025mangosense,
  title={MangoSense: A time-series vision sensing dataset for mango tree segmentation and detection toward yield prediction},
  author={Ven, Janaksinh and Sharma, Charu and Syed, Azeemuddin},
  journal={Computers and Electronics in Agriculture},
  volume={237},
  pages={110524},
  year={2025},
  publisher={Elsevier}
}
```
## 🙏 Acknowledgements
Ultralytics YOLO: https://github.com/ultralytics/ultralytics

Segment Anything Model (SAM): https://github.com/facebookresearch/segment-anything

