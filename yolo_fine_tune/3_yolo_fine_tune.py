from ultralytics import YOLO
import matplotlib.pyplot as plt
import argparse 

def train_yolo(data_yaml_file_path, model_weights_file_path='yolov8n.pt', epochs=100):
    model = YOLO(model_weights_file_path)
    results = model.train(data = data_yaml_file_path, epochs= epochs)

    return results

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="A script that takes three input arguments(visit number, yolo weights path).")
    parser.add_argument("--config", type=str, help="yolo model data confing file path ex. yolo_train_visit_07/data_config.yaml")
    # parser.add_argument("arg2", type=str, help="Model weights file path")
    # parser.add_argument("--optional_arg", type=str, default="yolov8n.pt", help="Model weights file path")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Model weights file path (default: yolov8m.pt)")
    args = parser.parse_args()

    data_yaml_file_path = args.config
    model_weights_file_path = args.weights

    results = train_yolo(data_yaml_file_path, model_weights_file_path)

    # prepare the dataset in the yolo formate make .yaml file and give path to train function

    # python yolo_fine_tune/3_yolo_fine_tune.py --config yolo_train_visit_07/data_config.yaml --weights runs/detect/train_visit_06/weights/best.pt