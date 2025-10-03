import json

# Function to calculate the IoU of two bounding boxes
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # Calculate intersection coordinates
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    # Calculate area of intersection
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    # Calculate areas of each box and the union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union = box1_area + box2_area - intersection

    # Calculate IoU
    iou = intersection / union if union != 0 else 0
    return iou

# Function to merge two bounding boxes
def merge_boxes(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # Find coordinates for merged box
    merged_box = [
        min(x1, x1_),
        min(y1, y1_),
        max(x2, x2_),
        max(y2, y2_)
    ]
    return merged_box

# Function to process the bounding boxes and merge them based on IoU and label
def merge_bboxes(data, iou_threshold=0.9):
    merged_data = {"class_names": data["class_names"]}
    
    for image_id, image_data in data.items():
        if image_id == "class_names":
            continue
        
        bboxes = image_data.get("bboxes", [])
        labels = image_data.get("labels", [])
        
        # Check all pairs of bounding boxes for merging
        merged_bboxes = []
        merged_labels = []
        skip_indices = set()

        for i in range(len(bboxes)):
            if i in skip_indices:
                continue
            box1 = bboxes[i]
            label1 = labels[i]
            merged_box = box1
            for j in range(i + 1, len(bboxes)):
                if j in skip_indices:
                    continue
                box2 = bboxes[j]
                label2 = labels[j]
                if label1 == label2:
                    iou = calculate_iou(merged_box, box2)
                    if iou > iou_threshold:
                        print(f'merged {image_id}, box{i}, box{j} ')
                        merged_box = merge_boxes(merged_box, box2)
                        skip_indices.add(j)
            merged_bboxes.append(merged_box)
            merged_labels.append(label1)
        
        merged_data[image_id] = {
            "bboxes": merged_bboxes,
            "labels": merged_labels
        }

    return merged_data
    
if __name__=='__main__':

    # Load JSON data
    with open("yolo_out_for_image_bbox_data.json", "r") as f:
        data = json.load(f)

    # Process and merge bounding boxes
    merged_data = merge_bboxes(data)

    # Save to JSON file
    with open("yolo_out_for_image_bbox_data.json", "w") as f:
        json.dump(merged_data, f, indent=4)

    print("Bounding boxes merged and saved to yolo_out_for_image_bbox_data.json.")
