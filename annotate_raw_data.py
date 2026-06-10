import cv2
import os
from ultralytics import YOLO
import torch

def calculate_iou(box1, box2):
    """Calculates IoU between two bounding boxes."""
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def convert_to_yolo_format(image_width, image_height, x1, y1, x2, y2):
    """
    Converts bounding box coordinates from pixel coordinates (x1, y1, x2, y2)
    to normalized YOLO format (center_x, center_y, width, height).

    Args:
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        x1 (int): x-coordinate of the top-left corner.
        y1 (int): y-coordinate of the top-left corner.
        x2 (int): x-coordinate of the bottom-right corner.
        y2 (int): y-coordinate of the bottom-right corner.

    Returns:
        tuple: (center_x, center_y, width, height) in normalized YOLO format.
    """
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    normalized_center_x = center_x / image_width
    normalized_center_y = center_y / image_height
    normalized_width = box_width / image_width
    normalized_height = box_height / image_height

    return normalized_center_x, normalized_center_y, normalized_width, normalized_height


def process_images_for_cyclist_annotations(image_dir, model_path, output_annotation_dir, iou_threshold=0.1):
    """
    Processes images from a directory, detects cyclists, persons, and bicycles, and creates normalized annotation files.

    Args:
        image_dir (str): Path to the directory containing images.
        model_path (str): Path to the YOLO model weights.
        output_annotation_dir (str): Path to save annotation files.
        iou_threshold (float): IOU threshold to consider a person and bicycle as a cyclist.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)

    print(f"Processing using {device}")
    os.makedirs(output_annotation_dir, exist_ok=True)

    if not os.path.exists(image_dir):
        print(f"{image_dir} - Directory does not exist")
        return

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            continue

        image_height, image_width = frame.shape[:2]  # Get image dimensions
        results = model(frame)
        person_boxes = []
        bicycle_boxes = []
        annotations = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if model.names[class_id] == "person":
                    person_boxes.append((x1, y1, x2, y2))
                elif model.names[class_id] == "bicycle":
                    bicycle_boxes.append((x1, y1, x2, y2))

        if person_boxes and bicycle_boxes:
            cyclist_detected = False
            for person_box in person_boxes:
                for bicycle_box in bicycle_boxes:
                    iou = calculate_iou(person_box, bicycle_box)
                    if iou > iou_threshold:
                        # Create cyclist bounding box
                        x1 = min(person_box[0], bicycle_box[0])
                        y1 = min(person_box[1], bicycle_box[1])
                        x2 = max(person_box[2], bicycle_box[2])
                        y2 = max(person_box[3], bicycle_box[3])
                        center_x, center_y, width, height = convert_to_yolo_format(image_width, image_height, x1, y1, x2, y2)
                        annotations.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")  # 0 is the cyclist class.
                        cyclist_detected = True
            if not cyclist_detected:
                for person_box in person_boxes:
                    center_x, center_y, width, height = convert_to_yolo_format(image_width, image_height, person_box[0], person_box[1], person_box[2], person_box[3])
                    annotations.append(f"1 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")  # 1 is the person class
                for bicycle_box in bicycle_boxes:
                    center_x, center_y, width, height = convert_to_yolo_format(image_width, image_height, bicycle_box[0], bicycle_box[1], bicycle_box[2], bicycle_box[3])
                    annotations.append(f"2 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")  # 2 is the bicycle class
        elif person_boxes:
            for person_box in person_boxes:
                center_x, center_y, width, height = convert_to_yolo_format(image_width, image_height, person_box[0], person_box[1], person_box[2], person_box[3])
                annotations.append(f"1 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")  # 1 is the person class
        elif bicycle_boxes:
            for bicycle_box in bicycle_boxes:
                center_x, center_y, width, height = convert_to_yolo_format(image_width, image_height, bicycle_box[0], bicycle_box[1], bicycle_box[2], bicycle_box[3])
                annotations.append(f"2 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")  # 2 is the bicycle class

        if annotations:
            annotation_filename = os.path.splitext(image_file)[0] + ".txt"
            annotation_path = os.path.join(output_annotation_dir, annotation_filename)
            with open(annotation_path, "w") as f:
                f.write("\n".join(annotations))



# Example usage:
if __name__ == "__main__":
    image_directory = "./dataset/images/"  # Replace with your image directory
    yolo_model_path = "yolov8l.pt"  # Replace with your YOLO model path
    annotation_directory = "./dataset/annotations"  # Replace with your annotation directory

    process_images_for_cyclist_annotations(image_directory, yolo_model_path, annotation_directory)
