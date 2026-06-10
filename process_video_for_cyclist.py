import cv2
import os
from ultralytics import YOLO
import torch
import yt_dlp

IOU_DISTANCE_THRESHOULD = 0.3

SHOULD_PROCESS_VIDEO_CLIPS_FROM_URLS = False

SHOULD_PROCESS_VIDEO_CLIPS_FROM_FOLDER = True
VIDEO_CLIPS_FOLDER_PATH = 'output_clips'

def download_youtube_video(url, output_filename="downloaded_video.mp4", output_directory="."):
    """Downloads a video from a given URL using yt-dlp."""
    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
        except OSError as e:
            print(f"Error creating directory {output_directory}: {e}")
            return None
    output_path = os.path.join(output_directory, output_filename)
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path,
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Download complete: {output_path}")
        return output_path
    except yt_dlp.DownloadError as e:
        print(f"Download error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

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

def detect_cyclists(video_path, model_path, confidence_threshold=0.5, output_image_dir=None, image_counter=0, overlap_between_frames = 0.25):
    """Detects cyclists in a video using YOLO, and saves the entire detected image, with IOU check."""
    serial_number = image_counter
    downloaded_video_path = None
    if video_path.startswith("http"):
        downloaded_video_path = download_youtube_video(video_path, output_directory="./dataset/")
        if downloaded_video_path is None:
            return
        video_path = downloaded_video_path
    else:
        if not os.path.exists(video_path):
            print(f"File dont not exists {video_path}")
            return 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        if downloaded_video_path:
            os.remove(downloaded_video_path)
        return

    last_cyclist_box = None

    if output_image_dir:
        os.makedirs(output_image_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        cyclists_detected = False
        current_cyclist_box = None

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                if confidence >= confidence_threshold and model.names[class_id] == "bicycle":
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cyclists_detected = True
                    current_cyclist_box = (x1, y1, x2, y2)
                    break

        if cyclists_detected:
            save_image = True
            if last_cyclist_box is not None and current_cyclist_box is not None:
                iou = calculate_iou(last_cyclist_box, current_cyclist_box)
                if iou > overlap_between_frames:
                    save_image = False

            if save_image and output_image_dir:
                image_filename = f"{image_counter}.jpg"
                cv2.imwrite(os.path.join(output_image_dir, image_filename), frame)
                image_counter += 1

            last_cyclist_box = current_cyclist_box



    cap.release()

    if downloaded_video_path:
        os.remove(downloaded_video_path)

    return image_counter - serial_number

# Example Usage (List of Video URLs):
if __name__ == "__main__":
    video_urls = [

    ]
    model_file = "yolov8l.pt"
    output_image_directory = "dataset/images"

    image_counter = 0

    if SHOULD_PROCESS_VIDEO_CLIPS_FROM_URLS:
        for video_url in video_urls:
            print(f"Processing video: {video_url}")
            image_counter += detect_cyclists(video_url, model_file, output_image_dir=output_image_directory, image_counter=image_counter, overlap_between_frames=0.25)
            print(f"Finished processing video: {image_counter}")

    image_counter = 5083
    if SHOULD_PROCESS_VIDEO_CLIPS_FROM_FOLDER:
        # Ensure the folder path exists
        if not os.path.exists(VIDEO_CLIPS_FOLDER_PATH):
            print(f"Error: Folder path does not exist: {VIDEO_CLIPS_FOLDER_PATH}")
            exit()

        # Get all files in the folder
        video_files = [
            os.path.join(VIDEO_CLIPS_FOLDER_PATH, f)
            for f in os.listdir(VIDEO_CLIPS_FOLDER_PATH)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]

        if not video_files:
            print(f"No video files found in: {VIDEO_CLIPS_FOLDER_PATH}")
            exit()

        for video_file in video_files:
            print(f"Processing video: {video_file}")
            image_counter += detect_cyclists(video_file,
                                             model_file,
                                             output_image_dir=output_image_directory,
                                             image_counter=image_counter,
                                             overlap_between_frames=0.2)
            print(f"Finished processing video: {image_counter}")
