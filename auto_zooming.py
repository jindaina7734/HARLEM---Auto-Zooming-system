!git clone https://github.com/ultralytics/yolov5
!cd yolov5
!pip install -r /content/yolov5/requirements.txt

import torch
import cv2
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#change the destination on your own
input_folder = ''
output_folder = ''


os.makedirs(output_folder, exist_ok=True)

#size & bounding box padding 
fixed_width, fixed_height = 480, 800
padding = 50
smoothing_window = 5

#moving average smoothing
def moving_average(data, window_size):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

#iterate all vids in a folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.avi', '.mp4', '.mov')):  #add other formats here (Optional)
        input_path = os.path.join(input_folder, filename)
        output_filename = f'Output_{filename}'
        output_path = os.path.join(output_folder, output_filename)

        print(f"Processing file: {filename}")

        cap = cv2.VideoCapture(input_path)
        out = None

        frame_count = 0
        max_width, max_height = 0, 0 
        bboxes_x_min = []
        bboxes_y_min = []
        bboxes_x_max = []
        bboxes_y_max = []

        #1st pass: Determine largest bounding box size
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame)

            detections = results.xyxy[0].numpy()
            highest_confidence_person = None
            highest_confidence_score = 0.0
            largest_bbox_area = 0.0

            #find the person with the largest bounding box or highest confidence (only get one)
            for detection in detections:
                x_min, y_min, x_max, y_max, confidence, class_id = detection
                if class_id == 0:  #class_id = 0 represents 'person' class in COCO dataset
                    bbox_width, bbox_height = x_max - x_min, y_max - y_min

                    #choose person with the largest bounding box or highest confidence
                    if bbox_width * bbox_height > largest_bbox_area:
                        largest_bbox_area = bbox_width * bbox_height
                        highest_confidence_person = (x_min, y_min, x_max, y_max)
                        highest_confidence_score = confidence

                    #update max width and height
                    if bbox_width > max_width:
                        max_width = bbox_width
                    if bbox_height > max_height:
                        max_height = bbox_height

        cap.release()

        #2nd pass: Apply consistent bounding box size + dynamic tracking
        cap = cv2.VideoCapture(input_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to retrieve frame or end of video.")
                break

            results = model(frame)

            detections = results.xyxy[0].numpy()
            highest_confidence_person = None
            highest_confidence_score = 0.0

            #find the person with largest bounding box/highest confidence (again)
            for detection in detections:
                x_min, y_min, x_max, y_max, confidence, class_id = detection
                if class_id == 0:
                    bbox_width, bbox_height = x_max - x_min, y_max - y_min
                    highest_confidence_person = (x_min, y_min, x_max, y_max)

            #if a person is detected, adjust the bounding box size
            if highest_confidence_person:
                x_min, y_min, x_max, y_max = highest_confidence_person

                #calculate the center
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2

                # + padding
                bboxes_x_min.append(center_x - max_width / 2 - padding)
                bboxes_y_min.append(center_y - max_height / 2 - padding)
                bboxes_x_max.append(center_x + max_width / 2 + padding)
                bboxes_y_max.append(center_y + max_height / 2 + padding)

                #apply smoothing
                if len(bboxes_x_min) > smoothing_window:
                    smoothed_x_min = moving_average(bboxes_x_min, smoothing_window)[-1]
                    smoothed_y_min = moving_average(bboxes_y_min, smoothing_window)[-1]
                    smoothed_x_max = moving_average(bboxes_x_max, smoothing_window)[-1]
                    smoothed_y_max = moving_average(bboxes_y_max, smoothing_window)[-1]
                else:
                    smoothed_x_min = bboxes_x_min[-1]
                    smoothed_y_min = bboxes_y_min[-1]
                    smoothed_x_max = bboxes_x_max[-1]
                    smoothed_y_max = bboxes_y_max[-1]

                #ensure bounding box coordinates are within frame bounds
                smoothed_x_min = int(max(0, smoothed_x_min))
                smoothed_y_min = int(max(0, smoothed_y_min))
                smoothed_x_max = int(min(frame.shape[1], smoothed_x_max))
                smoothed_y_max = int(min(frame.shape[0], smoothed_y_max))

                # crop & resize the frame 
                cropped_frame = frame[smoothed_y_min:smoothed_y_max, smoothed_x_min:smoothed_x_max]
                cropped_frame_resized = cv2.resize(cropped_frame, (fixed_width, fixed_height))

                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(
                        output_path,
                        fourcc,
                        30,  # fps
                        (fixed_width, fixed_height) 
                    )
                out.write(cropped_frame_resized)

                frame_count += 1

        print(f"Total frames of {filename}: {frame_count}")
        cap.release()
        if out:
            out.release()

print("Done. All videos have been processed.")