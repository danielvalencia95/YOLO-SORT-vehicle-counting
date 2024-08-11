import cv2
import torch
import numpy as np
from sort import Sort
import time

# Define a dictionary to map class IDs to class names COCO-VEHICLE
class_names = {
    # 0: "bicycle",
    1: "bus",
    2: "car",
    3: "motorcycle",
    4: "truck",
}

# Use the YOLOv5 trained (you need to provide the correct path)
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/.../yolov5l_VC.pt', force_reload=True)

yolo_model = yolo_model.cuda()

# Initialize SORT tracker
tracker = Sort(
    max_age=300,           # Maximum number of frames a track is kept without updates
    min_hits=10,           # Minimum number of detections required to create a track
    iou_threshold=0.1,    # IOU (Intersection over Union) threshold for matching detections to tracks
)

# Open the video capture (you need to provide the correct video path)
cap = cv2.VideoCapture('Videos/car3.mp4')

# Vehicle inspection
track_vector = np.array([1])
class_vector = np.array([1])

# Capture line from the bottom
line = 200
# Cut-off percentage between 0 and 100%.
cut = 0.0
# Reshape - 0: No / 1: Yes
rs = 0
# Image shae 640*480 without reshape
w = 640
l = 480 + rs * (w-480)

l_cut = int(l*cut)
line = l*(1-cut) - rs*line*(64/48) - line*(1-rs)

start_YOLO = time.time()
while True:
    ret, frame = cap.read()
    if ret:
        if rs == 1:
            frame = cv2.resize(frame, (l, w), interpolation=cv2.INTER_AREA)
        if cut > 0:
            frame = frame[l_cut:l, 0:w]
    if not ret:
        break

    # Perform object detection using YOLO
    results = yolo_model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # Access class IDs of each detection
    class_ids = detections[:, -1].astype(int)

    # Prepare detections for SORT
    detections = detections[:, [0, 1, 2, 3]]  # Select columns (xmin, ymin, xmax, ymax)

    # Update SORT tracker with detections
    trackers = tracker.update(detections)

    # Visualize the tracking results
    for i, track in enumerate(trackers):
        x1, y1, x2, y2, track_id = track
        class_id = class_ids[i]  # Get the class ID for the current detection
        class_name = class_names.get(class_id, "Unknown")  # Map class ID to class name

        if line < y1:
            if track_id == 1: # First vehicle
                list_vehicle = [[track_id, class_id]]
                track_vector = track_id
                class_vector = class_id

            is_in_vector = np.any(track_vector == track_id)
            if not is_in_vector:
                track_vector = np.append(track_vector, track_id)
                class_vector = np.append(class_vector, class_id)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

end_YOLO = time.time()
cap.release()
cv2.destroyAllWindows()

print("Time for video processing:", end_YOLO-start_YOLO)

print("Bus: ", np.count_nonzero(class_vector == 1))
print("Car: ", np.count_nonzero(class_vector == 2))
print("Motorcycle: ", np.count_nonzero(class_vector == 3))
print("Truck: ", np.count_nonzero(class_vector == 4))





