import cv2
from ultralytics import YOLO
from collections import defaultdict
import cvzone
import os

# Change working directory
os.chdir(r"C:\Users\RTC\Downloads\Youtube_content\classwise-using-YOLO11\Vehicle-detection-and-tracking-classwise-using-YOLO11")

# Load the YOLO model
model = YOLO('yolo11s.pt')

class_list = model.names
print(class_list)

# Open the video file
cap = cv2.VideoCapture('test_videos/4.mp4')

line_y_red = 430  # Red line position

# Dictionary to store object counts by class
class_counts = defaultdict(int)

# Dictionary to keep track of object IDs that have crossed the line
crossed_ids = set()

# Get video properties to set output file parameters
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize VideoWriter to save the output video
output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, classes=[1, 2, 3, 5, 6, 7])
    print(results)

    # Ensure results are not empty
    if results[0].boxes.data is not None:
        # Get the detected boxes, their class indices, and track IDs
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        # Draw the red line
        cv2.line(frame, (690, line_y_red), (1130, line_y_red), (0, 0, 255), 3)

        # Loop through each detected object
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2  # Calculate the center point
            cy = (y1 + y2) // 2

            class_name = class_list[class_idx]

            # Draw a circle for the center point
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Annotate with bounding box, class name, and ID
            color = (0, 255, 0) if cy <= line_y_red else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cvzone.putTextRect(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), scale=0.7, thickness=1, colorR=color)

            # Check if the object has crossed the red line
            if cy > line_y_red and track_id not in crossed_ids:
                # Mark the object as crossed
                crossed_ids.add(track_id)
                class_counts[class_name] += 1

        # Display the counts on the frame
        y_offset = 30
        for class_name, count in class_counts.items():
            cvzone.putTextRect(frame, f"{class_name}: {count}", (50, y_offset), scale=1, thickness=1, colorR=(0, 255, 0))
            y_offset += 40

    # Show the frame
    cv2.imshow("YOLO Object Tracking & Counting", frame)

    # Write the frame to the output video file
    output_video.write(frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()
