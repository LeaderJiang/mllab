# %%
import cv2
from ultralytics import YOLO


# %%
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
model = YOLO('shd_voc2028_yolov8.pt')


# %%
# 使用影片
# video_path = "helmet.mp4"
# cap = cv2.VideoCapture(video_path)

# 使用攝影機
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Catcher", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


# %%
