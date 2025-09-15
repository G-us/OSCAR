import cv2
import os
from ultralytics import YOLO

os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
# Load the YOLO model
model = YOLO("yolo11n.pt")

# Open the video file
cap = cv2.VideoCapture(1)

print("hello")

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame, stream = True)
        for result in results:
        # Visualize the results on the frame
            annotated_frame = result.plot()

        # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_frame)
            if result.boxes.cls == "person":
                xCenterOfBox = (result.boxes.xyxyn[0][0] + result.boxes.xyxyn[0][2]) / 2
                if xCenterOfBox < 0.45:
                    print("Object is on the left side of the frame")
                elif xCenterOfBox > 0.55:
                    print("Object is on the right side of the frame")
                else:
                    print("Object is in the center of the frame")


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()