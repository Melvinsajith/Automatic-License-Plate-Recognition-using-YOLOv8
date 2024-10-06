import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # or 'yolov8s.pt', etc.

# Define the class ID for 'car'
CAR_CLASS_ID = 2  # Typically, 'car' is class ID 2 in COCO dataset

def detect_cars_with_webcam():
    # Open a connection to the webcam (0 for the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Perform inference
        results = model(frame)

        # Loop through results and filter for cars
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes.data:  # Access bounding box data directly
                    x1, y1, x2, y2, conf, cls = box.tolist()  # Convert tensor to list
                    if int(cls) == CAR_CLASS_ID:  # Check if the detected object is a car
                        # Draw bounding box on the image
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'Car {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the output
        cv2.imshow('Detected Cars', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the webcam car detection
detect_cars_with_webcam()
