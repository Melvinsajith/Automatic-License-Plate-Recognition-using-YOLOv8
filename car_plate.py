import cv2
from ultralytics import YOLO

# Load the YOLOv8 models
car_model = YOLO('yolov8n.pt')  # Model for car detection
plate_model = YOLO('C:\\Users\\USER\\Downloads\\Automatic-License-Plate-Recognition-using-YOLOv8-main\\license_plate_detector.pt')  # Model for number plate detection (replace with your model)

# Define the class IDs
CAR_CLASS_ID = 2      # Class ID for 'car'
PLATE_CLASS_ID = 0    # Replace with the actual class ID for 'number plate' in your model

def detect_cars_and_plates_with_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Perform inference for cars
        car_results = car_model(frame)

        # Process car results
        for result in car_results:
            boxes = result.boxes
            if boxes is not None:
                for car_box in boxes.data:
                    x1, y1, x2, y2, car_conf, car_cls = car_box.tolist()
                    if int(car_cls) == CAR_CLASS_ID:
                        # Draw bounding box for car
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'Car {car_conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Crop the region inside the car bounding box
                        car_roi = frame[int(y1):int(y2), int(x1):int(x2)]

                        # Perform inference for number plates inside the car's bounding box
                        plate_results = plate_model(car_roi)

                        # Process plate results
                        for plate_result in plate_results:
                            plate_boxes = plate_result.boxes
                            if plate_boxes is not None:
                                for plate_box in plate_boxes.data:
                                    px1, py1, px2, py2, plate_conf, plate_cls = plate_box.tolist()
                                    if int(plate_cls) == PLATE_CLASS_ID:
                                        # Draw bounding box for number plate inside the car box
                                        cv2.rectangle(frame, (int(x1) + int(px1), int(y1) + int(py1)), 
                                                      (int(x1) + int(px2), int(y1) + int(py2)), (255, 0, 0), 2)
                                        cv2.putText(frame, f'Plate {plate_conf:.2f}', 
                                                    (int(x1) + int(px1), int(y1) + int(py1) - 10), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the output
        cv2.imshow('Detected Cars and Plates', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the combined detection
detect_cars_and_plates_with_webcam()
