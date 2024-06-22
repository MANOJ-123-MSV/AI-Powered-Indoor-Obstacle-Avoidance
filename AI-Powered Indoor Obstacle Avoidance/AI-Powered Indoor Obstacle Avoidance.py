import cv2
import numpy as np

# Load YOLO
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Detect objects in the image
def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

# Draw bounding boxes on the image
def draw_labels(boxes, confidences, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
    return img

# Navigation Algorithm (Simplified Example)
def navigate(boxes, img_shape):
    height, width, _ = img_shape
    for box in boxes:
        x, y, w, h = box
        center_x = x + w / 2
        center_y = y + h / 2

        # Check if the obstacle is in the center of the image
        if center_x > width / 3 and center_x < 2 * width / 3:
            return "Move Left" if center_x < width / 2 else "Move Right"
        elif center_y > height / 3 and center_y < 2 * height / 3:
            return "Move Up" if center_y < height / 2 else "Move Down"
    return "Move Forward"

# Main function to run the obstacle avoidance
if __name__ == "__main__":
    net, classes, output_layers = load_yolo()
    
    # Load a test video or webcam feed
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide video file path
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confidences, class_ids = detect_objects(frame, net, output_layers)
        frame = draw_labels(boxes, confidences, class_ids, classes, frame)
        action = navigate(boxes, frame.shape)

        # Display the action on the frame
        cv2.putText(frame, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Obstacle Avoidance", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
