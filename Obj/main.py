import os
import cv2
import numpy as np


def load_yolo_model(config_path, weights_path, classes_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


def detect_objects(net, output_layers, frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs, width, height


def draw_predictions(outputs, width, height, frame, classes, confidence_threshold=0.5):
    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def main():
    # put your file path here
    config_path = r"C:\Users\SUJITH\OneDrive\Desktop\Obj\yolov3.cfg"
    weights_path = r"C:\Users\SUJITH\OneDrive\Desktop\Obj\yolov3.weights"
    classes_path = r"C:\Users\SUJITH\OneDrive\Desktop\Obj\coco.names"

    net, classes, output_layers = load_yolo_model(config_path, weights_path, classes_path)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        outputs, width, height = detect_objects(net, output_layers, frame)
        frame = draw_predictions(outputs, width, height, frame, classes)
        cv2.imshow("Object Detection", frame)
        key = cv2.waitKey(10)
        print(key)
        if key != -1:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()