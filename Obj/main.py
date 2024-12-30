import os
import cv2
import numpy as np

def load_yolo_model(config_path, weights_path, classes_path):
    try:
        net = cv2.dnn.readNet(weights_path, config_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None, None, None
    try:
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: Classes file not found: {classes_path}")
        return None, None, None
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
    safe_objects = ["pen", "pencil", "toy", "book", "cup","chair","bed"]  # Add more safe objects here
    harmful_objects = ["knife", "scissors"]  # Add more harmful objects here
    results = []  # List to store results

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
    if indexes is None or len(boxes) == 0:  # Check if indexes is None or boxes is empty
        return frame, results

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            if label in safe_objects:
                color = (0, 255, 0)  # Green for safe objects
                result_str = f"{label}: Positive"
                results.append(5)  # numerical result
            elif label in harmful_objects:
                color = (0, 0, 255)  # Red for harmful objects
                result_str = f"{label}: Negative"
                results.append(-1)  # numerical result
            else:
                color = (255, 255, 255)  # White for others
                result_str = f"{label}: Neutral"
                results.append(2)  # numerical result
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(result_str)  # Print the result in the terminal
    return frame, results  # Return the results list

def process_image(image_path, net, classes, output_layers, output_folder):
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
            return None, None
    except Exception as e:
        print(f"Error reading image: {e}")
        return None, None

    outputs, width, height = detect_objects(net, output_layers, frame)
    if outputs is None: #check if detect_objects returns None
        return None, None
    frame, results = draw_predictions(outputs, width, height, frame, classes)

    if frame is None: #check if draw_predictions returns None
        return None, None
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(image_path)  # Get the filename
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{name}_detected{ext}")  # Use os.path.join
    cv2.imwrite(output_path, frame)
    print(f"Object detection results saved to: {output_path}")
    print(f"Numerical results: {results}")
    return results, output_path

def process_folder(folder_path, net, classes, output_layers, output_folder):
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: Invalid folder path: {folder_path}")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))]
    if not image_files:
        print(f"No images found in folder: {folder_path}")
        return

    all_results = {}
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing image: {image_path}")
        results, output_path = process_image(image_path, net, classes, output_layers, output_folder)
        if results is not None:
            all_results[output_path] = results

    if all_results:
        for output_path, results in all_results.items():
            if all(x == 1 for x in results):
                print(f"{output_path}: Overall result: Positive")
            elif all(x == -1 for x in results):
                print(f"{output_path}: Overall result: Negative")
            elif any(x == -1 for x in results):
                print(f"{output_path}: Overall result: Mixed (Contains Negative)")
            else:
                print(f"{output_path}: Overall result: Neutral")
    else:
        print("No images were processed to perform overall analysis.")

if __name__ == "__main__":
    # Define your paths here:
    base_folder = r"C:\Users\SUJITH\OneDrive\Desktop\Obj"  # your main folder
    image_folder = os.path.join(base_folder, "images")  # Construct path to 'images' folder
    config_path = os.path.join(base_folder, "yolov3.cfg")
    weights_path = os.path.join(base_folder, "yolov3.weights")
    classes_path = os.path.join(base_folder, "coco.names")
    output_folder = os.path.join(base_folder, "detected_images")

    # Check if files exist
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        exit()
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found: {weights_path}")
        exit()
    if not os.path.exists(classes_path):
        print(f"Error: Classes file not found: {classes_path}")
        exit()
    if not os.path.exists(image_folder): #check for image folder existence
        print(f"Error: Image folder not found: {image_folder}")
        exit()

    net, classes, output_layers = load_yolo_model(config_path, weights_path, classes_path)
    if net is None or classes is None or output_layers is None: #check if model loading is successful
        print("Error: Model loading failed.")
        exit()

    process_folder(image_folder, net, classes, output_layers, output_folder)
