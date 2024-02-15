from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load YOLO
weights_path = 'yolov4.weights'
config_path = 'yolov4.cfg'
classes_path = 'coco.names'
net = cv2.dnn.readNet(weights_path, config_path)

# Load COCO class labels
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define your categories based on COCO class labels
recycling_labels = ['bottle']  # Add more class names as needed
compost_labels = ['apple', 'banana', 'carrot', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake']  # Add more class names as needed
trash_labels = []  # Add any other class names you consider 'trash'

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json(force=True)
    image_encoded = data['image']
    image_decoded = base64.b64decode(image_encoded)
    image_np = np.frombuffer(image_decoded, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

    results = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                class_name = classes[class_id]
                if class_name in recycling_labels + compost_labels + trash_labels:
                    # Scale the bounding box back to the image size
                    box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    (centerX, centerY, width, height) = box.astype('int')
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # Make sure the bounding box does not go out of the frame
                    x = max(0, x)
                    y = max(0, y)
                    width = min(image.shape[1] - x, width)  # width should not exceed image's width
                    height = min(image.shape[0] - y, height)  # height should not exceed image's height

                    # Determine the label based on the class_name
                    if class_name in recycling_labels:
                        label = 'Recycling'
                    elif class_name in compost_labels:
                        label = 'Compost'
                    else:
                        label = 'Trash'  # Assume any other detected item is trash
                    
                    results.append({
                        'label': label,
                        'confidence': float(confidence),
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height
                    })

                    # Draw bounding box and label on the image
                    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Encode the modified image to return as a response
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode()

    # Send JSON response with the image and detections
    return jsonify({"image": jpg_as_text, "detections": results})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
