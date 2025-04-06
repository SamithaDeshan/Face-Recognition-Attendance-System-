import os
import mediapipe as mp
import face_recognition
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# Suppress TensorFlow warnings (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'test-secret-key'
socketio = SocketIO(app)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Sample known face encoding (for testing face_recognition)
# Replace this with an actual image of a known person for a real test
known_image_path = "test_face.jpg"  # You need to provide a sample image
known_face_encoding = None

def load_known_face():
    global known_face_encoding
    if os.path.exists(known_image_path):
        image = face_recognition.load_image_file(known_image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encoding = encodings[0]
            print("Known face loaded successfully.")
        else:
            print("Error: No face detected in the known image.")
    else:
        print(f"Error: {known_image_path} not found. Please provide a test image.")

# Load a known face on startup (for testing face_recognition)
load_known_face()

# Helper function to detect and recognize faces
def detect_and_recognize_faces(image_data):
    # Decode the base64 image
    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    img_array = np.array(image)

    # Image is already in RGB format (from PIL)
    img_rgb = img_array

    # Detect faces using MediaPipe
    results = face_detection.process(img_rgb)
    face_coords = []
    recognized_names = []

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = img_array.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Ensure the bounding box is within the image
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x + width)
            y2 = min(h, y + height)

            face_coords.append({'x': x, 'y': y, 'w': width, 'h': height})

            # Crop the face region for recognition
            face_image = img_array[y:y2, x:x2]
            face_encodings = face_recognition.face_encodings(face_image)

            name = "Unknown"
            if face_encodings and known_face_encoding is not None:
                encoding = face_encodings[0]
                matches = face_recognition.compare_faces([known_face_encoding], encoding, tolerance=0.6)
                if matches[0]:
                    name = "Known Person"  # Replace with the actual name if you have a known person
            recognized_names.append(name)

    return face_coords, recognized_names

# Flask Routes
@app.route('/')
def index():
    """Render a simple page to test the libraries."""
    return render_template('test.html')

@app.route('/detect-and-recognize', methods=['POST'])
def detect_and_recognize():
    """Detect and recognize faces in the image."""
    image_data = request.form.get('image_data')
    if not image_data:
        socketio.emit('message', {'message': 'No image data provided.'})
        return jsonify({'status': 'error', 'message': 'No image data'})

    try:
        face_coords, recognized_names = detect_and_recognize_faces(image_data)
        if not face_coords:
            socketio.emit('message', {'message': 'No faces detected.'})
        else:
            for i, (coords, name) in enumerate(zip(face_coords, recognized_names)):
                socketio.emit('message', {
                    'message': f'Face {i+1}: {name} at position (x: {coords["x"]}, y: {coords["y"]}, w: {coords["w"]}, h: {coords["h"]})'
                })
        return jsonify({'status': 'success', 'faces': face_coords, 'names': recognized_names})
    except Exception as e:
        socketio.emit('message', {'message': f'Error: {str(e)}'})
        return jsonify({'status': 'error', 'message': str(e)})

# Run the app with SocketIO
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)