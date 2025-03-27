import os
import cv2
import csv
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
import base64
from io import BytesIO
from PIL import Image

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Required for SocketIO
socketio = SocketIO(app)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads folder if it doesn't exist

# CSV file paths
STUDENTS_CSV = 'students.csv'
ATTENDANCE_CSV = 'attendance.csv'

# Initialize CSV files if they don't exist
if not os.path.exists(STUDENTS_CSV):
    with open(STUDENTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'id', 'image_path'])

if not os.path.exists(ATTENDANCE_CSV):
    with open(ATTENDANCE_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'id', 'time'])

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Helper functions
def train_recognizer():
    """Train the recognizer with all student images."""
    images = []
    labels = []
    label_dict = {}
    label_counter = 0

    try:
        with open(STUDENTS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            expected_headers = {'name', 'id', 'image_path'}
            if not expected_headers.issubset(reader.fieldnames):
                print(f"Error: {STUDENTS_CSV} has incorrect headers. Expected {expected_headers}, but found {reader.fieldnames}")
                return label_dict

            for row in reader:
                # Load the image
                img_path = row['image_path']
                if not os.path.exists(img_path):
                    print(f"Error: Image file {img_path} not found.")
                    continue

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Error: Failed to load image {img_path}.")
                    continue

                # Detect faces
                faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                if len(faces) != 1:
                    print(f"Error: No face or multiple faces detected in {img_path}.")
                    continue

                (x, y, w, h) = faces[0]
                face = img[y:y+h, x:x+w]

                # Assign a label to this student
                label_dict[label_counter] = {'name': row['name'], 'id': row['id']}
                images.append(face)
                labels.append(label_counter)
                label_counter += 1
    except FileNotFoundError:
        print(f"Error: {STUDENTS_CSV} not found.")
    except Exception as e:
        print(f"Error reading {STUDENTS_CSV}: {e}")

    if images:  # Train only if there are images
        recognizer.train(images, np.array(labels))
    return label_dict

def sanitize_filename(filename):
    """Sanitize the filename by replacing invalid characters."""
    invalid_chars = '<>:"/\\|?* '
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def save_student(name, student_id, image_data):
    """Save a new student to students.csv and store their image."""
    # Sanitize the name and student_id for the file name
    safe_name = sanitize_filename(name)
    safe_student_id = sanitize_filename(student_id)

    # Decode base64 image
    image_data = image_data.split(',')[1]  # Remove the "data:image/jpeg;base64," part
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    # Construct the file path
    filename = f"{safe_student_id}_{safe_name}.jpg"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Save the image
    image.save(image_path)

    # Check if the CSV file has the correct headers
    file_exists = os.path.exists(STUDENTS_CSV)
    if not file_exists:
        with open(STUDENTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'id', 'image_path'])

    # Save to CSV
    with open(STUDENTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, student_id, image_path])

def recognize_face(image_data):
    """Recognize a face in the given image."""
    label_dict = train_recognizer()  # Train the recognizer and get the label dictionary
    if not label_dict:  # If no students are registered
        return None, None

    # Decode base64 image
    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) != 1:
        return None, None  # Return None if no face or multiple faces are detected

    (x, y, w, h) = faces[0]
    face = img_gray[y:y+h, x:x+w]

    # Recognize the face
    label, confidence = recognizer.predict(face)
    if confidence < 100:  # Confidence threshold (lower is better)
        return label_dict[label]['name'], label_dict[label]['id']
    return None, None

def log_attendance(name, student_id):
    """Log attendance to attendance.csv and emit update."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(ATTENDANCE_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, student_id, timestamp])
    except Exception as e:
        print(f"Error writing to {ATTENDANCE_CSV}: {e}")
        return

    # Emit the new attendance record to all connected clients
    today = datetime.now().strftime('%Y-%m-%d')
    attendance = []
    try:
        with open(ATTENDANCE_CSV, 'r') as f:
            reader = csv.DictReader(f)
            # Check if the file is empty or has incorrect headers
            if reader.fieldnames is None:
                print(f"Error: {ATTENDANCE_CSV} is empty or has no headers.")
                return
            expected_headers = {'name', 'id', 'time'}
            if not expected_headers.issubset(reader.fieldnames):
                print(f"Error: {ATTENDANCE_CSV} has incorrect headers. Expected {expected_headers}, but found {reader.fieldnames}")
                return

            for i, row in enumerate(reader, 1):
                if row['time'].startswith(today):
                    attendance.append({
                        's_no': i,
                        'name': row['name'],
                        'id': row['id'],
                        'time': row['time']
                    })
    except FileNotFoundError:
        print(f"Error: {ATTENDANCE_CSV} not found.")
    except Exception as e:
        print(f"Error reading {ATTENDANCE_CSV}: {e}")
        return

    socketio.emit('attendance_update', {'attendance': attendance})

def get_today_attendance():
    """Get today's attendance records."""
    today = datetime.now().strftime('%Y-%m-%d')
    attendance = []
    try:
        with open(ATTENDANCE_CSV, 'r') as f:
            reader = csv.DictReader(f)
            expected_headers = {'name', 'id', 'time'}
            if not expected_headers.issubset(reader.fieldnames):
                print(f"Error: {ATTENDANCE_CSV} has incorrect headers. Expected {expected_headers}, but found {reader.fieldnames}")
                return attendance

            for i, row in enumerate(reader, 1):
                if row['time'].startswith(today):
                    attendance.append({
                        's_no': i,
                        'name': row['name'],
                        'id': row['id'],
                        'time': row['time']
                    })
    except FileNotFoundError:
        print(f"Error: {ATTENDANCE_CSV} not found.")
    except Exception as e:
        print(f"Error reading {ATTENDANCE_CSV}: {e}")
    return attendance

def get_all_attendance():
    """Get all attendance records."""
    attendance = []
    try:
        with open(ATTENDANCE_CSV, 'r') as f:
            reader = csv.DictReader(f)
            expected_headers = {'name', 'id', 'time'}
            if not expected_headers.issubset(reader.fieldnames):
                print(f"Error: {ATTENDANCE_CSV} has incorrect headers. Expected {expected_headers}, but found {reader.fieldnames}")
                return attendance

            for i, row in enumerate(reader, 1):
                attendance.append({
                    's_no': i,
                    'name': row['name'],
                    'id': row['id'],
                    'time': row['time']
                })
    except FileNotFoundError:
        print(f"Error: {ATTENDANCE_CSV} not found.")
    except Exception as e:
        print(f"Error reading {ATTENDANCE_CSV}: {e}")
    return attendance

def get_all_students():
    """Get all registered students."""
    students = []
    try:
        with open(STUDENTS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            expected_headers = {'name', 'id', 'image_path'}
            if not expected_headers.issubset(reader.fieldnames):
                print(f"Error: {STUDENTS_CSV} has incorrect headers. Expected {expected_headers}, but found {reader.fieldnames}")
                return students

            for i, row in enumerate(reader, 1):
                students.append({
                    's_no': i,
                    'name': row['name'],
                    'id': row['id']
                })
    except FileNotFoundError:
        print(f"Error: {STUDENTS_CSV} not found.")
    except Exception as e:
        print(f"Error reading {STUDENTS_CSV}: {e}")
    return students

# Flask Routes
@app.route('/')
def index():
    """Render the main dashboard."""
    today_attendance = get_today_attendance()
    total_students = len(get_all_students())
    return render_template('index.html', attendance=today_attendance, total_students=total_students)

@app.route('/add-student', methods=['POST'])
def add_student():
    """Add a new student."""
    name = request.form['name']
    student_id = request.form['id']
    image_data = request.form['image_data']

    if name and student_id and image_data:
        save_student(name, student_id, image_data)
        # Emit the updated total students count
        total_students = len(get_all_students())
        socketio.emit('total_students_update', {'total_students': total_students})
    return redirect(url_for('index'))

@app.route('/take-attendance', methods=['POST'])
def take_attendance():
    """Take attendance by recognizing a face."""
    image_data = request.form['image_data']
    if image_data:
        # Recognize the face
        name, student_id = recognize_face(image_data)
        if name and student_id:
            log_attendance(name, student_id)
    return redirect(url_for('index'))

@app.route('/view-attendance')
def view_attendance():
    """View all attendance records."""
    all_attendance = get_all_attendance()
    return render_template('view_attendance.html', attendance=all_attendance)

@app.route('/registered-students')
def registered_students():
    """View all registered students."""
    students = get_all_students()
    return render_template('registered_students.html', students=students)

# Run the app with SocketIO
if __name__ == '__main__':
    socketio.run(app, debug=True)