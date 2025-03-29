import os
import cv2
import csv
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify
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
                print(
                    f"Error: {STUDENTS_CSV} has incorrect headers. Expected {expected_headers}, but found {reader.fieldnames}")
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
                face = img[y:y + h, x:x + w]

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
    today = datetime.now().strftime('%Y-%m-%d')

    # Check if the student has already marked attendance today
    try:
        with open(ATTENDANCE_CSV, 'r') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                print(f"Error: {ATTENDANCE_CSV} is empty or has no headers.")
            else:
                for row in reader:
                    if row['name'] == name and row['id'] == student_id and row['time'].startswith(today):
                        # Emit a message to the client indicating attendance is already marked
                        socketio.emit('attendance_message',
                                      {'message': f'Attendance already marked for {name} (ID: {student_id}) today.'})
                        return
    except FileNotFoundError:
        print(f"Error: {ATTENDANCE_CSV} not found.")
    except Exception as e:
        print(f"Error reading {ATTENDANCE_CSV}: {e}")

    # If no duplicate is found, proceed to log the attendance
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Check if the CSV file has the correct headers
    file_exists = os.path.exists(ATTENDANCE_CSV)
    if not file_exists:
        with open(ATTENDANCE_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'id', 'time'])

    try:
        with open(ATTENDANCE_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, student_id, timestamp])
    except Exception as e:
        print(f"Error writing to {ATTENDANCE_CSV}: {e}")
        return

    # Emit the new attendance record to all connected clients
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
                print(
                    f"Error: {ATTENDANCE_CSV} has incorrect headers. Expected {expected_headers}, but found {reader.fieldnames}")
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
    socketio.emit('attendance_message', {'message': f'Attendance marked for {name} (ID: {student_id}).'})


def get_today_attendance():
    """Get today's attendance records."""
    today = datetime.now().strftime('%Y-%m-%d')
    attendance = []
    try:
        with open(ATTENDANCE_CSV, 'r') as f:
            reader = csv.DictReader(f)
            expected_headers = {'name', 'id', 'time'}
            if reader.fieldnames is None or not expected_headers.issubset(reader.fieldnames):
                print(
                    f"Error: {ATTENDANCE_CSV} has incorrect headers. Expected {expected_headers}, but found {reader.fieldnames}")
                # Read the existing data (treat the first row as data if headers are incorrect)
                f.seek(0)
                lines = f.readlines()
                if lines and reader.fieldnames is not None:
                    # Assume the first row is data if it doesn't match the expected headers
                    data = [line.strip().split(',') for line in lines if line.strip()]
                else:
                    data = []

                # Recreate the file with correct headers
                with open(ATTENDANCE_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['name', 'id', 'time'])
                    # Write back the data
                    for row in data:
                        if len(row) == 3:  # Ensure the row has the correct number of columns
                            writer.writerow(row)
                return attendance  # Return empty list for now, will reload on next call

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
        # Create the file with correct headers
        with open(ATTENDANCE_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'id', 'time'])
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
            if reader.fieldnames is None or not expected_headers.issubset(reader.fieldnames):
                print(
                    f"Error: {ATTENDANCE_CSV} has incorrect headers. Expected {expected_headers}, but found {reader.fieldnames}")
                # Read the existing data (treat the first row as data if headers are incorrect)
                f.seek(0)
                lines = f.readlines()
                if lines and reader.fieldnames is not None:
                    # Assume the first row is data if it doesn't match the expected headers
                    data = [line.strip().split(',') for line in lines if line.strip()]
                else:
                    data = []

                # Recreate the file with correct headers
                with open(ATTENDANCE_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['name', 'id', 'time'])
                    # Write back the data
                    for row in data:
                        if len(row) == 3:  # Ensure the row has the correct number of columns
                            writer.writerow(row)
                return attendance  # Return empty list for now, will reload on next call

            for i, row in enumerate(reader, 1):
                attendance.append({
                    's_no': i,
                    'name': row['name'],
                    'id': row['id'],
                    'time': row['time']
                })
    except FileNotFoundError:
        print(f"Error: {ATTENDANCE_CSV} not found.")
        # Create the file with correct headers
        with open(ATTENDANCE_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'id', 'time'])
    except Exception as e:
        print(f"Error reading {ATTENDANCE_CSV}: {e}")
    return attendance

def student_exists(student_id):
    """Check if a student with the given ID already exists in students.csv."""
    try:
        with open(STUDENTS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['id'] == student_id:
                    return True
    except FileNotFoundError:
        print(f"Error: {STUDENTS_CSV} not found.")
    except Exception as e:
        print(f"Error reading {STUDENTS_CSV}: {e}")
    return False

def get_all_students():
    """Get all registered students."""
    students = []
    try:
        with open(STUDENTS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            expected_headers = {'name', 'id', 'image_path'}
            if reader.fieldnames is None or not expected_headers.issubset(reader.fieldnames):
                print(
                    f"Error: {STUDENTS_CSV} has incorrect headers. Expected {expected_headers}, but found {reader.fieldnames}")
                # Read the existing data (treat the first row as data if headers are incorrect)
                f.seek(0)
                lines = f.readlines()
                if lines and reader.fieldnames is not None:
                    # Assume the first row is data if it doesn't match the expected headers
                    data = [line.strip().split(',') for line in lines if line.strip()]
                else:
                    data = []

                # Recreate the file with correct headers
                with open(STUDENTS_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['name', 'id', 'image_path'])
                    # Write back the data
                    for row in data:
                        if len(row) == 3:  # Ensure the row has the correct number of columns
                            writer.writerow(row)
                return students  # Return empty list for now, will reload on next call

            for i, row in enumerate(reader, 1):
                students.append({
                    's_no': i,
                    'name': row['name'],
                    'id': row['id']
                })
    except FileNotFoundError:
        print(f"Error: {STUDENTS_CSV} not found.")
        # Create the file with correct headers
        with open(STUDENTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'id', 'image_path'])
    except Exception as e:
        print(f"Error reading {STUDENTS_CSV}: {e}")
    return students


# Flask Routes

@app.route('/')
def index():
    """Render the main dashboard."""
    today_attendance = get_today_attendance()
    total_students = len(get_all_students())
    current_date = datetime.now().strftime('%Y-%m-%d')
    return render_template('index.html', attendance=today_attendance, total_students=total_students,
                           current_date=current_date)


@app.route('/add-student', methods=['POST'])
def add_student():
    """Add a new student."""
    name = request.form.get('name')
    student_id = request.form.get('id')
    image_data = request.form.get('image_data')

    if name and student_id and image_data:
        # Check if a student with this ID already exists
        if student_exists(student_id):
            socketio.emit('student_message', {'message': f'This student is already registered with ID {student_id}.'})
            return jsonify({'status': 'error', 'message': 'Student already exists'})

        # Check if the face already exists in the database
        existing_name, existing_id = recognize_face(image_data)
        if existing_name and existing_id:
            socketio.emit('student_message',
                          {'message': f'This face is already registered for {existing_name} (ID: {existing_id}).'})
            return jsonify({'status': 'error', 'message': 'Face already registered'})

        # If no duplicate ID or face is found, proceed to add the student
        save_student(name, student_id, image_data)
        # Emit the updated total students count
        total_students = len(get_all_students())
        socketio.emit('total_students_update', {'total_students': total_students})
        # Emit a confirmation message
        socketio.emit('student_message', {'message': f'Student {name} (ID: {student_id}) added successfully.'})
        return jsonify({'status': 'success', 'message': 'Student added successfully'})

    socketio.emit('student_message', {'message': 'Invalid data. Please provide a name, ID, and image.'})
    return jsonify({'status': 'error', 'message': 'Invalid data'})



@app.route('/take-attendance', methods=['POST'])
def take_attendance():
    """Take attendance by recognizing a face."""
    image_data = request.form.get('image_data')
    if image_data:
        # Recognize the face
        name, student_id = recognize_face(image_data)
        if name and student_id:
            # Check if attendance is already marked for today
            today = datetime.now().strftime('%Y-%m-%d')
            try:
                with open(ATTENDANCE_CSV, 'r') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames is None:
                        print(f"Error: {ATTENDANCE_CSV} is empty or has no headers.")
                    else:
                        for row in reader:
                            if row['name'] == name and row['id'] == student_id and row['time'].startswith(today):
                                socketio.emit('attendance_message', {
                                    'message': f'This student has already marked attendance today: {name} (ID: {student_id}).'})
                                return jsonify({'status': 'error', 'message': 'Attendance already marked'})
            except FileNotFoundError:
                print(f"Error: {ATTENDANCE_CSV} not found.")
            except Exception as e:
                print(f"Error reading {ATTENDANCE_CSV}: {e}")

            # If no duplicate attendance, proceed to log
            log_attendance(name, student_id)
            return jsonify({'status': 'success', 'message': 'Attendance marked successfully'})
        else:
            socketio.emit('attendance_message', {'message': 'Face not recognized. Please try again.'})
            return jsonify({'status': 'error', 'message': 'Face not recognized'})

    socketio.emit('attendance_message', {'message': 'Invalid data. Please provide an image.'})
    return jsonify({'status': 'error', 'message': 'Invalid data'})


@app.route('/view-attendance', methods=['GET', 'POST'])
def view_attendance():
    """View attendance records for a specific date."""
    selected_date = request.form.get('date') if request.method == 'POST' else datetime.now().strftime('%Y-%m-%d')
    attendance = []
    try:
        with open(ATTENDANCE_CSV, 'r') as f:
            reader = csv.DictReader(f)
            expected_headers = {'name', 'id', 'time'}
            if reader.fieldnames is None or not expected_headers.issubset(reader.fieldnames):
                print(
                    f"Error: {ATTENDANCE_CSV} has incorrect headers. Expected {expected_headers}, but found {reader.fieldnames}")
                # Read the existing data (treat the first row as data if headers are incorrect)
                f.seek(0)
                lines = f.readlines()
                if lines and reader.fieldnames is not None:
                    # Assume the first row is data if it doesn't match the expected headers
                    data = [line.strip().split(',') for line in lines if line.strip()]
                else:
                    data = []

                # Recreate the file with correct headers
                with open(ATTENDANCE_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['name', 'id', 'time'])
                    # Write back the data
                    for row in data:
                        if len(row) == 3:  # Ensure the row has the correct number of columns
                            writer.writerow(row)
                return render_template('view_attendance.html', attendance=attendance, selected_date=selected_date)

            for i, row in enumerate(reader, 1):
                if row['time'].startswith(selected_date):
                    attendance.append({
                        's_no': i,
                        'name': row['name'],
                        'id': row['id'],
                        'time': row['time']
                    })
    except FileNotFoundError:
        print(f"Error: {ATTENDANCE_CSV} not found.")
        # Create the file with correct headers
        with open(ATTENDANCE_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'id', 'time'])
    except Exception as e:
        print(f"Error reading {ATTENDANCE_CSV}: {e}")

    return render_template('view_attendance.html', attendance=attendance, selected_date=selected_date)


@app.route('/registered-students')
def registered_students():
    """View all registered students."""
    students = get_all_students()
    return render_template('registered_students.html', students=students)


# Run the app with SocketIO
if __name__ == '__main__':
    socketio.run(app, debug=True)