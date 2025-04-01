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
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CSV file paths
STUDENTS_CSV = 'students.csv'
ATTENDANCE_CSV = 'attendance.csv'

# Initialize CSV files if they don't exist
if not os.path.exists(STUDENTS_CSV):
    with open(STUDENTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'id', 'department', 'year', 'image_path'])

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
            expected_headers = {'name', 'id', 'department', 'year', 'image_path'}
            if not expected_headers.issubset(reader.fieldnames):
                print(f"Error: {STUDENTS_CSV} has incorrect headers. Expected {expected_headers}, but found {reader.fieldnames}")
                return label_dict

            for row in reader:
                img_path = row['image_path']
                if not os.path.exists(img_path):
                    print(f"Error: Image file {img_path} not found.")
                    continue

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Error: Failed to load image {img_path}.")
                    continue

                faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                if len(faces) != 1:
                    print(f"Error: No face or multiple faces detected in {img_path}.")
                    continue

                (x, y, w, h) = faces[0]
                face = img[y:y + h, x:x + w]

                label_dict[label_counter] = {'name': row['name'], 'id': row['id']}
                images.append(face)
                labels.append(label_counter)
                label_counter += 1
    except FileNotFoundError:
        print(f"Error: {STUDENTS_CSV} not found.")
    except Exception as e:
        print(f"Error reading {STUDENTS_CSV}: {e}")

    if images:
        recognizer.train(images, np.array(labels))
    return label_dict

def sanitize_filename(filename):
    """Sanitize the filename by replacing invalid characters."""
    invalid_chars = '<>:"/\\|?* '
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def save_student(name, student_id, department, year, image_data):
    """Save a new student to students.csv and store their image."""
    safe_name = sanitize_filename(name)
    safe_student_id = sanitize_filename(student_id)

    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    filename = f"{safe_student_id}_{safe_name}.jpg"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    image.save(image_path)

    file_exists = os.path.exists(STUDENTS_CSV)
    if not file_exists:
        with open(STUDENTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'id', 'department', 'year', 'image_path'])

    with open(STUDENTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, student_id, department, year, image_path])

def recognize_face(image_data):
    """Recognize a face in the given image."""
    label_dict = train_recognizer()
    if not label_dict:
        return None, None

    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) != 1:
        return None, None

    (x, y, w, h) = faces[0]
    face = img_gray[y:y+h, x:x+w]

    label, confidence = recognizer.predict(face)
    if confidence < 100:
        return label_dict[label]['name'], label_dict[label]['id']
    return None, None

def log_attendance(name, student_id):
    """Log attendance to attendance.csv and emit update."""
    today = datetime.now().strftime('%Y-%m-%d')

    try:
        with open(ATTENDANCE_CSV, 'r') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                print(f"Error: {ATTENDANCE_CSV} is empty or has no headers.")
            else:
                for row in reader:
                    if row['name'] == name and row['id'] == student_id and row['time'].startswith(today):
                        socketio.emit('attendance_message',
                                      {'message': f'Attendance already marked for {name} (ID: {student_id}) today.'})
                        return
    except FileNotFoundError:
        print(f"Error: {ATTENDANCE_CSV} not found.")
    except Exception as e:
        print(f"Error reading {ATTENDANCE_CSV}: {e}")

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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

    attendance = []
    try:
        with open(ATTENDANCE_CSV, 'r') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                print(f"Error: {ATTENDANCE_CSV} is empty or has no headers.")
                return
            expected_headers = {'name', 'id', 'time'}
            if not expected_headers.issubset(reader.fieldnames):
                print(f"Error: {ATTENDANCE_CSV} has incorrect headers.")
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
                print(f"Error: {ATTENDANCE_CSV} has incorrect headers.")
                f.seek(0)
                lines = f.readlines()
                if lines and reader.fieldnames is not None:
                    data = [line.strip().split(',') for line in lines if line.strip()]
                else:
                    data = []
                with open(ATTENDANCE_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['name', 'id', 'time'])
                    for row in data:
                        if len(row) == 3:
                            writer.writerow(row)
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
                print(f"Error: {ATTENDANCE_CSV} has incorrect headers.")
                f.seek(0)
                lines = f.readlines()
                if lines and reader.fieldnames is not None:
                    data = [line.strip().split(',') for line in lines if line.strip()]
                else:
                    data = []
                with open(ATTENDANCE_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['name', 'id', 'time'])
                    for row in data:
                        if len(row) == 3:
                            writer.writerow(row)
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
            expected_headers = {'name', 'id', 'department', 'year', 'image_path'}
            if reader.fieldnames is None or not expected_headers.issubset(reader.fieldnames):
                print(f"Error: {STUDENTS_CSV} has incorrect headers.")
                f.seek(0)
                lines = f.readlines()
                if lines and reader.fieldnames is not None:
                    data = [line.strip().split(',') for line in lines if line.strip()]
                else:
                    data = []
                with open(STUDENTS_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['name', 'id', 'department', 'year', 'image_path'])
                    for row in data:
                        if len(row) == 5:
                            writer.writerow(row)
                return students

            for i, row in enumerate(reader, 1):
                students.append({
                    's_no': i,
                    'name': row['name'],
                    'id': row['id'],
                    'department': row['department'],
                    'year': row['year']
                })
    except FileNotFoundError:
        print(f"Error: {STUDENTS_CSV} not found.")
        with open(STUDENTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'id', 'department', 'year', 'image_path'])
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
    student_id = request.form.get('id')  # Full ID constructed by frontend (e.g., D/BCE/23/0002)
    department = request.form.get('department')
    year = request.form.get('year')
    image_data = request.form.get('image_data')

    if name and student_id and department and year and image_data:
        # Validate student_id format (optional, since frontend constructs it)
        if not student_id.startswith('D/') or len(student_id.split('/')) != 4 or not student_id.split('/')[-1].isdigit():
            socketio.emit('student_message', {'message': 'Invalid student ID format. Use D/<Degree>/<Year>/<4-digit number>.'})
            return jsonify({'status': 'error', 'message': 'Invalid ID format'})

        if student_exists(student_id):
            socketio.emit('student_message', {'message': f'This student is already registered with ID {student_id}.'})
            return jsonify({'status': 'error', 'message': 'Student already exists'})

        existing_name, existing_id = recognize_face(image_data)
        if existing_name and existing_id:
            socketio.emit('student_message',
                          {'message': f'This face is already registered for {existing_name} (ID: {existing_id}).'})
            return jsonify({'status': 'error', 'message': 'Face already registered'})

        save_student(name, student_id, department, year, image_data)
        total_students = len(get_all_students())
        socketio.emit('total_students_update', {'total_students': total_students})
        socketio.emit('student_message', {'message': f'Student {name} (ID: {student_id}) added successfully.'})
        return jsonify({'status': 'success', 'message': 'Student added successfully'})

    socketio.emit('student_message', {'message': 'Invalid data. Please provide all required fields.'})
    return jsonify({'status': 'error', 'message': 'Invalid data'})

@app.route('/take-attendance', methods=['POST'])
def take_attendance():
    """Take attendance by recognizing a face."""
    image_data = request.form.get('image_data')
    if image_data:
        name, student_id = recognize_face(image_data)
        if name and student_id:
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
                print(f"Error: {ATTENDANCE_CSV} has incorrect headers.")
                f.seek(0)
                lines = f.readlines()
                if lines and reader.fieldnames is not None:
                    data = [line.strip().split(',') for line in lines if line.strip()]
                else:
                    data = []
                with open(ATTENDANCE_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['name', 'id', 'time'])
                    for row in data:
                        if len(row) == 3:
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

@app.route('/check-attendance', methods=['GET', 'POST'])
def check_attendance():
    """Check attendance by day or by student."""
    students = get_all_students()
    by_day_records = None
    by_student_records = None
    selected_date = None
    selected_student = None

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'by_day':
            selected_date = request.form.get('date')
            if selected_date:
                all_records = []
                try:
                    with open(ATTENDANCE_CSV, 'r') as f:
                        reader = csv.DictReader(f)
                        expected_headers = {'name', 'id', 'time'}
                        if reader.fieldnames is None or not expected_headers.issubset(reader.fieldnames):
                            print(f"Error: {ATTENDANCE_CSV} has incorrect headers.")
                            return render_template('check_attendance.html',
                                                  students=students,
                                                  by_day_records=by_day_records,
                                                  by_student_records=by_student_records,
                                                  selected_date=selected_date,
                                                  selected_student=selected_student)

                        for row in reader:
                            all_records.append(row)
                except FileNotFoundError:
                    print(f"Error: {ATTENDANCE_CSV} not found.")
                except Exception as e:
                    print(f"Error reading {ATTENDANCE_CSV}: {e}")

                by_day_records = []
                for record in all_records:
                    record_date = record['time'].split(' ')[0]
                    if record_date == selected_date:
                        by_day_records.append(record)

        elif action == 'by_student':
            selected_student_id = request.form.get('student_id')
            if selected_student_id:
                all_records = []
                try:
                    with open(ATTENDANCE_CSV, 'r') as f:
                        reader = csv.DictReader(f)
                        expected_headers = {'name', 'id', 'time'}
                        if reader.fieldnames is None or not expected_headers.issubset(reader.fieldnames):
                            print(f"Error: {ATTENDANCE_CSV} has incorrect headers.")
                            return render_template('check_attendance.html',
                                                  students=students,
                                                  by_day_records=by_day_records,
                                                  by_student_records=by_student_records,
                                                  selected_date=selected_date,
                                                  selected_student=selected_student)

                        for row in reader:
                            all_records.append(row)
                except FileNotFoundError:
                    print(f"Error: {ATTENDANCE_CSV} not found.")
                except Exception as e:
                    print(f"Error reading {ATTENDANCE_CSV}: {e}")

                by_student_records = []
                for record in all_records:
                    if record['id'] == selected_student_id:
                        record_date, record_time = record['time'].split(' ')
                        record_with_date = record.copy()
                        record_with_date['date'] = record_date
                        record_with_date['time'] = record_time
                        by_student_records.append(record_with_date)
                selected_student = next((s for s in students if s['id'] == selected_student_id), None)

    return render_template('check_attendance.html',
                          students=students,
                          by_day_records=by_day_records,
                          by_student_records=by_student_records,
                          selected_date=selected_date,
                          selected_student=selected_student)

# Run the app with SocketIO
if __name__ == '__main__':
    socketio.run(app, debug=True)