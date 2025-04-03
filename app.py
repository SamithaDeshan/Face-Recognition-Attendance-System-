import os
from cProfile import label

import cv2
import csv
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit
import base64
from io import BytesIO
from PIL import Image
from calendar import monthrange
from datetime import datetime, timedelta

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

# Add this near the top of app.py, with other imports
import hashlib

# Add this with other CSV file paths
USERS_CSV = 'users.csv'

# Initialize users.csv if it doesn't exist
if not os.path.exists(USERS_CSV):
    with open(USERS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['first_name', 'last_name', 'email', 'password'])

# Helper function to hash passwords (simple MD5 for demo purposes; use bcrypt in production)
def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

# Helper function to check if a user exists
def user_exists(email):
    try:
        with open(USERS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['email'] == email:
                    return True
    except FileNotFoundError:
        print(f"Error: {USERS_CSV} not found.")
    except Exception as e:
        print(f"Error reading {USERS_CSV}: {e}")
    return False

# Helper function to verify user credentials
def verify_user(email, password):
    try:
        with open(USERS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['email'] == email and row['password'] == hash_password(password):
                    return True
    except FileNotFoundError:
        print(f"Error: {USERS_CSV} not found.")
    except Exception as e:
        print(f"Error reading {USERS_CSV}: {e}")
    return False

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

def get_attendance_trends():
    """Calculate monthly attendance trends for the current year."""
    current_year = datetime.now().year
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    attendance_rates = [0] * 12  # Initialize attendance rates for each month

    # Read all attendance records
    all_records = []
    try:
        with open(ATTENDANCE_CSV, 'r') as f:
            reader = csv.DictReader(f)
            expected_headers = {'name', 'id', 'time'}
            if reader.fieldnames is None or not expected_headers.issubset(reader.fieldnames):
                print(f"Error: {ATTENDANCE_CSV} has incorrect headers.")
                return {'labels': labels, 'attendance_rates': attendance_rates}

            for row in reader:
                all_records.append(row)
    except FileNotFoundError:
        print(f"Error: {ATTENDANCE_CSV} not found.")
        return {'labels': labels, 'attendance_rates': attendance_rates}
    except Exception as e:
        print(f"Error reading {ATTENDANCE_CSV}: {e}")
        return {'labels': labels, 'attendance_rates': attendance_rates}

    # Process attendance records for the current year
    for month in range(1, 13):  # 1 to 12 for each month
        days_in_month = monthrange(current_year, month)[1]  # Number of days in the month
        days_with_attendance = set()  # Track unique days with attendance

        for record in all_records:
            try:
                record_date = datetime.strptime(record['time'], '%Y-%m-%d %H:%M:%S')
                if record_date.year == current_year and record_date.month == month:
                    days_with_attendance.add(record_date.day)
            except ValueError:
                continue

        # Calculate attendance rate as the percentage of days with attendance
        if days_in_month > 0:
            attendance_rate = (len(days_with_attendance) / days_in_month) * 100
            attendance_rates[month - 1] = round(attendance_rate, 2)

    return {'labels': labels, 'attendance_rates': attendance_rates}

def get_weekly_attendance_trends():
    """Calculate weekly attendance trends (Monday to Friday) for the current month."""
    current_year = datetime.now().year
    current_month = datetime.now().month
    total_students = len(get_all_students())
    labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    attendance_rates = [0] * 5  # Initialize attendance rates for Monday to Friday
    days_count = [0] * 5  # Count of each weekday in the month
    attendance_counts = [0] * 5  # Total attendance percentage for each weekday

    # Read all attendance records
    all_records = []
    try:
        with open(ATTENDANCE_CSV, 'r') as f:
            reader = csv.DictReader(f)
            expected_headers = {'name', 'id', 'time'}
            if reader.fieldnames is None or not expected_headers.issubset(reader.fieldnames):
                print(f"Error: {ATTENDANCE_CSV} has incorrect headers.")
                return {'labels': labels, 'attendance_rates': attendance_rates}

            for row in reader:
                all_records.append(row)
    except FileNotFoundError:
        print(f"Error: {ATTENDANCE_CSV} not found.")
        return {'labels': labels, 'attendance_rates': attendance_rates}
    except Exception as e:
        print(f"Error reading {ATTENDANCE_CSV}: {e}")
        return {'labels': labels, 'attendance_rates': attendance_rates}

    # Count the number of each weekday in the current month
    days_in_month = monthrange(current_year, current_month)[1]
    for day in range(1, days_in_month + 1):
        date = datetime(current_year, current_month, day)
        weekday = date.weekday()  # 0 = Monday, 4 = Friday
        if 0 <= weekday <= 4:  # Monday to Friday
            days_count[weekday] += 1

    # Process attendance records for the current month
    daily_attendance = {}
    for record in all_records:
        try:
            record_date = datetime.strptime(record['time'], '%Y-%m-%d %H:%M:%S')
            if record_date.year == current_year and record_date.month == current_month:
                date_str = record_date.strftime('%Y-%m-%d')
                if date_str not in daily_attendance:
                    daily_attendance[date_str] = set()
                daily_attendance[date_str].add(record['id'])
        except ValueError:
            continue

    # Calculate attendance rate for each weekday
    for date_str, students in daily_attendance.items():
        date = datetime.strptime(date_str, '%Y-%m-%d')
        weekday = date.weekday()  # 0 = Monday, 4 = Friday
        if 0 <= weekday <= 4:  # Monday to Friday
            if total_students > 0:
                attendance_rate = (len(students) / total_students) * 100
                attendance_counts[weekday] += attendance_rate

    # Average the attendance rates for each weekday
    for i in range(5):  # Monday to Friday
        if days_count[i] > 0:
            attendance_rates[i] = round(attendance_counts[i] / days_count[i], 2)
        else:
            attendance_rates[i] = 0  # If no such weekday in the month, set to 0

    return {'labels': labels, 'attendance_rates': attendance_rates}

# Flask Routes
@app.route('/')
def index():
    """Render the main dashboard."""
    today_attendance = get_today_attendance()
    total_students = len(get_all_students())
    current_month = datetime.now().strftime('%B %Y')  # e.g., "October 2023"
    return render_template('index.html', attendance=today_attendance, total_students=total_students, current_month=current_month)

@app.route('/detect-face', methods=['POST'])
def detect_face():
    """Detect faces in the given image and return their coordinates."""
    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    image_data = data['image_data']
    image_data = image_data.split(',')[1]  # Remove the "data:image/jpeg;base64," part
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
    face_coords = [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} for (x, y, w, h) in faces]

    return jsonify({'faces': face_coords})

@app.route('/attendance-trends')
def attendance_trends():
    """Provide monthly attendance trends data for the chart."""
    trends = get_attendance_trends()
    return jsonify(trends)

@app.route('/weekly-attendance-trends')
def weekly_attendance_trends():
    """Provide weekly attendance trends data for the chart."""
    trends = get_weekly_attendance_trends()
    return jsonify(trends)

@app.route('/take-attendance-page')
def take_attendance_page():
    """Render the Take Attendance page."""
    today_attendance = get_today_attendance()
    current_date = datetime.now().strftime('%Y-%m-%d')
    return render_template('take_attendance.html', attendance=today_attendance, current_date=current_date)

@app.route('/add-student-page')
def add_student_page():
    """Render the Add New User page."""
    return render_template('add_student.html')

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

# Add this route with your other Flask routes
@app.route('/auth', methods=['GET', 'POST'])
def auth():
    form_type = 'signup'  # Default to signup form
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'signup':
            first_name = request.form.get('first_name')
            last_name = request.form.get('last_name')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')

            if password != confirm_password:
                return render_template('auth.html', form_type='signup', error="Passwords do not match.")

            if user_exists(email):
                return render_template('auth.html', form_type='signup', error="User with this email already exists.")

            # Save the new user
            with open(USERS_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([first_name, last_name, email, hash_password(password)])

            # Redirect to dashboard after successful signup
            return redirect(url_for('index'))

        elif action == 'login':
            email = request.form.get('email')
            password = request.form.get('password')

            if verify_user(email, password):
                # Redirect to the dashboard
                return redirect(url_for('index'))
            else:
                return render_template('auth.html', form_type='login', error="Invalid email or password.")

    return render_template('auth.html', form_type=form_type)


@app.route('/check_attendance', methods=['GET', 'POST'])
def check_attendance():
    """Check attendance for a student within a specified time period."""
    students = get_all_students()
    student_attendance_records = None
    selected_student = None
    overall_percentage = 0
    attended_days = 0
    total_weekdays = 0
    weekday_percentages = [0, 0, 0, 0, 0]  # Monday to Friday

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'search_student':
            student_id = request.form.get('student_id')
            start_date_str = request.form.get('start_date')
            end_date_str = request.form.get('end_date')

            if student_id and start_date_str and end_date_str:
                try:
                    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                    if start_date > end_date:
                        start_date, end_date = end_date, start_date  # Swap if start_date is after end_date
                except ValueError:
                    return render_template('check_attendance.html', students=students)

                # Get all attendance records
                all_records = []
                try:
                    with open(ATTENDANCE_CSV, 'r') as f:
                        reader = csv.DictReader(f)
                        expected_headers = {'name', 'id', 'time'}
                        if reader.fieldnames is None or not expected_headers.issubset(reader.fieldnames):
                            print(f"Error: {ATTENDANCE_CSV} has incorrect headers.")
                            return render_template('check_attendance.html', students=students)

                        for row in reader:
                            all_records.append(row)
                except FileNotFoundError:
                    print(f"Error: {ATTENDANCE_CSV} not found.")
                except Exception as e:
                    print(f"Error reading {ATTENDANCE_CSV}: {e}")

                # Filter records for the selected student within the date range
                student_attendance_records = []
                for record in all_records:
                    if record['id'] == student_id:
                        try:
                            record_date = datetime.strptime(record['time'], '%Y-%m-%d %H:%M:%S')
                            if start_date <= record_date <= end_date:
                                date_str, time_str = record['time'].split(' ')
                                student_attendance_records.append({
                                    'name': record['name'],
                                    'id': record['id'],
                                    'date': date_str,
                                    'time': time_str
                                })
                        except ValueError:
                            continue

                # Find the selected student
                selected_student = next((s for s in students if s['id'] == student_id), None)

                # Calculate total weekdays in the period (Monday to Friday)
                current_date = start_date
                while current_date <= end_date:
                    if current_date.weekday() <= 4:  # Monday to Friday
                        total_weekdays += 1
                    current_date += timedelta(days=1)

                # Calculate attended days and weekday percentages
                attended_days_set = set()  # To avoid counting multiple records on the same day
                weekday_counts = [0] * 5  # Number of times attended on each weekday
                weekday_totals = [0] * 5  # Total number of each weekday in the period

                # Recalculate total weekdays for each day of the week
                current_date = start_date
                while current_date <= end_date:
                    weekday = current_date.weekday()
                    if weekday <= 4:  # Monday to Friday
                        weekday_totals[weekday] += 1
                    current_date += timedelta(days=1)

                for record in student_attendance_records:
                    record_date = datetime.strptime(record['date'], '%Y-%m-%d')
                    if record_date.weekday() <= 4:  # Only count weekdays
                        date_str = record['date']
                        if date_str not in attended_days_set:
                            attended_days_set.add(date_str)
                            weekday = record_date.weekday()
                            weekday_counts[weekday] += 1

                attended_days = len(attended_days_set)
                overall_percentage = round((attended_days / total_weekdays * 100) if total_weekdays > 0 else 0, 2)

                # Calculate weekday attendance percentages
                for i in range(5):  # Monday to Friday
                    if weekday_totals[i] > 0:
                        weekday_percentages[i] = round((weekday_counts[i] / weekday_totals[i]) * 100, 2)
                    else:
                        weekday_percentages[i] = 0

    return render_template('check_attendance.html',
                          students=students,
                          student_attendance_records=student_attendance_records,
                          selected_student=selected_student,
                          overall_percentage=overall_percentage,
                          attended_days=attended_days,
                          total_weekdays=total_weekdays,
                          weekday_percentages=weekday_percentages)



# Run the app with SocketIO
if __name__ == '__main__':
    socketio.run(app, debug=True)