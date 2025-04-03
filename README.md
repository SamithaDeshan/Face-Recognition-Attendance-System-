# SmartAttend: A Face Recognition-Based Attendance System

## Overview

**SmartAttend** is a web-based application designed to automate attendance tracking using face recognition technology. It allows administrators to register students, mark attendance via a webcam, view attendance records, and analyze attendance trends. The system leverages Flask, OpenCV, and SocketIO to provide a seamless and efficient experience with real-time updates and visual feedback.

### Key Features
- **User Authentication**: Sign-up and login for administrators.
- **Student Management**: Add and view registered students.
- **Attendance Management**: Mark attendance using face recognition, view records, and check individual student history.
- **Attendance Trends**: Visualize monthly and weekly attendance trends.
- **Real-Time Updates**: Instant notifications for attendance marking and student additions.
- **Graphical Test Reports**: Unit test results visualized with pie charts or bar charts.

## Technology Stack
- **Backend**: Flask, Python, SocketIO
- **Frontend**: HTML, CSS, JavaScript, Jinja2, Bootstrap, Chart.js
- **Face Recognition**: OpenCV (Haar Cascade Classifier, LBPH Face Recognizer), Pillow, NumPy
- **Data Storage**: CSV files
- **Testing**: pytest, pytest-html, matplotlib, pandas

## Installation

### Prerequisites
- Python 3.11 (recommended due to compatibility with `opencv-contrib-python`)
- A webcam (for attendance marking)
- Git (optional, for cloning the repository)

### Steps
1. **Clone the Repository** (if hosted on GitHub):
   ```bash
   git clone https://github.com/yourusername/smartattend.git
   cd smartattend