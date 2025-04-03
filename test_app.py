import unittest
import os
import csv
from app import app, user_exists, verify_user, student_exists, hash_password, sanitize_filename, USERS_CSV, STUDENTS_CSV

class TestSmartAttend(unittest.TestCase):
    def setUp(self):
        # Set up a test Flask app
        app.config['TESTING'] = True
        self.app = app.test_client()

        # Create temporary users.csv for testing
        with open(USERS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['first_name', 'last_name', 'email', 'password'])
            writer.writerow(['John', 'Doe', 'john@example.com', hash_password('password123')])

        # Create temporary students.csv for testing
        with open(STUDENTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'id', 'department', 'year', 'image_path'])
            writer.writerow(['Jane Doe', 'D/BCE/23/0001', 'BCE', '2023', 'uploads/D_BCE_23_0001_Jane_Doe.jpg'])

    def tearDown(self):
        # Clean up test files
        if os.path.exists(USERS_CSV):
            os.remove(USERS_CSV)
        if os.path.exists(STUDENTS_CSV):
            os.remove(STUDENTS_CSV)

    def test_user_exists_existing(self):
        # Test for an existing user
        self.assertTrue(user_exists('john@example.com'))

    def test_user_exists_non_existing(self):
        # Test for a non-existent user
        self.assertFalse(user_exists('notfound@example.com'))

    def test_verify_user_valid(self):
        # Test with valid credentials
        self.assertTrue(verify_user('john@example.com', 'password123'))

    def test_verify_user_invalid_password(self):
        # Test with invalid password
        self.assertFalse(verify_user('john@example.com', 'wrongpassword'))

    def test_verify_user_invalid_email(self):
        # Test with invalid email
        self.assertFalse(verify_user('wrong@example.com', 'password123'))

    def test_student_exists_existing(self):
        # Test for an existing student
        self.assertTrue(student_exists('D/BCE/23/0001'))

    def test_student_exists_non_existing(self):
        # Test for a non-existent student
        self.assertFalse(student_exists('D/BCE/23/9999'))

    def test_hash_password(self):
        # Test password hashing
        password = "testpass"
        hashed = hash_password(password)
        self.assertEqual(hashed, hash_password(password))  # Same password should produce same hash
        self.assertNotEqual(hashed, password)  # Hash should not be the same as plaintext

    def test_sanitize_filename(self):
        # Test filename sanitization
        filename = "test<file>:name?.jpg"
        sanitized = sanitize_filename(filename)
        self.assertEqual(sanitized, "test_file__name_.jpg")

    def test_signup_and_login(self):
        # Test sign-up
        signup_data = {
            'action': 'signup',
            'first_name': 'Alice',
            'last_name': 'Smith',
            'email': 'alice@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        }
        response = self.app.post('/auth', data=signup_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Current Month', response.data)  # Check if redirected to dashboard

        # Test login
        login_data = {
            'action': 'login',
            'email': 'alice@example.com',
            'password': 'password123'
        }
        response = self.app.post('/auth', data=login_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Current Month', response.data)  # Check if redirected to dashboard

if __name__ == '__main__':
    unittest.main()