import unittest
from app import app, ADMIN_EMAIL, ADMIN_PASSWORD

class TestAdmin(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['DEBUG'] = False
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_admin_login(self):
        """Test that admin can log in with hardcoded credentials."""
        print("\n[Test 1] Attempting admin login...")
        response = self.client.post('/login', data={
            'email': ADMIN_EMAIL,
            'password': ADMIN_PASSWORD
        }, follow_redirects=True)
        
        print("[Test 1] Verifying response status is 200 (OK)...")
        self.assertEqual(response.status_code, 200)
        
        print("[Test 1] Checking for 'Admin Command Center' text on dashboard...")
        self.assertIn(b'Admin Command Center', response.data)
        
        print("[Test 1] Checking for 'Dashboard' navigation tab...")
        self.assertIn(b'Dashboard', response.data)
        print("[PASS] Admin login verified.")

    def test_admin_access_denied_for_standard_user(self):
        """Test that a standard user cannot access admin routes."""
        print("\n[Test 2] Attempting to access /admin area without an account...")
        response = self.client.get('/admin', follow_redirects=False)
        
        print("[Test 2] Verifying that the server issues a 302 Redirect...")
        self.assertEqual(response.status_code, 302)
        
        print("[Test 2] Verifying the redirect destination is the /login page...")
        self.assertIn('/login', response.location)
        print("[PASS] Security block verified.")

if __name__ == '__main__':
    unittest.main()
