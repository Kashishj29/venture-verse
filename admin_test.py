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
        response = self.client.post('/login', data={
            'email': ADMIN_EMAIL,
            'password': ADMIN_PASSWORD
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Admin Command Center', response.data)
        self.assertIn(b'Dashboard', response.data)

    def test_admin_access_denied_for_standard_user(self):
        """Test that a standard user cannot access admin routes."""
        # Note: We'd need a real user in the test DB or a mock to test standard login.
        # But we can test that an unauthenticated user is redirected.
        response = self.client.get('/admin', follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.location)

if __name__ == '__main__':
    unittest.main()
