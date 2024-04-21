# -*- coding: utf-8 -*-

"""
Unit tests for the flask app
We check if each route leads to the correct page
"""

# importations
import unittest
from app import app

class FlaskAppTests(unittest.TestCase):
    """ Class defining unit tests for Flask puzzle application routes."""

    def setUp(self):
        # Create a test client
        self.app = app.test_client()

    def test_index_route(self):
        # Test the index route
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Coloriser vos images', response.data)

    def test_sliding_pieces_route(self):
        # Test the sliding_pieces route
        response = self.app.get('/sliding_pieces')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Sliding Puzzle', response.data)

    def test_free_pieces_route(self):
        # Test the free_pieces route
        response = self.app.get('/free_pieces')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Free Pieces Puzzle', response.data)

if __name__ == '__main__':
    unittest.main()
