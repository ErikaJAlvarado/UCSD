import unittest
from tweeter_disaster import get_severity

class test_get_severity(unittest.TestCase):
    
    def test_values(self):
        # Make sure values errors are raised when a non-numeric value is enter
        self.assertRaises(ValueError,get_severity,"hola")