import sys
import unittest
sys.path.insert(1, '../code')
from tweeter_disaster import get_severity, pre_processing_data, cleaning_data, saving_clean_data, model, prediction

class test_get_severity(unittest.TestCase):
    
    def test_values(self):
        # Make sure values errors are raised when a non-numeric value is enter
        self.assertRaises(ValueError,get_severity,"hola")
        self.assertRaises(ValueError,get_severity,-4)
        self.assertRaises(ValueError,get_severity,-4.35)
    
    def test_results(self):
        self.assertAlmostEqual(get_severity(0.3),0)
        self.assertAlmostEqual(get_severity(0.51),1)
        self.assertAlmostEqual(get_severity(0.76),2)

    def test_file_exist(self):
        self.assertRaises(ValueError,pre_processing_data,None)
        self.assertRaises(ValueError,cleaning_data,None)
        self.assertRaises(ValueError,saving_clean_data,None)
        self.assertRaises(ValueError,model,None)
        self.assertRaises(ValueError,prediction,None,None,None,None)
        
