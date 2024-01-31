import unittest
from FullVision import FRAME_MANIP

frame = "path_to_image"

class TEST_NORMALIZATION(unittest.TESTCASE):
   def JTest1(self):
      ### this test will check to see how the normalize function handles a path to no frame
      self.assertRaises(frame)
