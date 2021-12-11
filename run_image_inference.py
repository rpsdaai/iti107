# Import the required libraries for Object detection inference
import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys, getopt
from pathlib import Path

# Ref: https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so
matplotlib.use('Qt5Agg')

# setting min confidence threshold
MIN_CONF_THRESH=.6

PRE_TRAINED_MODEL = "efficientdet_d0_coco17_tpu-32"
# EXPERIMENT = "\run2" # CHANGE HERE
# EXPERIMENT = "\run3"

# Loading the exported model from saved_model directory
# PATH_TO_SAVED_MODEL ="./exported_models/efficientdet_d0_coco17_tpu-32/run3/saved_model"
# print('Loading model...', end='')
# start_time = time.time()
# # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
# detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
# end_time = time.time()
# elapsed_time = end_time - start_time

# print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA
PATH_TO_LABELS=r'.\data\label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
print ('category_index: ', category_index)

# Image file for inference
# IMAGE_PATH=r'.\data\test\test_001.jpg'

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array of shape (height, width, channels), where channels=3 for RGB to feed into tensorflow graph.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

def main(argv):
   INPUT_TEST_FILE = ''
   OUTPUT_TEST_FILE = ''
   try:
      opts, args = getopt.getopt(argv,"h:i:e:")
   except getopt.GetoptError:
      print ('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
    if opt == '-h':
        print ('test.py -i <inputfile> -e <experiment>')
        sys.exit()
    elif opt == '-i':
        INPUT_TEST_FILE = arg
    elif opt == "-e":
        EXPERIMENT = arg
   print ('Input file is "', INPUT_TEST_FILE)
   # print ('Output file is "', outputfile)
   print ('Experiment: ', EXPERIMENT)
   
   # Video
   basename = Path(INPUT_TEST_FILE).stem
   print (basename)
   OUTFILE = basename.replace('test', 'test_detected') + '.jpg'
   print (OUTFILE)
   
   PATH_TO_SAVED_MODEL ="./exported_models/" + PRE_TRAINED_MODEL + "/" + EXPERIMENT + "/saved_model"
   
   print (PATH_TO_SAVED_MODEL)

   start_time = time.time()
   # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
   detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
   end_time = time.time()
   elapsed_time = end_time - start_time

   print('Model Loading Done! Took {} seconds'.format(elapsed_time))

   IMAGE_PATH=r'./data/test/' + INPUT_TEST_FILE
   image_np = load_image_into_numpy_array(IMAGE_PATH)
   # Running the infernce on the image specified in the  image path
   # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
   input_tensor = tf.convert_to_tensor(image_np)
   # The model expects a batch of images, so add an axis with `tf.newaxis`.
   input_tensor = input_tensor[tf.newaxis, ...]
   detections = detect_fn(input_tensor)

   # All outputs are batches tensors.
   # Convert to numpy arrays, and take index [0] to remove the batch dimension.
   # We're only interested in the first num_detections.
   num_detections = int(detections.pop('num_detections'))
   detections = {key: value[0, :num_detections].numpy()
                 for key, value in detections.items()}
   detections['num_detections'] = num_detections
   # detection_classes should be ints.
   detections['detection_classes'] = detections['detection_classes'].astype(np.int64) # print(detections['detection_classes'])
   image_np_with_detections = image_np.copy()

   viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=MIN_CONF_THRESH,
          agnostic_mode=False)


   # fig = plt.figure(figsize=(6, 6), dpi=300)
   plt.figure()
   plt.imshow(image_np_with_detections)
   print('Done')
   # Ref: https://stackoverflow.com/questions/9012487/matplotlib-pyplot-savefig-outputs-blank-image
   plt.savefig(r'./data/test/' + OUTFILE)
   plt.show()

# To run: python run_image_inference.py -i test_006.jpg -e run3
if __name__ == "__main__":
   main(sys.argv[1:])