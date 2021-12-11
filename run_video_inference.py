import os
import pathlib
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from six import BytesIO
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import cv2
import sys, getopt
from pathlib import Path

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

PRE_TRAINED_MODEL = "efficientdet_d0_coco17_tpu-32"
# EXPERIMENT = "\run2" # CHANGE HERE
# EXPERIMENT = "\run3"

# # Loading the exported model from saved_model directory
# PATH_TO_SAVED_MODEL ="./exported_models/efficientdet_d0_coco17_tpu-32/run3/saved_model"
# print('Loading model...', end='')
# start_time = time.time()
# # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
# detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

PATH_TO_LABELS=r'.\data\label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
print ('category_index: ', category_index)

def write_video(video_in_filepath, video_out_filepath, detection_model):
    if not os.path.exists(video_in_filepath):
        print('video filepath not valid')
    
    video_reader = cv2.VideoCapture(video_in_filepath)
    
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    
    video_writer = cv2.VideoWriter(video_out_filepath,
                               cv2.VideoWriter_fourcc(*'mp4v'), 
                               fps, 
                               (frame_w, frame_h))

    for i in tqdm(range(nb_frames)):
        ret, image_np = video_reader.read()
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
        results = detection_model(input_tensor)
        viz_utils.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  results['detection_boxes'][0].numpy(),
                  (results['detection_classes'][0].numpy()+1).astype(int),
                  results['detection_scores'][0].numpy(),
                  category_index,
                  use_normalized_coordinates=True,
                  max_boxes_to_draw=200,
                  min_score_thresh=.50,
                  agnostic_mode=False,
                  line_thickness=2)

        video_writer.write(np.uint8(image_np))
                
    # Release camera and close windows
    video_reader.release()
    video_writer.release() 
    cv2.destroyAllWindows() 
    cv2.waitKey(1)

# video_in = "./data/test/lion_cheetah_test_video.mp4"
# video_out = "./data/test/lion_cheetah_detected_test_video.mp4"

# video_in = "./data/test/lion_test_video.mp4"
# video_out = "./data/test/lion_test_detected_test_video.mp4"

# video_in = "./data/test/tiger_test_video.mp4"
# video_out = "./data/test/tiger_test_detected_test_video.mp4"

# video_in = "./data/test/cheetah_test_video.mp4"
# video_out = "./data/test/cheetah_test_detected_test_video.mp4"

def main(argv):
   INPUT_TEST_FILE = ''
   OUTPUT_TEST_FILE = ''
   INPUT_IMAGE_FILE = ''
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
   OUTFILE = basename.replace('test', 'test_detected') + '.mp4'
   print (OUTFILE)
   
   PATH_TO_SAVED_MODEL ="./exported_models/" + PRE_TRAINED_MODEL + "/" + EXPERIMENT + "/saved_model"
   
   print (PATH_TO_SAVED_MODEL)

   start_time = time.time()
   # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
   detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
   end_time = time.time()
   elapsed_time = end_time - start_time

   print('Model Loading Done! Took {} seconds'.format(elapsed_time))

   INPUT_IMAGE_FILE = r'./data/test/' + INPUT_TEST_FILE
   OUTPUT_TEST_FILE = r'./data/test/' + OUTFILE
   write_video(INPUT_IMAGE_FILE, OUTPUT_TEST_FILE, detect_fn)
   
# To run: python run_image_inference.py -i test_006.jpg -e run3
if __name__ == "__main__":
   main(sys.argv[1:])