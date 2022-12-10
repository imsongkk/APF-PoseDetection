# Information about server
ip = "43.200.230.144"
port = 5000
#ip = "127.0.0.1"
#port = 2119

# Import modules
print("Importing modules... ", end='')
import tensorflow as tf
import numpy as np
import cv2
import socket
import time
print("Done!")

# Load MoveNet Lightning
print("Loading MoveNet Lightning... ", end='')
model = tf.saved_model.load("./movenet_singlepose_lightning_4").signatures['serving_default']
input_size = 192
print("Done!")

# Dictionary that maps from joint names to keypoint indices
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Point Connections
KEYPOINT_CONN = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                 (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

# Run detection on 'input_image' and return 'keypoints_with_scores'
def movenet(model, input_image):
    """
    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.
    
    Returns:
      A [17, 3] float numpy array representing the predicted keypoint
      coordinates and scores. Each row is [y, x, score] (all values normalized to be 0 ~ 1)
    """
    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores_wrapped = outputs['output_0'].numpy()
    return keypoints_with_scores_wrapped[0][0]

# Correctly modify 'keypoints_with_scores' for a convenient use
def correctKeypoints(keypoints_with_scores):
    '''
    Each row of keypoints_with_scores is in [y, x, score] form,
    where x and y are flipped (high x == left in image, high y == bottom in image).
    This function converts each row to [x, y, score] form,
    where x and y are correctly modified (high x == right in image, high y == top in image).
    The nose is at x = 0, -1 <= x <= 1.
    '''
    converted = []
    xBias = keypoints_with_scores[0][1] # x of nose
    for row in keypoints_with_scores:
        y, x, score = row
        y = 1 - y
        x = 1 - x - xBias
        converted.append([x, y, score])
    return np.array(converted)

# Convert list into formatted string
def formatList(myList):
    data_str = ""
    for i in range(len(myList)):
        data = myList[i]
        for j in range(len(data)):
            data_str += str(data[j])
            if (j != len(data) - 1): data_str += ","
        if (i != len(myList) -1): data_str += "/"
    data_str += "!"
    return data_str

# Initialize Camera
print("Waiting for Camera... ", end='')
capture = cv2.VideoCapture(0)
print("Done!")

while True:
    try:
        print("Connecting to ", ip, ":", port, "... ", sep='', end='')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((ip, port))
        print("Done!")
        break
    except KeyboardInterrupt: exit()
    except ConnectionRefusedError: print("Failed!")

prev_time = time.time()
frame_count = 0
while True:
    try:        
        # Capture image from camera
        _, frame = capture.read() # BGR
        cv2.imshow("Input image", cv2.flip(frame, 1))
        cv2.waitKey(1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB
        image = tf.convert_to_tensor(frame, dtype=tf.float32)

        # Resize and pad the image to keep the aspect ratio and fit the expected size
        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

        # Run model inference
        keypoints_with_scores = movenet(model, input_image)
        keypoints_with_scores = correctKeypoints(keypoints_with_scores)
        frame_count += 1
        current_time = time.time()
        if (current_time - prev_time > 1):
            print(frame_count, "FPS")
            frame_count = 0
            prev_time = current_time

        # Send data to server
        # Note that each keypoint row of 'keypoints_with_scores' is in the form [x, y, score]
        data_str = formatList(keypoints_with_scores.tolist())
        client_socket.sendall(data_str.encode("ascii", "replace"))
    except KeyboardInterrupt: exit()
    except:
        print("Connection Lost!")
        while True:
            try:
                print("Reconnecting to ", ip, ":", port, "... ", sep='', end='')
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((ip, port))
                print("Done!")
                break
            except KeyboardInterrupt: exit()
            except: print("Failed!")
           
client_socket.close()
print('End transmission')
