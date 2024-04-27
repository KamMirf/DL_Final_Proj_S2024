"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python preprocess.py
    -i <folder with video files or path to video file>
    -o <path to output folder, will write one or multiple output videos there>

Original Author: Andreas Rössler
Editied and modified: Jason Pien & Joey Ricciardulli
"""


import os
import argparse
import cv2  #conda install -c conda-forge opencv        Then conda install cv2
import dlib #conda install -c conda-forge dlib
from os.path import join
from tqdm import tqdm
import numpy as np

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()    # Get the left position of the face
    y1 = face.top()     # Get the top position of the face
    x2 = face.right()   # Get the right position of the face
    y2 = face.bottom()  # Get the bottom position of the face
    
    # Calculate the size of the bounding box, scaling it by 'scale'
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    
    # If a minimum size is specified, enforce this minimum size
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    
    # Calculate the center of the bounding box
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Adjust the top-left corner of the bounding box to ensure it's within the frame
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    
    # Ensure the bounding box does not exceed the dimensions of the image
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    
    return x1, y1, size_bb

def extract_faces(video_path, output_path, scale=1.3, minsize=None, frame_skip=5, target_size=(256,256)):
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Extract the video name without extension to use in output filenames
    print(f'Processing: {video_path}')  # Print which video is currently being processed
    
    reader = cv2.VideoCapture(video_path) # Open the video file for processing
    
    if not reader.isOpened():   # Check if the video was opened successfully
        print(f"Error: Cannot open video {video_path}.")
        return

    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) # Get the total number of frames in the video
    
    face_detector = dlib.get_frontal_face_detector()# Initialize the face detector from dlib
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Pretrained facial landmark detector

    
    pbar = tqdm(total=max(1, num_frames // frame_skip), desc="Processing frames")# Setup a progress bar with the total number of frames to process considering skips
    frame_num = 0  # Initialize frame counter

    
    while True:# Process video frame by frame
        ret, image = reader.read()  # Read a frame from the video
        if not ret:  # If no frame is read (end of video or error), break the loop
            break
        if frame_num % frame_skip == 0:  # Process this frame only if it's a 'frame_skip' interval
            height, width = image.shape[:2]  # Get dimensions of the frame
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale for face detection
            faces = face_detector(gray, 1)  # Detect faces in the grayscale image

            
            for i, face in enumerate(faces):# Iterate through each detected face
                landmarks = predictor(gray, face)

                nose_point = landmarks.part(33).x, landmarks.part(33).y
                left_eye = landmarks.part(36).x, landmarks.part(36).y
                right_eye = landmarks.part(45).x, landmarks.part(45).y

                # Calculate the angle to rotate the face to be aligned
                dY = right_eye[1] - left_eye[1]
                dX = right_eye[0] - left_eye[0]
                angle = np.degrees(np.arctan2(dY, dX))

                # Calculate the center of the face
                eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

                # Align the face
                M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
                aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

                h, w = aligned_face.shape[:2]
                center_of_image = (w // 2, h // 2)
                shift_x = center_of_image[0] - nose_point[0]
                shift_y = center_of_image[1] - nose_point[1]

                # Translation matrix
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                centered_face = cv2.warpAffine(aligned_face, M, (w, h))

                # Detect faces in the grayscale image
                gray2 = cv2.cvtColor(centered_face, cv2.COLOR_BGR2GRAY)
                faces2 = face_detector(gray2, 1)

                for face2 in faces2:
                    x, y, size = get_boundingbox(face2, width, height, scale=scale, minsize=minsize)  # Calculate bounding box
                    cropped_face = centered_face[y:y+size, x:x+size]  # Crop the face from the image
                    save_path = join(output_path, f'{video_name}_frame_{frame_num}_face_{i}.jpg')# Construct the path where the cropped image will be saved
                    resized_face = cv2.resize(cropped_face, target_size)
                    cv2.imwrite(save_path, resized_face)  # Save the cropped face image

            pbar.update(1)  # Update the progress bar for every processed frame

        frame_num += 1  # Increment the frame counter

    pbar.close()  # Close the progress bar when all frames are processed
    reader.release()  # Release the video file
    print(f'Finished processing {video_path}')  # Print completion message

if __name__ == '__main__':
    """
    Here's how to run:
    python3 preprocess.py --video_path /path/to/video_directory --output_path /path/to/output_directory

    Working example for original:
    python3 preprocess.py --video_path ../sample_data/original --output_path ../cropped_data/original_cropped --frame_skip 100

    Working example for deepfake:
    python3 preprocess.py --video_path ../sample_data/deepfake --output_path ../cropped_data/deepfake_cropped --frame_skip 100

    Make sure destination folders are made BEFOREHAND

    The skip frame argument tells how many frames to skip between each 'screenshot.' Otherwise we'd have a jpg for 
    every frame in the video. at 60 fps with 30 sec clips, you get the idea
    """
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str, required=True)
    p.add_argument('--output_path', '-o', type=str, required=True)
    p.add_argument('--frame_skip', type=int, default=5, help='Number of frames to skip between processing.')
    args = p.parse_args()

    # Check if the provided video path is a directory or a single file
    if os.path.isdir(args.video_path):
        # List all video files in the directory with specified extensions
        videos = [join(args.video_path, video) for video in os.listdir(args.video_path) if video.endswith(('.mp4', '.avi'))]
        if not videos:
            print("No video files found in the directory.")  # Inform if no videos are found
        else:
            # Process each video found in the directory
            for video in videos:
                extract_faces(video, args.output_path, frame_skip=args.frame_skip)
    else:
        # Process a single video file
        extract_faces(args.video_path, args.output_path, frame_skip=args.frame_skip)

