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
from sklearn.model_selection import train_test_split
import shutil

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

def split_data(data_dir, test_size=0.2):
    """
    Notes:
    Randomization: train_test_split function shuffles the data by default 
                   before splitting (unless the shuffle parameter is explicitly 
                   set to False)
    Stratification: The stratify=labels parameter ensures that both training 
                    and testing datasets have approximately the same percentage 
                    of samples of each target class as the complete set
    """
    # Ensure that the correct directories are referenced
    deepfake_dir = os.path.join(data_dir, 'deepfake_cropped')
    original_dir = os.path.join(data_dir, 'original_cropped')

    # Validate the existence of directories
    if not os.path.exists(deepfake_dir) or not os.path.exists(original_dir):
        print(f"Required directories not found. Ensure both 'deepfake_cropped' and 'original_cropped' exist under {data_dir}.")
        return

    # Collect all deepfake and original images
    deepfake_images = [os.path.join(deepfake_dir, f) for f in os.listdir(deepfake_dir) if f.endswith('.jpg')]
    original_images = [os.path.join(original_dir, f) for f in os.listdir(original_dir) if f.endswith('.jpg')]


    # Create labels: 1 for deepfake, 0 for original
    labels = [1] * len(deepfake_images) + [0] * len(original_images)
    all_images = deepfake_images + original_images

    # Split data into training and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, labels, test_size=test_size, stratify=labels)

    # Create directories for the train/test datasets
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Function to copy files to new training/testing directories
    def copy_files(files, labels, dest_dir):
        for file_path, label in zip(files, labels):
            label_dir = os.path.join(dest_dir, 'deepfake' if label == 1 else 'original')
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy(file_path, label_dir)

    # Copy the split files into their respective directories
    copy_files(train_images, train_labels, train_dir)
    copy_files(test_images, test_labels, test_dir)
    print(f"Training and testing data prepared: {len(train_images)} training and {len(test_images)} testing images.")


if __name__ == '__main__':
    """
    Here's how to run:
    python3 preprocess.py --video_path /path/to/video_directory --output_path /path/to/output_directory

    #############   GETTING NON-DEEPFAKE FRAMES   ##############
    Working example for original:
    python3 preprocess.py --video_path ../sample_data/original --output_path ../cropped_data/original_cropped --frame_skip 100

    #############   GETTING DEEPFAKE FRAMES   ##############
    Working example for deepfake:
    python3 preprocess.py --video_path ../sample_data/deepfake --output_path ../cropped_data/deepfake_cropped --frame_skip 100

        Make sure destination folders are made BEFOREHAND

        The skip frame argument tells how many frames to skip between each 'screenshot.' Otherwise, we'd have a jpg for 
        every frame in the video at 60 fps with 30 sec clips, you get the idea

    #############   SPLITTING THE DATA INTO TRAIN/TEST   ##############
    Splitting Test and Train data (Optional):
    After processing both original and deepfake videos, run:
    python3 preprocess.py --data_dir ../cropped_data --split_data
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_path', '-i', type=str, help='Path to the video directory or a single video file')
    parser.add_argument('--output_path', '-o', type=str, help='Path to the output directory where cropped images will be saved')
    parser.add_argument('--frame_skip', type=int, default=5, help='Number of frames to skip between processing')
    parser.add_argument('--data_dir', type=str, help='Directory containing both deepfake_cropped and original_cropped for data splitting')
    parser.add_argument('--split_data', action='store_true', help='Set this flag to split the data into training and testing sets after processing')
    args = parser.parse_args()

    # Process videos if video_path and output_path are provided
    if args.video_path and args.output_path:
        if os.path.isdir(args.video_path):
            videos = [join(args.video_path, video) for video in os.listdir(args.video_path) if video.endswith(('.mp4', '.avi'))]
            if not videos:
                print("No video files found in the directory.")
            else:
                for video in videos:
                    extract_faces(video, args.output_path, frame_skip=args.frame_skip)
        else:
            extract_faces(args.video_path, args.output_path, frame_skip=args.frame_skip)

    # Optionally split the data if the flag is set and data_dir is provided
    if args.split_data and args.data_dir:
        print("Splitting the data into training and testing sets...")
        split_data(args.data_dir, test_size=0.2)
        print("Data splitting complete.")
