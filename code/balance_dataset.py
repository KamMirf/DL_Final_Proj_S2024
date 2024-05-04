import os
import shutil
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split

def collect_source_files(directory):
    """ Collects all files and groups them by video source prefix. """
    files_by_source = defaultdict(list)
    for root, dirs, files in os.walk(directory):
        for file in files:
            prefix = file.split('_frame')[0]
            files_by_source[prefix].append(os.path.join(root, file))
    return files_by_source

def create_train_test_split(files_by_source, test_size=0.2):
    """ Splits the files into train and test sets, keeping the same source in one set. """
    sources = list(files_by_source.keys())
    train_sources, test_sources = train_test_split(sources, test_size=test_size, random_state=42)
    
    train_files = [file for src in train_sources for file in files_by_source[src]]
    test_files = [file for src in test_sources for file in files_by_source[src]]
    return train_files, test_files

def copy_files(files, target_dir):
    """ Copies files to the specified directory, flattening the directory structure. """
    for file_path in files:
        file_name = os.path.basename(file_path)
        new_path = os.path.join(target_dir, file_name)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(file_path, new_path)

def process_directory(source_dir, output_dir):
    """ Processes a single directory of images. """
    categories = {'deepfake_cropped': 'deepfake', 'original_cropped': 'original'}

    for category, label in categories.items():
        path = os.path.join(source_dir, category)
        files_by_source = collect_source_files(path)
        train_files, test_files = create_train_test_split(files_by_source)
        
        # Define target directories
        train_target = os.path.join(output_dir, 'train', label)
        test_target = os.path.join(output_dir, 'test', label)
        
        # Copy files to respective directories
        copy_files(train_files, train_target)
        copy_files(test_files, test_target)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into train/test sets for deepfake detection.')
    parser.add_argument('--source_dir', type=str, help='Directory containing the dataset.')
    parser.add_argument('--output_dir', type=str, default='../combined_data', help='Output directory for combined data.')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the directory
    process_directory(args.source_dir, args.output_dir)
    
#python3 balance_dataset.py --source_dir ../cropped_data --output_dir ../combined_data
