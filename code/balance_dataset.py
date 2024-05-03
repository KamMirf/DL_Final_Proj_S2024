import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

def create_directory(path):
    """ Ensure directory exists """
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def copy_files(source, destination, files):
    """ Copy a list of files from source to destination """
    for file in files:
        shutil.copy2(os.path.join(source, file), os.path.join(destination, file))

def distribute_data(source_dir, train_dir, test_dir, split_ratio):
    """ Distribute files into train and test directories by grouping by source """
    for category in ['deepfake_cropped', 'original_cropped']:
        files = os.listdir(os.path.join(source_dir, category))
        source_dict = defaultdict(list)

        # Group files by source
        for file in files:
            source_prefix = file.split('frame')[0]
            source_dict[source_prefix].append(file)

        # Shuffle the sources
        sources = list(source_dict.keys())
        random.shuffle(sources)

        # Split sources into train and test
        split_point = int(len(sources) * split_ratio)
        train_sources = sources[:split_point]
        test_sources = sources[split_point:]

        # Copy files to train and test directories
        for source in train_sources:
            copy_files(os.path.join(source_dir, category), os.path.join(train_dir, category.replace('_cropped', '')), source_dict[source])
        for source in test_sources:
            copy_files(os.path.join(source_dir, category), os.path.join(test_dir, category.replace('_cropped', '')), source_dict[source])

        print(f"Distributed {len(train_sources)} sources to train for {category.replace('_cropped', '')}")
        print(f"Distributed {len(test_sources)} sources to test for {category.replace('_cropped', '')}")

def main():
    base_dirs = ['../cropped_data', '../cropped_data_Celeb']
    combined_data = '../combined_data'

    # Create combined data directories
    train_dir = create_directory(os.path.join(combined_data, 'train'))
    test_dir = create_directory(os.path.join(combined_data, 'test'))
    create_directory(os.path.join(train_dir, 'deepfake'))
    create_directory(os.path.join(train_dir, 'original'))
    create_directory(os.path.join(test_dir, 'deepfake'))
    create_directory(os.path.join(test_dir, 'original'))

    # Distribute data from each base directory
    for base_dir in base_dirs:
        print(f"Processing directory: {base_dir}")
        distribute_data(base_dir, train_dir, test_dir, 0.8)

if __name__ == "__main__":
    main()
