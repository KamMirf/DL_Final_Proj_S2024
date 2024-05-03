import os
import shutil
import random

def gather_files(source_dir):
    """ Gather all files from the source directories. """
    file_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def split_data(files, test_size=0.2):
    """ Randomly split data into train and test sets based on the test_size percentage. """
    random.shuffle(files)  # Shuffle here ensures each frame is considered independently
    split_idx = int(len(files) * (1 - test_size))
    return files[:split_idx], files[split_idx:]

def balance_classes(files):
    """ Ensure that there is an equal number of deepfake and original images. """
    deepfake_files = [f for f in files if "deepfake_cropped" in f]
    original_files = [f for f in files if "original_cropped" in f]
    
    min_size = min(len(deepfake_files), len(original_files))
    random.shuffle(deepfake_files)  # Shuffle each category separately to ensure randomness
    random.shuffle(original_files)
    
    balanced_files = deepfake_files[:min_size] + original_files[:min_size]
    random.shuffle(balanced_files)  # Shuffle combined files again for good measure
    return balanced_files, len(deepfake_files), len(original_files)

def create_directories(base_path):
    """ Create the main output directory and subdirectories if they don't exist. """
    os.makedirs(base_path, exist_ok=True)
    print(f"Created main directory: {base_path}")

    train_dir = os.path.join(base_path, 'train')
    test_dir = os.path.join(base_path, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Created subdirectories: {train_dir}, {test_dir}")
    return train_dir, test_dir

def main(cropped_data_dir, cropped_data_celeb_dir, output_dir, test_size=0.2):
    """ Main function to handle the entire process of dataset combination and balancing. """
    # Create output directories
    train_dir, test_dir = create_directories(output_dir)

    # Gather files from both datasets
    cropped_data_files = gather_files(cropped_data_dir)
    cropped_data_celeb_files = gather_files(cropped_data_celeb_dir)

    # Combine and balance the datasets
    combined_files = cropped_data_files + cropped_data_celeb_files
    balanced_files, df_count, orig_count = balance_classes(combined_files)

    print(f"Total deepfake files: {df_count}, total original files: {orig_count}")
    print(f"Balanced to {min(df_count, orig_count)} of each for total of {min(df_count, orig_count) * 2} files.")

    # Split into train and test
    train_files, test_files = split_data(balanced_files, test_size=test_size)
    
    # Ensure both train and test are balanced
    train_files, train_df_count, train_orig_count = balance_classes(train_files)
    test_files, test_df_count, test_orig_count = balance_classes(test_files)

    print(f"Train distribution: {train_df_count} deepfake and {train_orig_count} original.")
    print(f"Test distribution: {test_df_count} deepfake and {test_orig_count} original.")

    # Copy files to their new locations
    for file_path in train_files:
        shutil.copy(file_path, train_dir)
    for file_path in test_files:
        shutil.copy(file_path, test_dir)

    print(f"Training data prepared with {len(train_files)} files.")
    print(f"Testing data prepared with {len(test_files)} files.")

if __name__ == '__main__':
    cropped_data_dir = '../cropped_data'
    cropped_data_celeb_dir = '../cropped_data_Celeb'
    output_dir = '../combined_data'
    main(cropped_data_dir, cropped_data_celeb_dir, output_dir)
