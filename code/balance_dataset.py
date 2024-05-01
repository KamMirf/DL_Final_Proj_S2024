import os
import shutil
import random

def balance_datasets(base_dir, output_dir):
    sets = ['train', 'test']
    categories = ['deepfake', 'original']

    # Create the main output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created main directory: {output_dir}")

    # Create subdirectories for train and test sets within the output directory
    for set_name in sets:
        set_path = os.path.join(base_dir, set_name)
        output_set_path = os.path.join(output_dir, set_name)
        if not os.path.exists(output_set_path):
            os.makedirs(output_set_path)
            print(f"Created directory for {set_name} set in {output_set_path}")

        category_counts = {}

        # Determine the number of images in each category
        for category in categories:
            category_path = os.path.join(set_path, category)
            images = [name for name in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, name))]
            category_counts[category] = len(images)
            print(f"Found {category_counts[category]} images in {set_name}/{category}.")

        # Determine the minimum number of images to balance the dataset
        min_images = min(category_counts.values())
        print(f"Balancing each category in {set_name} set to {min_images} images each.")

        # Copy the minimum number to new balanced directory
        for category in categories:
            category_path = os.path.join(set_path, category)
            output_category_path = os.path.join(output_set_path, category)
            if not os.path.exists(output_category_path):
                os.makedirs(output_category_path)

            images = [name for name in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, name))]
            random.shuffle(images)  # Shuffle to select random images
            selected_images = images[:min_images]  # Select the first 'min_images' images

            # Copy selected images to new directory
            for image in selected_images:
                shutil.copy(os.path.join(category_path, image), os.path.join(output_category_path, image))

            print(f"Copied {len(selected_images)} images to {output_category_path}.")

        print(f"Completed processing for {set_name} set.\n")

if __name__ == '__main__':
    # Adjust paths relative to the script's location in the 'code' directory
    base_dir = '../cropped_data'  # Adjusted to go up one level from 'code' to root and into 'cropped_data'
    output_dir = '../even_represented_data'  # Adjusted to place the output directory at the root
    balance_datasets(base_dir, output_dir)
