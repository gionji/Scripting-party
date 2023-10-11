import os
import argparse
import random
import shutil

def split_dataset(input_folder, ratios):
    # Create train, test, and valid subfolders
    train_folder = os.path.join(input_folder, "train")
    test_folder = os.path.join(input_folder, "test")
    valid_folder = os.path.join(input_folder, "valid")

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)

    # List all PNG files in the input folder
    png_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]

    # Shuffle the list of PNG files
    random.shuffle(png_files)

    # Calculate the number of images for each set based on the ratios
    total_images = len(png_files)
    num_train = int(total_images * ratios[0])
    num_test = int(total_images * ratios[1])
    num_valid = total_images - num_train - num_test

    # Copy images to the train, test, and valid subfolders
    for i, file_name in enumerate(png_files):
        source_path = os.path.join(input_folder, file_name)
        if i < num_train:
            destination_folder = train_folder
        elif i < num_train + num_test:
            destination_folder = test_folder
        else:
            destination_folder = valid_folder

        destination_path = os.path.join(destination_folder, file_name)
        shutil.copy(source_path, destination_path)

    print(f"Split dataset into train: {num_train} images, test: {num_test} images, valid: {num_valid} images")



def main():
    parser = argparse.ArgumentParser(description="Split a dataset into train, test, and valid sets.")
    parser.add_argument("--input", required=True, help="Path to the input folder containing PNG images.")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1], help="Ratios for train, test, and valid sets.")

    args = parser.parse_args()
    input_folder = args.input
    ratios = args.ratios

    split_dataset(input_folder, ratios)




if __name__ == "__main__":
    main()
