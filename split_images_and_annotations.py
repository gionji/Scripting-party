import os
import argparse
import random
import shutil
import json
import yaml




def generate_yaml_file(yaml_out_path, yolo_dataset_path, class_names):

    train_path = os.path.join(yolo_dataset_path, 'train')  
    valid_path = os.path.join(yolo_dataset_path, 'valid')  
    test_path = os.path.join(yolo_dataset_path, 'test')  

    yaml_data = {
        "train": train_path,
        "val": valid_path,
        "test": test_path,
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_filename = os.path.join(yaml_out_path, "data.yaml"a)
    
    with open(yaml_filename, "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)
    
    print(f"Generated {yaml_filename}")


def find_yolo_annotation_file(labels_folder, image_filename):
    yolo_annotation_filename = image_filename.replace(".png", ".txt")
    yolo_annotation_path = os.path.join(labels_folder, yolo_annotation_filename)
    return yolo_annotation_path if os.path.exists(yolo_annotation_path) else None


def split_dataset(images_folder, labels_folder, ratios):
    # Create train, test, and valid subfolders
    output_folder = './yolo_dataset'
    train_folder = os.path.join(output_folder, "train")
    test_folder = os.path.join(output_folder, "test")
    valid_folder = os.path.join(output_folder, "valid")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)

    # List all PNG files in the input folder
    png_files = [f for f in os.listdir(images_folder) if f.endswith(".png")]

    # Shuffle the list of PNG files
    random.shuffle(png_files)

    # Calculate the number of images for each set based on the ratios
    total_images = len(png_files)
    num_train = int(total_images * ratios[0])
    num_test = int(total_images * ratios[1])
    num_valid = total_images - num_train - num_test

    # Copy images and YOLO labels to the train, test, and valid subfolders
    for i, file_name in enumerate(png_files):
        source_image_path = os.path.join(images_folder, file_name)
        yolo_annotation_path = find_yolo_annotation_file(labels_folder, file_name)

        if i < num_train:
            destination_image_folder = train_folder
            destination_label_folder = train_folder
        elif i < num_train + num_test:this
            destination_image_folder = test_folder
            destination_label_folder = test_folder
        else:
            destination_image_folder = valid_folder
            destination_label_folder = valid_folder

        destination_image_path = os.path.join(destination_image_folder, file_name)
        if yolo_annotation_path:
            destination_label_path = os.path.join(destination_label_folder, file_name.replace(".png", ".txt"))
            shutil.copy(source_image_path, destination_image_path)
            shutil.copy(yolo_annotation_path, destination_label_path)
        else:
            # Handle the case where YOLO annotation file is not found
            print(f"Warning: YOLO annotation not found for {file_name}, skipping.")

    print(f"Split dataset into train: {num_train} images, test: {num_test} images, valid: {num_valid} images")
    


def create_yolo_labels_folder(input_folder):
    yolo_labels_folder = os.path.join(input_folder, "yolo_labels")
    if not os.path.exists(yolo_labels_folder):
        os.makedirs(yolo_labels_folder)
    return yolo_labels_folder

def parse_annotations(input_folder):
    class_counts = {}

    # List annotation files in the input folder
    annotation_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

    for annotation_file in annotation_files:
        annotation_file_path = os.path.join(input_folder, annotation_file)

        with open(annotation_file_path, "r") as f:
            annotation_data = json.load(f)

        vehicle_classes = annotation_data.get("vehicle_class", [])

        for class_id in vehicle_classes:
            if class_id not in class_counts:
                class_counts[class_id] = 0
            class_counts[class_id] += 1

        yolo_label = convert_to_yolo_format(annotation_data)

        # Save the YOLO format annotation to the yolo_labels folder
        yolo_label_path = os.path.join(create_yolo_labels_folder(input_folder), annotation_file)
        with open(yolo_label_path, "w") as yolo_file:
            yolo_file.write(yolo_label)

    return class_counts

def convert_to_yolo_format(annotation_data, image_size=(1920,1080)):
    yolo_label = ""
    bboxes = annotation_data.get("bboxes", [])
    vehicle_classes = annotation_data.get("vehicle_class", [])
    image_width  = image_size[0]
    image_height = image_size[1]

    for bbox, class_id in zip(bboxes, vehicle_classes):
        x1 = bbox[0][0]
        y1 = bbox[0][1]
        x2 = bbox[1][0]
        y2 = bbox[1][1]

        # Calculate YOLO format coordinates (normalized)
        x_center = (x1 + x2) / (2 * image_width)
        y_center = (y1 + y2) / (2 * image_height)
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        yolo_label += f"{class_id} {x_center} {y_center} {width} {height}\n"

    return yolo_label


def print_class_counts(class_counts):
    print("Classes present in annotation files and their occurrences:")
    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} occurrences")
        

def main():
    parser = argparse.ArgumentParser(description="Split a dataset into train, test, and valid sets.")
    parser.add_argument("--input", required=True, help="Path to the input folder containing PNG images.")
    parser.add_argument("--labels", required=True, help="Path to the labels folder containing Viser look-alike annotation files.")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1], help="Ratios for train, test, and valid sets.")

    args = parser.parse_args()
    images_folder = args.input
    labels_folder = args.labels
    ratios = args.ratios
    
    # convert to yolo format
    class_counts = parse_annotations(labels_folder)

    # split the dataset
    yolo_labels_folder = os.path.join(labels_folder, "yolo_labels")
    split_dataset(images_folder, yolo_labels_folder, ratios)
    
    class_names = ['0', '1', '2', '3', 'multirotor', 'fixedwing', 'airliner', 'bird']    
    output_folder = os.path.abspath('./yolo_dataset')
    generate_yaml_file(output_folder, output_folder, class_names)
    

if __name__ == "__main__":
    main()
