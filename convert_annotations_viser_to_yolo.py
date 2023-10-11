import os
import json
import argparse

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
    parser = argparse.ArgumentParser(description="Parse annotation files and convert bounding boxes to YOLO format.")
    parser.add_argument("--input", required=True, help="Path to the annotation folder.")
    args = parser.parse_args()

    input_folder = args.input
    class_counts = parse_annotations(input_folder)

    # Print class counts
    print_class_counts(class_counts)

if __name__ == "__main__":
    main()
