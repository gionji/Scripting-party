import os
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_stats_folder(input_folder):
    stats_folder = os.path.join(input_folder, "stats")
    if not os.path.exists(stats_folder):
        os.makedirs(stats_folder)
    return stats_folder




def calculate_statistics(annotation_folder, stats_folder):
    # Implement your statistics calculations here
    # For example, you can calculate statistics on bounding boxes, metadata, vehicle classes, etc.
    # Save the results in the stats_folder.

    # Example: Calculate and save some statistics
    # Replace this with your actual statistics calculations
    statistics_result = {
        "example_stat": 123,
    }

    with open(os.path.join(stats_folder, "example_stat.json"), "w") as stat_file:
        json.dump(statistics_result, stat_file, indent=4)




def get_available_classes(annotation_folder):
    available_classes = set()

    # List annotation files in the input folder
    annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith(".txt")]

    for annotation_file in annotation_files:
        annotation_file_path = os.path.join(annotation_folder, annotation_file)

        with open(annotation_file_path, "r") as f:
            annotation_data = json.load(f)

        vehicle_classes = annotation_data.get("vehicle_class", [])
        available_classes.update(set(vehicle_classes))

    return sorted(list(available_classes))
    
    
def calculate_average_bboxes_per_image(annotation_folder):
    class_bboxes_count = {}
    total_images = 0

    # List annotation files in the input folder
    annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith(".txt")]

    for annotation_file in annotation_files:
        annotation_file_path = os.path.join(annotation_folder, annotation_file)

        with open(annotation_file_path, "r") as f:
            annotation_data = json.load(f)

        bboxes = annotation_data.get("bboxes", [])
        vehicle_classes = annotation_data.get("vehicle_class", [])

        if not bboxes:
            continue

        total_images += 1

        for bbox, bbox_class in zip(bboxes, vehicle_classes):
            if bbox_class not in class_bboxes_count:
                class_bboxes_count[bbox_class] = 0
            class_bboxes_count[bbox_class] += 1

    # Calculate average bboxes per image for each class
    average_bboxes_per_image = {class_id: count / total_images for class_id, count in class_bboxes_count.items()}

    return average_bboxes_per_image

def generate_average_bboxes_bar_graph(annotation_folder, stats_folder):
    average_bboxes_per_image = calculate_average_bboxes_per_image(annotation_folder)

    class_ids = list(average_bboxes_per_image.keys())
    average_bboxes = list(average_bboxes_per_image.values())

    plt.figure(figsize=(12, 6))
    plt.bar(class_ids, average_bboxes, color='skyblue')
    plt.xlabel("Class ID")
    plt.ylabel("Average Bboxes per Image")
    plt.title("Average Bboxes per Image for Each Class")
    plt.xticks(class_ids)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    bar_graph_path = os.path.join(stats_folder, "average_bboxes_per_image.png")
    plt.savefig(bar_graph_path)
    plt.close()    
    
   

def generate_probability_image(annotation_folder, stats_folder, image_size= (1920, 1080)):
    output_image = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    # List annotation files in the input folder
    annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith(".txt")]

    for annotation_file in annotation_files:
        annotation_file_path = os.path.join(annotation_folder, annotation_file)

        with open(annotation_file_path, "r") as f:
            annotation_data = json.load(f)

        bboxes = annotation_data.get("bboxes", [])

        if not bboxes:
            continue

        for bbox in bboxes:
            # Calculate bounding box coordinates based on the specified image size
            x1 = int(bbox[0][0] )
            y1 = int(bbox[0][1] )
            x2 = int(bbox[1][0] )
            y2 = int(bbox[1][1] )

            # Create a binary mask for the bounding box
            mask = np.zeros_like(output_image, dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=cv2.FILLED)      

            # Add the binary mask to the output image
            output_image += mask
            

    # Normalize the output image for visualization
    output_image_normalized = (output_image / np.max(output_image) * 255).astype(np.uint8)

    # Create and save a heatmap
    heatmap = cv2.applyColorMap(output_image_normalized, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(stats_folder, "bboxes_distribution_map.png")
    cv2.imwrite(heatmap_path, heatmap)



def generate_class_heatmap(annotation_folder, stats_folder, class_id, image_size=(1920, 1080)):
    # List annotation files in the input folder
    annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith(".txt")]

    # Initialize an image with all zeros for the selected class
    class_heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    for annotation_file in annotation_files:
        annotation_file_path = os.path.join(annotation_folder, annotation_file)

        with open(annotation_file_path, "r") as f:
            annotation_data = json.load(f)

        bboxes = annotation_data.get("bboxes", [])
        vehicle_classes = annotation_data.get("vehicle_class", [])

        if not bboxes:
            continue

        for bbox, bbox_class in zip(bboxes, vehicle_classes):
            if bbox_class == class_id:
                # Calculate bounding box coordinates based on the specified image size
                x1 = int(bbox[0][0] )
                y1 = int(bbox[0][1] )
                x2 = int(bbox[1][0] )
                y2 = int(bbox[1][1] )

                # Create a binary mask for the bounding box
                mask = np.zeros_like(class_heatmap, dtype=np.uint32)
                mask[y1:y2, x1:x2] = 1

                # Add the binary mask to the class-specific heatmap
                class_heatmap += mask

    # Normalize and save the class-specific heatmap
    normalized_heatmap = (class_heatmap / np.max(class_heatmap) * 255).astype(np.uint8)
    
    normalized_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(stats_folder, f"class_{class_id}_heatmap.png")
    cv2.imwrite(heatmap_path, normalized_heatmap)
    
    
def get_weather_parameters(annotation_folder):
    # List annotation files in the input folder
    annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith(".txt")]

    # Initialize dictionaries to store weather parameter values
    weather_parameters = {
        "cloudiness": [],
        "precipitation": [],
        "precipitation_deposits": [],
        "wind_intensity": [],
        "sun_azimuth_angle": [],
        "sun_altitude_angle": [],
        "fog_density": [],
        "fog_distance": [],
        "wetness": [],
    }

    for annotation_file in annotation_files:
        annotation_file_path = os.path.join(annotation_folder, annotation_file)

        with open(annotation_file_path, "r") as f:
            annotation_data = json.load(f)

        metadata = annotation_data.get("metadata", {})

        for param, value in metadata.items():
            if param in weather_parameters:
                weather_parameters[param].append(value)

    return weather_parameters


def generate_weather_parameter_histograms(annotation_folder, stats_folder):
    weather_parameters = get_weather_parameters(annotation_folder)

    for param, values in weather_parameters.items():
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel(param)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {param}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        histogram_path = os.path.join(stats_folder, f"{param}_distribution.png")
        plt.savefig(histogram_path)
        plt.close()    

    
    
    

def main():
    parser = argparse.ArgumentParser(description="Generate a probability image from annotation files.")
    parser.add_argument("--input", required=True, help="Path to the annotation folder.")
    parser.add_argument("--image-size", default=(1920, 1080), type=lambda s: tuple(map(int, s.split(','))),
                        help="Image size in the format 'width,height'. Default is full HD (1920, 1080).")
    args = parser.parse_args()

    annotation_folder = args.input
    stats_folder = create_stats_folder(annotation_folder)
    image_size = args.image_size
    
    print('Image size', image_size)
     
    # Calculate statistics
    calculate_statistics(annotation_folder, stats_folder)

    #generate_probability_image(annotation_folder, stats_folder)
    
    class_ids = get_available_classes(annotation_folder)
    
    for class_id in class_ids:
        generate_class_heatmap(annotation_folder, stats_folder, class_id)
        
    generate_average_bboxes_bar_graph(annotation_folder, stats_folder)    
    
    generate_weather_parameter_histograms(annotation_folder, stats_folder)
    
    


if __name__ == "__main__":
    main()
