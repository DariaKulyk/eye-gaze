import json
import os
import random
import shutil
import copy

desired_categories = ["Human hand", "Human head", "Person", "Hat", "Human hair", "Dress", "Human eye", "Building", "Tree", "Animal", "Human mouth"]
desired_limits = {
    'Person': 13470, 
    'Human hand': 4835, 
    'Human hair': 4835, 
    'Human head': 6907, 
    'Human eye': 5450, 
    'Human mouth': 2752, 
    'Building': 2613, 
    'Tree': 1726, 
    'Animal': 294, 
    'Dress': 1036, 
    'Hat': 200
}

# Function to modify the number of instances per category according to desired limits
def modify_json(json_data, desired_limits):
    # Count the existing annotations per category
    annotation_counts = {}
    for annotation in json_data['annotations']:
        category_id = annotation['category_id']
        if category_id in annotation_counts:
            annotation_counts[category_id] += 1
        else:
            annotation_counts[category_id] = 1
    
    # Create a mapping of category_id to a list of annotations for that category
    annotations_by_category = {}
    for annotation in json_data['annotations']:
        category_id = annotation['category_id']
        if category_id in annotations_by_category:
            annotations_by_category[category_id].append(annotation)
        else:
            annotations_by_category[category_id] = [annotation]
    
    # Lists to store modified annotations and images
    modified_annotations = []
    modified_images = []
    
    for category, limit in desired_limits.items():
        category_id = get_category_id(json_data, category)
        
        if category_id is None:
            print(f"Category '{category}' not found in the JSON.")
            continue
        
        annotations_for_category = annotations_by_category.get(category_id, [])
        selected_annotations = annotations_for_category[:limit]
        modified_annotations.extend(selected_annotations)
        
        
    valid_image_ids = set(annotation["image_id"] for annotation in modified_annotations)
    # Filter images
    modified_images = [image for image in json_data["images"] if image["id"] in valid_image_ids]
    
    # Update the modified annotations and images in the JSON data
    json_data['annotations'] = modified_annotations
    json_data['images'] = modified_images
    
    return json_data
    
def get_category_id(json_data, category_name):
    for category in json_data['categories']:
        if category['name'] == category_name:
            return category['id']
    return None

# Function to delete images that are not in the modified json annotation file
def delete_images_not_in_json(image_directory, json_data):
    # Collect the file names from the updated JSON
    updated_file_names = {image['file_name'] for image in json_data['images']}
    
    # Iterate over the files in the directory and delete the ones not in the JSON
    for file_name in os.listdir(image_directory):
        if file_name not in updated_file_names:
            file_path = os.path.join(image_directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

# Function to split the final dataset into train and validation
def split_dataset(coco_data, split, image_directory):  
    # Shuffle image IDs
    images = coco_data['images']
    annotations = coco_data['annotations']
    random.shuffle(images)
    split_point = int(len(images) * split)
    print(f"Total images: {len(images)}, Total annotations: {len(annotations)}, Image split point: {split_point}") # 11260, 9571

    # Split the image annotations into training and validation sets
    train_image_annotations = images[:split_point]
    train_ids = []
    for _image in train_image_annotations:
        train_ids.append(_image['id'])
    val_image_annotations = images[split_point:]
    val_ids = []
    for _image in val_image_annotations:
        val_ids.append(_image['id'])

    # Create new annotation files for training and validation
    train_data = copy.deepcopy(coco_data)
    train_data["images"] = [img for img in train_data["images"] if img["id"] in train_ids]
    train_data["annotations"] = [ann for ann in train_data["annotations"] if ann["image_id"] in train_ids]
    print (f"Number of train images: {len(train_data['images'])}, Number of train annotations: {len(train_data['annotations'])}") # 9571, 37638
    
    val_data = copy.deepcopy(coco_data)
    val_data["images"] = [img for img in val_data["images"] if img["id"] in val_ids]
    val_data["annotations"] = [ann for ann in val_data["annotations"] if ann["image_id"] in val_ids]
    print (f"Number of validation images: {len(val_data['images'])}, Number of validation annotations: {len(val_data['annotations'])}") # 1689, 6670
    
    # Save the new annotation files
    output_train_annotation_file = os.path.join("/Volumes/Samsung_USB/annotations", 'train_annotations.json')
    with open(output_train_annotation_file, 'w') as f:
        json.dump(train_data, f, indent = 2)
    
    output_val_annotation_file = os.path.join("/Volumes/Samsung_USB/annotations", 'val_annotations.json')
    with open(output_val_annotation_file, 'w') as f:
        json.dump(val_data, f, indent = 2)
    
    
    # Copy images to the respective splits
    for _image in train_data['images']:
        src_path = image_directory + "/" + _image['file_name']
        dst_path = "/Volumes/Samsung_USB/train/" + _image['file_name']
        shutil.copy(src_path, dst_path)
        
    for _image in val_data['images']:
        src_path = image_directory + "/" + _image['file_name']
        dst_path = "/Volumes/Samsung_USB/validation/" + _image['file_name']
        shutil.copy(src_path, dst_path)


def main():
    # JSON file with unmodified annotations
    input_json="/Volumes/Samsung_USB/coco-dataset/labels.json"
    # JSON file to store modified annotations
    output_json = "/Volumes/SAMSUNG_USB/coco-dataset/positive_limited_ann.json"
    # Directory with all images to be updated
    image_directory = "/Volumes/SAMSUNG_USB/coco-dataset/data"
    
    # Load the JSON file with all anotations
    with open(input_json, 'r') as file:
        json_data = json.load(file)

    # Filter categories 
    filtered_categories = [category for category in json_data["categories"] if category["name"] in desired_categories]
    category_id_mapping = {category["id"]: index + 1 for index, category in enumerate(filtered_categories)}
    for category in filtered_categories:
        category["id"] = category_id_mapping[category["id"]]

    # Filter annotations and update category IDs
    filtered_annotations = [annotation for annotation in json_data["annotations"] if annotation["category_id"] in category_id_mapping]
    for annotation in filtered_annotations:
        annotation["category_id"] = category_id_mapping[annotation["category_id"]]

    valid_image_ids = set(annotation["image_id"] for annotation in filtered_annotations)
    # Filter images
    filtered_images = [image for image in json_data["images"] if image["id"] in valid_image_ids]

    # Update categories, annotations, and images in the JSON data
    json_data["categories"] = filtered_categories
    json_data["annotations"] = filtered_annotations
    json_data["images"] = filtered_images

    # Function to modify the number of instances per category 
    modified_json = modify_json(json_data, desired_limits)

    with open(output_json, 'w') as file:
        json.dump(modified_json, file, indent=2)

    # Function to delete images that are not in the modified json annotation file
    delete_images_not_in_json(image_directory, modified_json)
    print("Images filtered")
    print("Splitting dataset...")
    
    split_dataset(modified_json, 0.85, image_directory)
    
    
if __name__ == "__main__":
    main()
    


