'''Run this code first. It reads csv files and images and performs 
the calculations for statistical analysis and heatmaps
'''

import numpy as np
import pandas as pd 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
import os
import glob
import math

from map_coordinates import calculate_gaze_duration_in_objects, find_most_viewed_object_ind, find_most_viewed_object_group
from stat_analysis import calculate_significance, perform_one_way_anova
from utils import normalize_colors, modify_csv
from config import CLASSES

# This function plots the heatmap overlay
def plot_heatmap(painting_img, x_coordinates, y_coordinates, duration_arr, my_colormap, radius):
    fig, ax = plt.subplots() # Create axes
    ax.imshow(painting_img)  # Display the painting image
    cmap = ListedColormap(my_colormap)
    
    for xi, yi, dur in zip(x_coordinates, y_coordinates, duration_arr):
        circle = plt.Circle((xi, yi), radius, color=cmap(dur), alpha = 0.4) # Plot gaze point as circle
        ax.add_patch(circle)
    
    norm = plt.Normalize(min(duration_arr), max(duration_arr)) # Linearly normalizes data into the [0.0, 1.0] interval
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
    
    cbar.set_label('Duration of eye gaze')
    plt.show()
 
# Colors for the heatmap, change according to your requirements     
rgb_colors = [(251, 187, 20), 
            (251, 183, 19),
            (251, 179, 17),
            (250, 175, 17),
            (250, 171, 16),
            (250, 167, 16),
            (249, 164, 16),
            (249, 160, 17),
            (248, 156, 17),
            (248, 152, 18),
            (247, 148, 19),
            (246, 144, 20),
            (245, 140, 21),
            (244, 136, 23),
            (244, 132, 24),
            (242, 128, 25),
            (241, 124, 27),
            (240, 120, 28),
            (239, 116, 29),
            (238, 112, 31),
            (236, 108, 32),
            (235, 104, 33),
            (234, 100, 34),
            (232, 96, 36),
            (230, 92, 37),
            (229, 88, 38),
            (227, 84, 39),
            (225, 80, 40),
            (223, 75, 41),
            (221, 71, 43),
            (220, 67, 44),
            (217, 62, 45),
            (215, 58, 46),
            (213, 53, 47),
            (211, 48, 48),
            (209, 43, 49),
            (206, 38, 49),
            (204, 32, 50),
            (202, 25, 51),
            (199, 17, 52)]
# Function to change RGB colors to RGB normalized colors
my_colormap = normalize_colors(rgb_colors) 

# Stores information on how many times H0 hypothesis is rejected
rejected = 0
# Stores information on most viewed objects by all participants across all paintings
array_of_dicts = []
# Stores information on average gaze duration on areas with objects; each element corresponds to one painting
average_durations_object_t_test = []
# Stores information on average gaze duration on areas without objects; each element corresponds to one painting
average_durations_no_object_t_test = []
# Stores how many times the average gaze duration on areas with objects was higher
average_higher = 0
# Stores how many times the total gaze duration on areas with objects was higher
total_higher = 0

# Directory where images of paintings are stored
DIR_TEST = "/Volumes/SAMSUNG_USB/test-images" 
test_images = []
if os.path.isdir(DIR_TEST):
    image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*JPG', "*.PNG"]
    for file_type in image_file_types:
        test_images.extend(glob.glob(f"{DIR_TEST}/{file_type}"))
else:
    test_images.append(DIR_TEST)
print(f"Test instances: {len(test_images)}")

 # Read JSON file with ground truth boxes
annotation_file = '/Volumes/SAMSUNG_USB/annotations/test-ann-27-06.json'
with open(annotation_file, 'r') as f:
    json_data = json.load(f)

# Directory where heatmaps are stored
heatmaps = 'data/heatmaps'
files = []
for path in os.listdir(heatmaps):
    file = os.path.join(heatmaps, path)
    files.append(file)

for file_csv in files:
    # Function to add header to csv
    modify_csv(file_csv)
    df = pd.read_csv(file_csv, delimiter=",") # Read each csv heatmap file
    
    paricipant_num = file_csv.split('/')[1].split('_')[0]
    print("Participant:", paricipant_num)
    
    object_names=[]
    for _, row in df.iterrows():
        if pd.isna(row[1]): # rows with object names (paintings) have an empty second element
            object_names.append(row[0])
            
    # Create a 3d array
    result = []
    current_array = None

    heatmap_data = df.values
    for row in heatmap_data:
        if row[0] in object_names:
            if current_array is not None:
                result.append(current_array)
            current_array = np.expand_dims(row, axis=0)
        else:
            if current_array is not None:
                current_array = np.concatenate((current_array, np.expand_dims(row, axis=0)), axis=0)
    
    if current_array is not None:
        result.append(current_array)

    array_3d = np.stack(result, axis=0)

    painting_index = None
    
    # Stores the total gaze duration for an individual participant per painting 
    participant_dur = []
    # Stores the average gaze duration on objects for an individual participant per painting 
    average_durations_objects_arr = []
    # Stores the average gaze duration outside objects for an individual participant per painting
    average_durations_no_objects_arr = []
    # Stores gaze duration on different objects for all participants
    participant_dict = {}
    
    # Iterate over all paintings
    for i in range(len(test_images)):
        # Arrays to store eye gaze data
        x_coordinates = []
        y_coordinates = []
        duration_arr = []
        
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        
        # Find index of a 2d array with painting name 
        for a, array_2d in enumerate(array_3d):  # a = index of 2d array representing different paintings
            if array_2d[0, 0].startswith(image_name):
                painting_index = a
                break
        
        if painting_index is not None:
            painting_img = mpimg.imread(test_images[i])
            painting_height = painting_img.shape[0]
            painting_width = painting_img.shape[1]
            scale_width = painting_width / 100
            scale_height = painting_height / 100
            for b, row in enumerate(array_3d[painting_index][1:], start=1):   # b = index of a row in a 2d array
                for c, element in enumerate(row, start=1): # c = index of an element in a row
                    float_element = float(element)
                    if float_element != 0:
                        x = scale_width * c
                        y = scale_height * b
                        x_coordinates.append(x)
                        y_coordinates.append(y)
                        duration_arr.append(float_element)
            
            
            # Ensure one gaze point corresponds to one painting unit
            unit_width = painting_width/100 
            unit_height = painting_height/100
            radius = min(unit_width, unit_height) / 2 # Calculate radius of gaze point
            
            
            # Array to store gaze point coordinates and their corresponding durations
            gaze_data = []
            for xi, yi, dur in zip(x_coordinates, y_coordinates, duration_arr):
                gaze_data.append([xi, yi, dur])
            
            
            total_time = np.sum(duration_arr)
            mean_duration = np.mean(duration_arr)  
            print(f"Image: {image_name}, Total time:", total_time, "Average gaze:", mean_duration)
            
            participant_dur.append(total_time)
            
            # Create a dictionary based on the ground truth json file that maps categories to lists of bounding boxes associated with each category
            boxes_category_dict = {}
            for image in json_data["images"]:
                if image["file_name"].startswith(image_name):
                    image_id = image["id"]
                    for annotation in json_data["annotations"]:
                        if annotation["image_id"] == image_id:
                            box = annotation["bbox"]
                            category = CLASSES[annotation["category_id"]]
                            if category in boxes_category_dict:
                                # If it exists, append the new bbox to the existing list of bbox values
                                boxes_category_dict[category].append(box)
                            else:
                                # If it doesn't exist, create a new list with the bbox value
                                boxes_category_dict[category] = [box]
                          
            
            # Function to calculate durations for gaze points within objects and gaze points outside objects 
            total_duration_in_objects, total_duration_outside_objects, average_duration_in_objects, average_duration_outside_objects, max_duration_in_objects, max_duration_outside_objects, object_duration_dict = calculate_gaze_duration_in_objects(gaze_data, radius, boxes_category_dict)
            
            # Stores gaze duration on different objects for all participants
            participant_painting_dict = {}
            
            if average_duration_in_objects > average_duration_outside_objects:
                average_higher += 1
            if total_duration_in_objects > total_duration_outside_objects:
                total_higher += 1
            
            # Prints summary for an individual participant for one painting image
            participant_painting_dict[image_name] = object_duration_dict
            print(f"Total duration in objects: {total_duration_in_objects}")
            print(f"Total duration outside objects: {total_duration_outside_objects}")
            print(f"Average duration in objects: {average_duration_in_objects}")
            print(f"Average duration outside objects: {average_duration_outside_objects}")
            # print(f"Max duration in objects: {max_duration_in_objects}")
            # print(f"Max duration outside objects: {max_duration_outside_objects}")
            print(participant_painting_dict)
            print('')
            
            participant_dict.update(participant_painting_dict)
   
            average_durations_objects_arr.append(average_duration_in_objects)
            average_durations_no_objects_arr.append(average_duration_outside_objects)
            
            # Function to plot heatmap overlay
            plot_heatmap(painting_img, x_coordinates, y_coordinates, duration_arr, my_colormap, radius)
            
            
    avg_durations_object_participant = np.mean(average_durations_objects_arr)
    avg_durations_no_object_participant = np.mean(average_durations_no_objects_arr)
    average_durations_object_t_test.append(avg_durations_object_participant)
    average_durations_no_object_t_test.append(avg_durations_no_object_participant)
    
    # Print summary for an individual participant for all painting images
    average_time = np.mean(participant_dur)
    most_viewed_by_participant = find_most_viewed_object_ind(participant_dict)
    array_of_dicts.append(most_viewed_by_participant)
    print(f"Participant: {paricipant_num}, Average time:", average_time)
    # print("Within bounding boxes: ", average_durations_objects_arr)
    # print("Outside bounding boxes: ", average_durations_no_objects_arr)
    print("Most viewed by participant:", most_viewed_by_participant)
    print('-' * 200)

# Print summary for all participants for all painting images
most_viewed_group = find_most_viewed_object_group(array_of_dicts)
print("Most viewed objects per painting:")
for _dictionary in most_viewed_group:
    print(_dictionary)
print()

print("Meaningful versus non-meaningful areas:")
calculate_significance(average_durations_object_t_test, average_durations_no_object_t_test)
average_higher = math.floor(average_higher/(31*19)*100)
total_higher = math.floor(total_higher/(31*19)*100)
print(f"Average gaze duration on objects was higher in {average_higher}% cases")
print(f"Total gaze duration on objects was higher in {total_higher}% cases")
print()

print("Object-Interest analysis:")
perform_one_way_anova()


