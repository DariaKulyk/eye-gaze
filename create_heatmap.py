'''Run this code first. It reads csv files and images and performs 
the calculations for statistical analysis and heatmaps
'''

import numpy as np
import pandas as pd 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy import stats
import json
import os
import glob
import math

from map_coordinates import calculate_gaze_duration_in_objects, find_most_viewed_object_ind, find_most_viewed_object_group
from stat_analysis import calculate_significance, perform_one_way_anova
from config import CLASSES
from utils import normalize_colors, modify_csv

def plot_heatmap(painting_img, x_coordinates, y_coordinates, duration_arr, my_colormap, radius):
    fig, ax = plt.subplots() # Create axes
    ax.imshow(painting_img)  # Display the painting image
    cmap = ListedColormap(my_colormap)
    
    for xi, yi, dur in zip(x_coordinates, y_coordinates, duration_arr):
        # color_index = math.ceil(dur * 20)
        # circle = plt.Circle((xi, yi), radius, color=my_colormap[color_index])
        circle = plt.Circle((xi, yi), radius, color=cmap(dur), alpha = 0.4)
        ax.add_patch(circle)
    
    norm = plt.Normalize(min(duration_arr), max(duration_arr)) # linearly normalizes data into the [0.0, 1.0] interval
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
my_colormap = normalize_colors(rgb_colors)

# Stores information on how many times H0 hypothesis is rejected
rejected = 0
array_of_dicts = []
average_durations_object_t_test = []
average_durations_no_object_t_test = []
total_durations_object_participant_test = []
total_durations_no_object_participant_test = []
average_higher = 0
total_higher = 0


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

heatmaps = 'heatmaps'
files = []
for path in os.listdir(heatmaps):
    file = os.path.join(heatmaps, path)
    files.append(file)

for file_csv in files:
    # Function to add header to csv
    modify_csv(file_csv)
    df = pd.read_csv(file_csv, delimiter=",")
    
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
    participant_dur = []
    total_durations_objects_arr = []
    total_durations_no_objects_arr = []
    average_durations_objects_arr = []
    average_durations_no_objects_arr = []
    participant_dict = {}
    
    for i in range(len(test_images)):
    # for i in range(3):
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
            # painting_img = mpimg.imread(painting)
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
            
            # average_gaze = np.mean(duration_arr)
            # for _gaze in duration_arr:
            #     if _gaze < average_gaze:
            #         duration_arr.remove(_gaze)
            
            # print(x_coordinates) 
            # print(y_coordinates)
            # print(duration_arr)

            # print(len(duration_arr)) 
            # print(len(x_coordinates))
            # print(len(y_coordinates))
            
            unit_width = painting_width/100
            unit_height = painting_height/100
            radius = min(unit_width, unit_height) / 2
            
            # my_colormap = [
            # (0.984313725490196, 0.7333333333333333, 0.0784313725490196),
            # (0.984313725490196, 0.7176470588235294, 0.07450980392156863),
            # (0.984313725490196, 0.7019607843137254, 0.06666666666666667),
            # (0.9803921568627451, 0.6862745098039216, 0.06666666666666667),
            # (0.9803921568627451, 0.6705882352941176, 0.06274509803921569),
            # (0.9803921568627451, 0.6549019607843137, 0.06274509803921569),
            # (0.9764705882352941, 0.6431372549019608, 0.06274509803921569),
            # (0.9764705882352941, 0.6274509803921569, 0.06666666666666667),
            # (0.9725490196078431, 0.611764705882353, 0.06666666666666667),
            # (0.9725490196078431, 0.596078431372549, 0.07058823529411765),
            # (0.9686274509803922, 0.5803921568627451, 0.07450980392156863),
            # (0.9647058823529412, 0.5647058823529412, 0.0784313725490196),
            # (0.9607843137254902, 0.5490196078431373, 0.08235294117647059),
            # (0.9568627450980393, 0.5333333333333333, 0.09019607843137255),
            # (0.9568627450980393, 0.5176470588235295, 0.09411764705882353),
            # (0.9490196078431372, 0.5019607843137255, 0.09803921568627451),
            # (0.9450980392156862, 0.48627450980392156, 0.10588235294117647),
            # (0.9411764705882353, 0.47058823529411764, 0.10980392156862745),
            # (0.9372549019607843, 0.4549019607843137, 0.11372549019607843),
            # (0.9333333333333333, 0.4392156862745098, 0.12156862745098039),
            # (0.9254901960784314, 0.4235294117647059, 0.12549019607843137),
            # (0.9215686274509803, 0.40784313725490196, 0.12941176470588237),
            # (0.9176470588235294, 0.39215686274509803, 0.13333333333333333),
            # (0.9098039215686274, 0.3764705882352941, 0.1411764705882353),
            # (0.9019607843137255, 0.3607843137254902, 0.1450980392156863),
            # (0.8980392156862745, 0.34509803921568627, 0.14901960784313725),
            # (0.8901960784313725, 0.32941176470588235, 0.15294117647058825),
            # (0.8823529411764706, 0.3137254901960784, 0.1568627450980392),
            # (0.8745098039215686, 0.29411764705882354, 0.1607843137254902),
            # (0.8666666666666667, 0.2784313725490196, 0.16862745098039217),
            # (0.8627450980392157, 0.2627450980392157, 0.17254901960784313),
            # (0.8509803921568627, 0.24313725490196078, 0.17647058823529413),
            # (0.8431372549019608, 0.22745098039215686, 0.1803921568627451),
            # (0.8352941176470589, 0.20784313725490197, 0.1843137254901961),
            # (0.8274509803921568, 0.18823529411764706, 0.18823529411764706),
            # (0.8196078431372549, 0.16862745098039217, 0.19215686274509805),
            # (0.807843137254902, 0.14901960784313725, 0.19215686274509805),
            # (0.8, 0.12549019607843137, 0.19607843137254902),
            # (0.792156862745098, 0.09803921568627451, 0.2),
            # (0.7803921568627451, 0.06666666666666667, 0.20392156862745098)
            # ]
            
            plot_heatmap(painting_img, x_coordinates, y_coordinates, duration_arr, my_colormap, radius)
            
            # fig, ax = plt.subplots() # Create axes
            # ax.imshow(painting_img)  # Display the painting image

            gaze_data = []
          
            for xi, yi, dur in zip(x_coordinates, y_coordinates, duration_arr):
                gaze_data.append([xi, yi, dur])
                # rect = plt.Circle((xi, yi), radius, color=cmap(dur))
                # ax.add_patch(rect)
            
            
            # # # # # # Create color map
            # cmap = plt.cm.tab10   
            # norm = plt.Normalize(min(duration_arr), max(duration_arr)) #linearly normalizes data into the [0.0, 1.0] interval
            # cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
            # cbar.set_label('Duration of eye gaze')
            # plt.show()
            
            total_time = np.sum(duration_arr)
            mean_duration = np.mean(duration_arr)  
            print(f"Image: {image_name}, Total time:", total_time, "Average gaze:", mean_duration)
            
            participant_dur.append(total_time)
            
            # ground_truth_boxes = []
            boxes_category_dict = {}
            for image in json_data["images"]:
                if image["file_name"].startswith(image_name):
                    image_id = image["id"]
                    for annotation in json_data["annotations"]:
                        if annotation["image_id"] == image_id:
                            box = annotation["bbox"]
                            # category_id = annotation["category_id"]
                            category = CLASSES[annotation["category_id"]]
                            # ground_truth_boxes.append(box)
                            if category in boxes_category_dict:
                                # If it exists, append the new bbox to the existing list of bbox values
                                boxes_category_dict[category].append(box)
                            else:
                                # If it doesn't exist, create a new list with the bbox value
                                boxes_category_dict[category] = [box]
                          
            
            # total_duration_in_objects, total_duration_outside_objects, average_duration_in_objects, average_duration_outside_objects, max_duration_in_objects, max_duration_outside_objects = calculate_gaze_duration_in_objects(gaze_data, ground_truth_boxes)
            total_duration_in_objects, total_duration_outside_objects, average_duration_in_objects, average_duration_outside_objects, max_duration_in_objects, max_duration_outside_objects, object_duration_dict = calculate_gaze_duration_in_objects(gaze_data, radius, boxes_category_dict)
            participant_painting_dict = {}
            
            if average_duration_in_objects > average_duration_outside_objects:
                average_higher += 1
            if total_duration_in_objects > total_duration_outside_objects:
                total_higher += 1
            
            participant_painting_dict[image_name] = object_duration_dict
            print(f"Total duration in objects: {total_duration_in_objects}")
            print(f"Total duration outside objects: {total_duration_outside_objects}")
            print(f"Average duration in objects: {average_duration_in_objects}")
            print(f"Average duration outside objects: {average_duration_outside_objects}")
            print(f"Max duration in objects: {max_duration_in_objects}")
            print(f"Max duration outside objects: {max_duration_outside_objects}")
            print(participant_painting_dict)
            print('')
            
            participant_dict.update(participant_painting_dict)
            
#             # total_durations_objects_arr.append(total_duration_in_objects)
#             # total_durations_no_objects_arr.append(total_duration_outside_objects)
            
            average_durations_objects_arr.append(average_duration_in_objects)
            average_durations_no_objects_arr.append(average_duration_outside_objects)
            
            
#             # quantile_threshold = 0.25  # Set the desired percentile (e.g., 90%)
#             # threshold = np.quantile(duration_arr, quantile_threshold)

#             # filtered_durations = [duration for duration in duration_arr if duration >= threshold]
#             # # Print the filtered durations
#             # print("Durations in the 25th quartile:", filtered_durations)
    avg_durations_object_participant = np.mean(average_durations_objects_arr)
    avg_durations_no_object_participant = np.mean(average_durations_no_objects_arr)
    average_durations_object_t_test.append(avg_durations_object_participant)
    average_durations_no_object_t_test.append(avg_durations_no_object_participant)
    
    average_time = np.mean(participant_dur)
    most_viewed_by_participant = find_most_viewed_object_ind(participant_dict)
    array_of_dicts.append(most_viewed_by_participant)
    # paricipant_num = file_csv.split('/')[1].split('_')[0]
    print(f"Participant: {paricipant_num}, Average time:", average_time)
    # rejected = calculate_significance(average_durations_objects_arr, average_durations_no_objects_arr, rejected)
    # statistic, p_value, is_t_test, rejected_t, rejected_w = calculate_significance(average_durations_objects_arr, average_durations_no_objects_arr, rejected_t, rejected_w)
    # print("Within bounding boxes: ", durations_objects_arr)
    # print("Outside bounding boxes: ", durations_no_objects_arr)
    print("Within bounding boxes: ", average_durations_objects_arr)
    print("Outside bounding boxes: ", average_durations_no_objects_arr)
    print("Most viewed by participant:", most_viewed_by_participant)
    print('-' * 200)

# print("Number of times average time spent on areas containting objects was significantly greater: ", rejected)
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


# print(most_viewed_group)
# most_viewed, duration = find_most_viewed_object(array_of_dicts)
# print("Most viewed object:", most_viewed)
# print("Duration:", duration)
# print("Number of times average time spent on areas containting objects was significantly greater: ", rejected_t)
# print("Number of times median time spent on areas containting objects was significantly greater: ", rejected_w)























