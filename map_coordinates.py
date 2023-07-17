""" Contains functions that prepare the data for future statistical analysis """

import matplotlib.pyplot as plt

""" This function check if the gaze points fall within the bounding 
box of at least one of the categories in categories_for_test """
# categories_for_test = ['Animal', 'Building', 'Dress', 'Hat', 'Human eye', 'Human mouth', 'Human hand', 'Tree']
def check_gaze_in_bounding_boxes(gaze_x, gaze_y, radius, boxes_category_dict):
    categories = []
    flag = False
    for category, bounding_boxes in boxes_category_dict.items():
        # if category in categories_for_test:
            for box in bounding_boxes:
                x_min, y_min, width, height = box
                if gaze_x >= x_min and gaze_x <= x_min + width and gaze_y >= y_min and gaze_y <= y_min + height:
                # Gaze coordinates fall within the bounding box
                    categories.append(category)
                    flag = True
             
    # Gaze coordinates do not fall within any bounding box
    return flag, categories

""" This function calculates duration (average, total, max) 
for gaze points within objects and gaze points outside objects """
def calculate_gaze_duration_in_objects(gaze_data, radius, boxes_category_dict):
    total_duration_in_objects = 0.0
    total_duration_outside_objects = 0.0
    count_objects = 0
    count_no_objects = 0
    processed_gaze_points = []
    duration_objects_arr = []
    duration_no_objects_arr = []
    object_duration_dict = {}

    for gaze_point in gaze_data:
        gaze_x, gaze_y, duration = gaze_point

        flag, categories = check_gaze_in_bounding_boxes(gaze_x, gaze_y, radius, boxes_category_dict)
        if flag:
            # if gaze_point not in processed_gaze_points:
            duration_objects_arr.append(duration)
                # processed_gaze_points.append(gaze_point)
            for category in categories:
                # Uncomment for meaningful versus non-meaningful areas, Experiment 2
                # if category != "Person" and category != "Human head" and category != "Human hair":
                    if category in object_duration_dict:
                        # If it exists, update the value
                        current_value = object_duration_dict[category]
                        new_value = current_value + duration
                        object_duration_dict[category] = new_value
                    else:
                        # If it doesn't exist, add a new key-value pair
                        object_duration_dict[category] = duration
        else:
            duration_no_objects_arr.append(duration)
            
    total_duration_in_objects = sum(duration_objects_arr)
    total_duration_outside_objects = sum(duration_no_objects_arr)
    count_objects = len(duration_objects_arr)
    count_no_objects = len(duration_no_objects_arr)
    average_duration_ouside_objects = total_duration_outside_objects/count_no_objects if count_no_objects != 0 else 0
    average_duration_in_objects = total_duration_in_objects/count_objects if count_objects != 0 else 0
    max_duration_in_objects = max(duration_objects_arr) if count_objects != 0 else 0
    max_duration_outside_objects = max(duration_no_objects_arr) if count_no_objects != 0 else 0

    return total_duration_in_objects, total_duration_outside_objects, average_duration_in_objects, average_duration_ouside_objects, max_duration_in_objects, max_duration_outside_objects, object_duration_dict

""" This function returns a dictionary of most viewed object 
by each participant per each painting """
def find_most_viewed_object_ind(participant_data):
    most_viewed_per_painting = {}
    for painting, objects in participant_data.items():
        if objects:
            most_viewed_object = max(objects, key=objects.get)
            duration = objects[most_viewed_object]
            most_viewed_per_painting[painting] = {'Object': most_viewed_object, 'Duration': duration}
        else:
            most_viewed_per_painting[painting] = {'Object': None, 'Duration': None}
    return most_viewed_per_painting


""" This function returns a dictionary of most viewed object 
by all participants per each painting """
def find_most_viewed_object_group(participant_data):
    most_viewed_objects = {}

    for participant_dict in participant_data:
        for painting, data in participant_dict.items():
            object_name = data['Object']
            duration = data['Duration']
            if duration is not None:
                if painting not in most_viewed_objects:
                    most_viewed_objects[painting] = {object_name: {'Count': 1, 'Duration': duration}}
                else:
                    painting_data = most_viewed_objects[painting]
                    if object_name not in painting_data:
                        painting_data[object_name] = {'Count': 1, 'Duration': duration}
                    else:
                        painting_data[object_name]['Count'] += 1
                        painting_data[object_name]['Duration'] += duration

    most_viewed_list_per_painting = []
    for painting, painting_data in most_viewed_objects.items():
        painting_most_viewed_object = max(painting_data, key=lambda x: painting_data[x]['Count'])
        count = painting_data[painting_most_viewed_object]['Count']
        duration = painting_data[painting_most_viewed_object]['Duration']
        percentage = (count / len(participant_data)) * 100
        most_viewed_list_per_painting.append({'Painting': painting, 'Object': painting_most_viewed_object, 'Count': count, 'Duration': duration, 'Percentage': percentage})

    paintings, objects, count, duration_objects, percentage = split_dictionary(most_viewed_list_per_painting)
    # Plot a bar chart of most viewed objects per painting
    plot_objects_by_painting(objects, paintings, percentage)
    
    return most_viewed_list_per_painting


""" This function splits the dictionary of most viewed objects 
into multiple arrays """
def split_dictionary(dictionary):
    paintings = []
    objects = []
    count = []
    duration_objects = []
    percentage = []
    
    for item in dictionary:
        paintings.append(item['Painting'])
        objects.append(item['Object'])
        count.append(item['Count'])
        duration_objects.append(item['Duration'])
        percentage.append(item['Percentage'])
    
    return paintings, objects, count, duration_objects, percentage

""" This functions plots a bar chart of most popular 
objects by painting """
def plot_objects_by_painting(objects, paintings, percentage):
    color_dict = {
        'Human hair': '#BF9B58',
        'Building': '#7CACDE',
        "Hat": '#73AC55',
        "Person": '#DE9AA4', 
        "Human head": '#ADABE6'
    }

    # Create the bar plot
    fig, ax = plt.subplots()
    bar_width = 0.4

    # Plotting bars for duration on objects
    bars1 = []
    for i, obj in enumerate(objects):
        if obj in color_dict:
            # Use the same color for objects with the same name
            color = color_dict[obj]
        else:
            # Generate a new color for new objects
            color = plt.cm.Pastel1(i % plt.cm.Pastel1.N)
            color_dict[obj] = color

        bar = ax.bar(paintings[i], percentage[i], bar_width, color=color)
        bars1.append(bar)

        # Add object labels to the bars with the same color
        ax.annotate(obj, xy=(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', color='black')

    # Customize the plot
    ax.set_xlabel('Painting')
    # ax.set_ylabel('Total Duration on Objects')
    ax.set_ylabel('Percentage of Participants')
    ax.set_title('Most Popular Objects by Painting')
    ax.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=45)

    # Adjust the layout to prevent overlapping of bars
    fig.tight_layout()

    # Show the plot
    plt.show()


# def find_most_viewed_object_group(participant_data):
#     most_viewed_objects = {}
#     total_participants = len(participant_data)

#     for participant_dict in participant_data:
#         viewed_objects = set()
#         for painting, data in participant_dict.items():
#             object_name = data['Object']
#             # duration = data['Duration']
#             viewed_objects.add(object_name)
            
#         for object_name in viewed_objects:
#             if object_name not in most_viewed_objects:
#                 most_viewed_objects[object_name] = 1
#             else:
#                 most_viewed_objects[object_name] += 1

#     most_viewed_list = []
#     for object_name, count in most_viewed_objects.items():
#         percentage = (count / total_participants) * 100
#         most_viewed_list.append({'Object': object_name, 'Count': count, 'Percentage': percentage})
#             # if painting not in most_viewed_objects:
#             #     most_viewed_objects[painting] = {'Object': object_name, 'Count': 1, 'Duration' : duration}
#             # else:
#             #     if duration > most_viewed_objects[painting].get('Duration'):
#             #         print("update object")
#             #         most_viewed_objects[painting] = {'Object': object_name, 'Count': 1}
#             #     elif duration == most_viewed_objects[painting].get('Duration'):
#             #         most_viewed_objects[painting]['Count'] += 1

#     # for painting, data in most_viewed_objects.items():
#     #     percentage = (data['Count'] / total_participants) * 100
#     #     most_viewed_objects[painting]['Percentage'] = percentage

#     return most_viewed_list

# def find_most_viewed_object(participant_data):
#     object_duration_dict = {}

#     # Iterate through participant data
#     for participant in participant_data:
#         # Get the object with the maximum duration for the participant
#         max_duration_object = max(participant, key=participant.get)

#         # Increment the total duration for that object
#         if max_duration_object in object_duration_dict:
#             object_duration_dict[max_duration_object] += participant[max_duration_object]
#         else:
#             object_duration_dict[max_duration_object] = participant[max_duration_object]

#     # Find the object with the highest total duration
#     most_viewed_object = max(object_duration_dict, key=object_duration_dict.get)
#     duration = object_duration_dict[most_viewed_object]

#     return most_viewed_object, duration



# def calculate_gaze_duration_in_objects(gaze_data, boxes_category_dict):
#     total_duration_in_objects = 0.0
#     total_duration_outside_objects = 0.0
#     count_objects = 0
#     count_no_objects = 0
#     processed_gaze_points = []
#     duration_objects_arr = []
#     duration_no_objects_arr = []
#     object_duration_dict = {}

#     for gaze_point in gaze_data:
#         gaze_x, gaze_y, duration = gaze_point

#         if gaze_point not in processed_gaze_points:
#             flag, category_id = check_gaze_in_bounding_boxes(gaze_x, gaze_y, boxes_category_dict)
#             if flag:
#             # if check_gaze_in_bounding_boxes(gaze_x, gaze_y, boxes_category_dict):
#                 # total_duration_in_objects += duration
#                 # count_objects += 1
#                 duration_objects_arr.append(duration)
#                 if category_id in object_duration_dict:
#                     # If it exists, append the new bbox to the existing list of bbox values
#                     current_value = object_duration_dict[category_id]
#                     new_value = current_value + duration
#                     object_duration_dict[category_id] = new_value
#                 else:
#                     # If it doesn't exist, create a new list with the bbox value
#                     object_duration_dict[category_id] = duration
#             else:
#                 # total_duration_outside_objects += duration
#                 # count_no_objects += 1
#                 duration_no_objects_arr.append(duration)
            
#             processed_gaze_points.append(gaze_point)
    
#     total_duration_in_objects = sum(duration_objects_arr)
#     total_duration_outside_objects = sum(duration_no_objects_arr)
#     count_objects = len(duration_objects_arr)
#     count_no_objects = len(duration_no_objects_arr)
#     average_duration_ouside_objects = total_duration_outside_objects/count_no_objects if count_no_objects != 0 else 0
#     average_duration_in_objects = total_duration_in_objects/count_objects if count_objects != 0 else 0
#     max_duration_in_objects = max(duration_objects_arr) if count_objects != 0 else 0
#     max_duration_outside_objects = max(duration_no_objects_arr) if count_no_objects != 0 else 0
#     # average_duration_ouside_objects = total_duration_outside_objects/count_no_objects if count_no_objects != 0 else 0
#     # average_duration_in_objects = total_duration_in_objects/count_objects if count_objects != 0 else 0

#     return total_duration_in_objects, total_duration_outside_objects, average_duration_in_objects, average_duration_ouside_objects, max_duration_in_objects, max_duration_outside_objects




# def calculate_gaze_duration_in_objects(gaze_data, radius, boxes_category_dict):
#     total_duration_in_objects = 0.0
#     total_duration_outside_objects = 0.0
#     count_objects = 0
#     count_no_objects = 0
#     processed_gaze_points = []
#     duration_objects_arr = []
#     duration_no_objects_arr = []
#     object_duration_dict = {}

#     for gaze_point in gaze_data:
#         gaze_x, gaze_y, duration = gaze_point

#         flag, categories = check_gaze_in_bounding_boxes(gaze_x, gaze_y, radius, boxes_category_dict)
#         if flag:
#             # if gaze_point not in processed_gaze_points:
#             duration_objects_arr.append(duration)
#                 # processed_gaze_points.append(gaze_point)
#             for category in categories:
#                 if category in object_duration_dict:
#                     # If it exists, update the value
#                     current_value = object_duration_dict[category]
#                     new_value = current_value + duration
#                     object_duration_dict[category] = new_value
#                 else:
#                     # If it doesn't exist, add a new key-value pair
#                     object_duration_dict[category] = duration
#         else:
#             duration_no_objects_arr.append(duration)
            
#     total_duration_in_objects = sum(duration_objects_arr)
#     total_duration_outside_objects = sum(duration_no_objects_arr)
#     count_objects = len(duration_objects_arr)
#     count_no_objects = len(duration_no_objects_arr)
#     average_duration_ouside_objects = total_duration_outside_objects/count_no_objects if count_no_objects != 0 else 0
#     average_duration_in_objects = total_duration_in_objects/count_objects if count_objects != 0 else 0
#     max_duration_in_objects = max(duration_objects_arr) if count_objects != 0 else 0
#     max_duration_outside_objects = max(duration_no_objects_arr) if count_no_objects != 0 else 0

#     return total_duration_in_objects, total_duration_outside_objects, average_duration_in_objects, average_duration_ouside_objects, max_duration_in_objects, max_duration_outside_objects, object_duration_dict


