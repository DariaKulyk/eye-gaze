import csv

""" This function changes hex colors to normalized 
rgb values to enable matplotlib ListedColormap generation"""
def normalize_colors(rgb_colors):
    normalized_colors = []
    for color in rgb_colors:
        red_normalized = color[0] / 255
        green_normalized = color[1] / 255
        blue_normalized = color[2] / 255
        normalized_colors.append((red_normalized, green_normalized, blue_normalized))
    return normalized_colors


""" This function adds a header row to csv for easier 
manipulation"""
def modify_csv (csv_file):
    # Read the existing CSV file
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    # Check if csv already modified
    for row in rows:
        if row[0] == "c1":
            return
            
    row_length = len(rows[2])
    # Define values for the first row
    first_row_names = ['c{}'.format(i) for i in range(1, row_length + 1)]

    # Add the first row to the rows list
    rows.insert(0, first_row_names)

    # Write the modified rows to a new CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
