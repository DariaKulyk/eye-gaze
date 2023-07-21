# eye-gaze

Code from the paper "Combining object detection and eye-tracking to identify points of interest for VR art exhibition visitors" (thesis.pdf).

## Usage

### Create a heatmap visualization and perform statistical tests

'''
usage: create_heatmap.py [-h] [--test_images TEST_IMAGES] [--annotation_file ANNOTATION_FILE] [--heatmaps HEATMAPS] [--anova_file ANOVA_FILE]

arguments:
  -h, --help            show this help message and exit
  --test_images TEST_IMAGES
                        directory where painting images are stored (default: /Volumes/SAMSUNG_USB/test-images)
  --annotation_file ANNOTATION_FILE
                        json annotation file with ground truth bounding boxes (default: /Volumes/SAMSUNG_USB/annotations/test-ann-27-06.json)
  --heatmaps HEATMAPS   directory where csv heatmaps are stored (default: ../../data/heatmaps)
  --anova_file ANOVA_FILE
                        csv file for one-way anova test (default: ../../data/processed/anova-data-filtered-time.csv)

'''
