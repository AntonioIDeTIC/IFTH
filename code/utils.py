import numpy as np
import os
import re
from tabulate import tabulate
from colorama import Fore, Style
import cv2


def load_images_by_location(folder_path, n):
    """
    Load and organize images from a specified folder into batches based on their naming conventions.

    The function scans a specified directory for image files, sorts them based on their names and numbering, 
    and organizes them into batches of a specified size. It also handles gaps in the numbering by including 
    placeholder images where expected images are missing.

    Parameters:
    -----------
    folder_path : str
        The path to the folder containing the image files to be loaded. The function expects images to follow 
        a specific naming pattern, e.g., 'prefix_number.jpeg'.
    
    n : int
        The desired batch size for the final output. This indicates how many images should be included in each 
        sub-batch of the final output.

    Returns:
    --------
    List[List[str]]
        A nested list containing batches of image filenames. Each inner list corresponds to a batch of images,
        with the number of images in each batch being at most `n`. If there are gaps in the image numbering, 
        placeholders are added for the missing images.
    """
    # Step 1: List all images in the folder
    all_images = os.listdir(folder_path)
    
    # Step 2: Sort the images based on their names and numbering
    image_pattern = re.compile(r'([A-Za-z]+\d*)_(\d+)\.jpeg')

    def sorting_key(image_name):
        match = image_pattern.search(image_name)
        if match:
            prefix = match.group(1)  # Extract the whole prefix including letters and numbers (e.g., 'A35', 'Tau')
            number = int(match.group(2))  # Extract the number after the underscore
            return (prefix, number)
        return ('', -1)  # Fallback if pattern doesn't match

    sorted_images = sorted(all_images, key=sorting_key)

    # Initialize variables
    location_images = []
    current_batch = []
    current_start_index = None
    expected_number = None
    
    # Iterate through sorted images and build batches
    for image in sorted_images:
        match = image_pattern.search(image)
        if match:
            capture_number = int(match.group(2))
            prefix = match.group(1)

            # Initialize expected number if it's the first image
            if expected_number is None:
                expected_number = capture_number

            # If there's a gap between the current capture_number and the expected_number
            while expected_number < capture_number:
                # Add a placeholder for missing images, to maintain the correct batch size
                if current_start_index is None or expected_number > current_start_index + (n - 1):
                    # Finalize the current batch if it's not empty
                    if current_batch:
                        location_images.append(current_batch)
                    
                    # Start a new batch
                    current_batch = [f'{prefix}_{expected_number}_missing.jpeg']  # Placeholder
                    current_start_index = expected_number
                else:
                    current_batch.append(f'{prefix}_{expected_number}_missing.jpeg')  # Placeholder
                
                expected_number += 1

            # Now add the existing image
            if current_start_index is None or capture_number > current_start_index + (n - 1):
                # Finalize the current batch if it's not empty
                if current_batch:
                    location_images.append(current_batch)
                
                # Start a new batch
                current_batch = [image]
                current_start_index = capture_number
            else:
                current_batch.append(image)
            
            # Move the expected_number forward
            expected_number = capture_number + 1
    
    # Add the last batch if any
    if current_batch:
        location_images.append(current_batch)
    
    # Now split each 20-image batch into sub-batches of size n
    final_batches = []
    for location in location_images:
        for i in range(0, len(location), n):
            final_batches.append(location[i:i + n])
    
    return final_batches

    
def verify_detection(boxes, labels, iou_tresh):
    """
    Verify object detection results against ground truth labels.

    This function computes a confusion matrix for a single class based on the predicted bounding boxes 
    and the ground truth labels. It calculates True Positives, False Positives, True Negatives, 
    and False Negatives, depending on the Intersection over Union (IoU) threshold specified.

    Parameters:
    -----------
    boxes : List[np.ndarray]
        A list of predicted bounding boxes, where each box is represented as an array (e.g., [x_min, y_min, x_max, y_max]).
        
    labels : List[Tuple[int, np.ndarray]]
        A list of ground truth labels, where each label is a tuple consisting of:
        - label_cls: The class label (int or str).
        - label_bbox: The corresponding bounding box (e.g., [x_min, y_min, x_max, y_max]).
        
    iou_thresh : float
        The IoU threshold used to determine whether a detected box is a True Positive or a False Positive.

    Returns:
    --------
    np.ndarray
        A 2x2 confusion matrix with the following layout:
        [[TN, FP],
         [FN, TP]]
        - TN: True Negatives
        - FP: False Positives
        - FN: False Negatives
        - TP: True Positives
    """
    # Initialize confusion matrix for a single class
    confusion_matrix = np.zeros((2, 2), dtype=int)  # 2x2 matrix: [[TN, FP], [FN, TP]]

    matched_labels = set()
    label_set = set([(label_cls, label_bbox) for label_cls, label_bbox in labels])

    if boxes is None or len(boxes) == 0:
        num_labels = len(labels)  # Number of ground truth objects
        
        if num_labels == 0:
            # No labels present, count as True Negative (TN)
            # print("No ground truth labels present. Counting as True Negative (TN).")
            confusion_matrix[0][0] += 1  # True Negative
        else:
            confusion_matrix[1][0] += num_labels  # Increment False Negatives by the number of labels
            
        return confusion_matrix
    
    # Loop through predictions
    for box in boxes:
        best_iou_score = 0
        best_label = (0.0, (0, 0, 0, 0))
        
        for label in labels:
            label_cls, label_bbox = label
            
            iou_score = box_iou(box, label_bbox)
                        
            if iou_score > best_iou_score:
                best_iou_score = iou_score
                best_label = (label_cls, label_bbox)
            
        if best_iou_score >= iou_tresh:
            # Match found (for class 0)
            confusion_matrix[1][1] += 1 # True Positive
            matched_labels.add(best_label)
        else:
            # Probability below threshold; skip detection
            confusion_matrix[0][1] += 1  # False Positive
            # if best_iou_score > 0:
                
            
    # Calculate False Negatives
    unmatched_labels = label_set - matched_labels
    
    for label in unmatched_labels:
        confusion_matrix[1][0] += 1  # False Negative
    
    return confusion_matrix

def unificate_labels(labels):
    """
    Unify bounding boxes from multiple frames into a single list of labels.

    This function consolidates bounding boxes from different frames by applying Non-Maximum Suppression (NMS)
    or other processing to eliminate redundant boxes and generate a final list of labels.

    Parameters:
    -----------
    labels : List[List[Tuple[int, np.ndarray]]]
        A list of lists, where each inner list contains tuples of class labels and corresponding bounding boxes.
        Each tuple consists of:
        - class_label: The class label (int or str).
        - bounding_box: The bounding box represented as an array (e.g., [x_min, y_min, x_max, y_max]).

    Returns:
    --------
    List[Tuple[int, Tuple[float, float, float, float]]]
        A unified list of labels where each entry is a tuple containing:
        - class_label: The class label (int or str).
        - bounding_box: The corresponding bounding box as a tuple (x_min, y_min, x_max, y_max).
    """
    
    final_label = []
    
    for i in range(len(labels[0])):  # Use the length of the first sublist
        # Collect the bounding boxes at position i from all frames
        
        classes_at_pos = [label[i][0] for label in labels if len(label) > i]  # Extract the class
        boxes_at_pos = [label[i][1] for label in labels if len(label) > i]  # Extract the bounding box
        
        if not boxes_at_pos:
            print(f"No boxes found at position {i}, skipping.")
            continue  # Skip if no valid boxes at this position
        
        # Apply NMS or any other processing function to the bounding boxes
        nms_boxes = general_nms_python(boxes_at_pos, boxes_at_pos) 
        
        for box in nms_boxes:
            # Find the original index of this box in the boxes_at_pos list using array comparison
            for j, original_box in enumerate(boxes_at_pos):
                if np.array_equal(box, original_box):  # Use np.array_equal to compare arrays
                    class_label = classes_at_pos[j]  # Get the corresponding class
                    final_label.append((class_label, tuple(box)))  # Combine class and box
                    break  # Exit the loop once the match is found
    return final_label

def general_nms_python(yolo_boxes, alg_boxes, threshold = 0.00001):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    This function takes two sets of bounding boxes and applies NMS to eliminate redundant boxes 
    based on their overlap (IoU). The function returns the filtered bounding boxes that have the 
    highest areas while minimizing overlap.

    Parameters:
    -----------
    yolo_boxes : List[np.ndarray]
        A list of bounding boxes from the YOLO detection model, where each box is represented as 
        an array (e.g., [x_min, y_min, x_max, y_max]).
        
    alg_boxes : List[np.ndarray]
        A list of bounding boxes from another detection algorithm, similarly represented.
        
    threshold : float, optional
        The IoU threshold used to determine which boxes to keep. Boxes with an IoU above this threshold 
        will be suppressed. Default is 0.00001.

    Returns:
    --------
    np.ndarray
        An array of filtered bounding boxes after applying NMS. Each box is represented as an array 
        of shape [x_min, y_min, x_max, y_max].
    """
    
    bboxes = concatenate_boxes(yolo_boxes, alg_boxes)
    
    #Unstacking Bounding Box Coordinates
    bboxes = np.array(bboxes).astype('float')

    # Filter out all boxes that are [0, 0, 0, 0]
    bboxes = bboxes[~np.all(bboxes == np.array([0, 0, 0, 0]), axis=1)]
    
    # Check if there are any boxes left after filtering
    if len(bboxes) == 0:
        return np.array([])  # Return an empty array if no boxes left
    
    
    x_min = bboxes[:,0]
    y_min = bboxes[:,1]
    x_max = bboxes[:,2]
    y_max = bboxes[:,3]
    
    # Calculating areas of all bboxes (area = (width * height))
    bbox_areas = (x_max - x_min + 1) * (y_max - y_min + 1)
    
    # sorted_idx = np.arange(len(bboxes))
    
    # Sorting the bboxes by area in descending order  and keeping respective indices.
    sorted_idx = bbox_areas.argsort()[::-1]
        
    #list to keep filtered bboxes.
    filtered = []
    while len(sorted_idx) > 0:
        #Keeping highest pscore bbox as reference.
        rbbox_i = sorted_idx[0]
        #Appending the reference bbox index to filtered list.
        filtered.append(rbbox_i)
        
        #Calculating (xmin,ymin,xmax,ymax) coordinates of all bboxes w.r.t to reference bbox
        overlap_xmins = np.maximum(x_min[rbbox_i],x_min[sorted_idx[1:]])
        overlap_ymins = np.maximum(y_min[rbbox_i],y_min[sorted_idx[1:]])
        overlap_xmaxs = np.minimum(x_max[rbbox_i],x_max[sorted_idx[1:]])
        overlap_ymaxs = np.minimum(y_max[rbbox_i],y_max[sorted_idx[1:]])
        
        #Calculating overlap bbox widths,heights and there by areas.
        overlap_widths = np.maximum(0,(overlap_xmaxs-overlap_xmins+1))
        overlap_heights = np.maximum(0,(overlap_ymaxs-overlap_ymins+1))
        overlap_areas = overlap_widths*overlap_heights
        
        #Calculating IOUs for all bboxes except reference bbox
        ious = overlap_areas / (bbox_areas[rbbox_i] + bbox_areas[sorted_idx[1:]] - overlap_areas)
        
        #select indices for which IOU is greather than threshold
        delete_idx = np.where(ious > threshold)[0]+1
        delete_idx = np.concatenate(([0],delete_idx))
        
        #delete the above indices
        sorted_idx = np.delete(sorted_idx,delete_idx)
        
    
    #Return filtered bboxes
    return bboxes[filtered].astype('int')
 

def concatenate_boxes(*boxes):
    """
    Concatenate multiple lists of bounding boxes into a single array.

    This function takes any number of bounding box lists and combines them into a single numpy 
    array. Each bounding box is expected to have four coordinates.

    Parameters:
    -----------
    *boxes : List[List[np.ndarray]]
        One or more lists containing bounding boxes. Each bounding box should be represented as 
        an array of four elements (e.g., [x_min, y_min, x_max, y_max]).

    Returns:
    --------
    np.ndarray
        A concatenated array of bounding boxes, with each box represented as a row in the array.
        If no boxes are provided, an empty array with shape (0, 4) will be returned.
    """
    
    concatenated_boxes = []

    for box in boxes:
        # Handle case where box is [] or [[]]
        if isinstance(box, list) and len(box) == 0:
            box = np.empty((0, 4))  # Convert [] to an empty array with 4 columns
        elif isinstance(box, list) and len(box) == 1 and len(box[0]) == 0:
            box = np.empty((0, 4))  # Convert [[]] to an empty array with 4 columns

        # Convert to numpy array if not already
        box = np.array(box).astype(int)
        
        if box.size > 0:
            if box.ndim == 1:  # If box is a 1D array, reshape to 2D
                box = box.reshape(1, -1)
        else:
            box = np.empty((0, 4))  # Empty array with 4 columns
        
        concatenated_boxes.append(box)
    
    # Concatenate all arrays along axis 0
    result = np.concatenate(concatenated_boxes, axis=0)
    
    return result
        
def read_labels(label_file, image):
    """
    Read bounding box labels from a file and convert them to pixel coordinates.

    This function reads object detection labels from a text file and converts the normalized 
    coordinates (center x, center y, width, height) into pixel coordinates for the provided image.

    Parameters:
    -----------
    label_file : str
        The path to the label file containing bounding box annotations. Each line in the file 
        should have the format: class_id x_center y_center width height.
        
    image : np.ndarray
        The image array for which the labels are to be read, used to determine the dimensions of the image.

    Returns:
    --------
    List[Tuple[float, Tuple[int, int, int, int]]]
        A list of tuples, where each tuple contains:
        - class_id: The class ID of the detected object (float).
        - bounding_box: A tuple of pixel coordinates (x1, y1, x2, y2) representing the bounding box.
    """
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    height, width = image.shape[:2]
    labels = []
    
    for line in lines:
        class_id, x_center, y_center, w, h = map(float, line.strip().split())
        x_center *= width
        y_center *= height
        w *= width
        h *= height
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
        labels.append((class_id, (x1, y1, x2, y2)))
    
    return labels


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    This function computes the IoU, which is the ratio of the area of intersection 
    to the area of the union of two bounding boxes. The boxes can be either single 
    or multiple bounding boxes represented in the format [x1, y1, x2, y2].

    Parameters:
    -----------
    box1 : np.ndarray
        A numpy array representing the first bounding box or a set of bounding boxes.
        Each box should be in the format [x1, y1, x2, y2].

    box2 : np.ndarray
        A numpy array representing the second bounding box or a set of bounding boxes, 
        also in the format [x1, y1, x2, y2].

    eps : float, optional
        A small constant added to the denominator for numerical stability to prevent 
        division by zero. Default is 1e-7.

    Returns:
    --------
    np.ndarray
        A numpy array containing the IoU values for each pair of boxes. The shape of 
        the output will be (N, M), where N is the number of boxes in `box1` and M 
        is the number of boxes in `box2`.

    Raises:
    -------
    ValueError
        If either `box1` or `box2` does not have exactly 4 coordinates.
    """
    # Ensure input is at least 2D (N, 4) by reshaping if necessary
    box1 = np.atleast_2d(box1)
    box2 = np.atleast_2d(box2)

    # Check if box1 and box2 have the correct number of columns (4)
    if box1.shape[1] != 4 or box2.shape[1] != 4:
        raise ValueError("Each box should have 4 coordinates (x1, y1, x2, y2)")
        
    
    # Split boxes into coordinates
    x1_1, y1_1, x2_1, y2_1 = np.split(box1, 4, axis=1)
    x1_2, y1_2, x2_2, y2_2 = np.split(box2, 4, axis=1)
    
    # Compute intersection box coordinates
    inter_x1 = np.maximum(x1_1, np.transpose(x1_2))
    inter_y1 = np.maximum(y1_1, np.transpose(y1_2))
    inter_x2 = np.minimum(x2_1, np.transpose(x2_2))
    inter_y2 = np.minimum(y2_1, np.transpose(y2_2))
    
    # Compute intersection area
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    
    # Compute areas of both sets of boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Compute IoU
    iou = inter_area / (area1 + np.transpose(area2) - inter_area + eps)
    
    return iou

def print_confusion_matrix_one_class(confusion_matrix):
    """
    Print a confusion matrix for a single class in a formatted table.

    This function displays the confusion matrix for a binary classification task,
    showing True Positives (TP), False Positives (FP), False Negatives (FN), 
    and True Negatives (TN) in a readable format.

    Parameters:
    -----------
    confusion_matrix : np.ndarray
        A 2x2 numpy array representing the confusion matrix, structured as follows:
        [[TN, FP],
         [FN, TP]]

    Returns:
    --------
    None
        This function prints the confusion matrix to the console.

    """
    
    # Ensure confusion_matrix is a 2x2 matrix of integers
    # Define headers for the confusion matrix table
    headers = ["",  f"{Fore.YELLOW}Predicted Positive{Style.RESET_ALL}",  f"{Fore.YELLOW}Predicted Negative{Style.RESET_ALL}"]
    
    # Prepare the table rows
    table = [
        [ f"{Fore.YELLOW}Actual Positive",  f"{Fore.GREEN}{confusion_matrix[1][1]} (TP){Style.RESET_ALL}", f"{Fore.RED}{confusion_matrix[1][0]} (FN){Style.RESET_ALL}"],
        [ f"{Fore.YELLOW}Actual Negative", f"{Fore.RED}{confusion_matrix[0][1]} (FP){Style.RESET_ALL}", f"{Fore.GREEN}{confusion_matrix[0][0]} (TN){Style.RESET_ALL}"],
    ]
    
    print("\n")
    # Print the confusion matrix using tabulate
    print(tabulate(table, headers, tablefmt="rounded_grid"))
    print("\n")


def calculate_background_variance_without_detection(images, bbox):
    """
    Calculate the variance of background pixels outside a given bounding box.

    This function computes the variance of pixel values in the background of 
    a set of images, excluding the area defined by the provided bounding box.

    Parameters:
    -----------
    images : List[np.ndarray]
        A list of images (as numpy arrays) from which the background variance will be calculated.

    bbox : np.ndarray
        A bounding box defined as an array with four coordinates [x1, y1, x2, y2] 
        indicating the area to be excluded from the variance calculation.

    Returns:
    --------
    List[float]
        A list of variance values calculated for each image. Each entry corresponds to 
        the variance of the background pixels for the respective image.
    """
    
    x1, y1, x2, y2 = bbox.astype(int)
    variances = []
    
    for image in images:
        # Create a mask that includes only the pixels outside the bounding box
        mask = np.ones(image.shape[:2], dtype=bool)
        mask[y1:y2, x1:x2] = False  # Set the bounding box region to False

        # Extract only the pixels outside the bounding box
        background_pixels = image[mask]

        # Calculate variance and mean once
        bg_variance = np.var(background_pixels)
        bg_mean = np.mean(background_pixels)
        
        variances.append(bg_variance / bg_mean)
      
    return variances

def calculate_frame_variance_without_detection(images, bbox, margin):
    """
    Calculate the variance of pixels in the surrounding frame of a bounding box 
    while excluding the area defined by the bounding box itself.

    This function computes the variance of pixel values outside a specified bounding 
    box region, within a surrounding frame defined by a margin. The calculated 
    variance helps analyze the background characteristics around detected objects.

    Parameters:
    -----------
    images : List[np.ndarray]
        A list of images (as numpy arrays) from which the variance will be calculated.

    bbox : np.ndarray
        A bounding box defined as an array with four coordinates [x1, y1, x2, y2] 
        indicating the area of interest to be excluded from the variance calculation.

    margin : int
        The margin around the bounding box to include in the variance calculation. 
        It defines the size of the surrounding frame from which the variance will 
        be computed.

    Returns:
    --------
    List[float]
        A list of variance values calculated for each image. Each entry corresponds to 
        the variance of the pixels outside the bounding box region for the respective image.
    """
    
    x1, y1, x2, y2 = bbox.astype(int)
    variances = []
    img_height, img_width = images[0].shape[:2]  # Calculate once, assume all images have the same dimensions
    
    for image in images:
        # Extract surrounding frame with margin, using max/min to avoid index errors
        surrounding_frame_image = image[max(0, y1-margin):min(img_height, y2+margin),
                                        max(0, x1-margin):min(img_width, x2+margin)]
        
        # Create a mask that excludes the bounding box region from the surrounding frame
        mask = np.ones(surrounding_frame_image.shape, dtype=bool)
        
        # Calculate relative coordinates within the margin region
        focus_relative_y_start = margin
        focus_relative_y_end = margin + (y2 - y1)
        focus_relative_x_start = margin
        focus_relative_x_end = margin + (x2 - x1)
        
        # Apply the mask to exclude the bounding box region from the surrounding frame
        mask[focus_relative_y_start:focus_relative_y_end, focus_relative_x_start:focus_relative_x_end] = False
        
        # Extract the pixels outside the bounding box
        frame_pixels = surrounding_frame_image[mask]
        
        # Calculate variance and mean once
        frame_variance = np.var(frame_pixels)
        frame_mean = np.mean(frame_pixels)

        variances.append(frame_variance / frame_mean)
    

    return variances

def calculate_variances_in_boxes(images, bbox):
    """
    Calculate the variance of pixels inside a specified bounding box.

    This function computes the variance of pixel values within the defined bounding 
    box region across a list of images. This can be useful for analyzing the 
    characteristics of detected objects.

    Parameters:
    -----------
    images : List[np.ndarray]
        A list of images (as numpy arrays) from which the variance will be calculated.

    bbox : np.ndarray
        A bounding box defined as an array with four coordinates [x1, y1, x2, y2] 
        indicating the area of interest for which the variance calculation will be performed.

    Returns:
    --------
    List[float]
        A list of variance values calculated for each image. Each entry corresponds to 
        the variance of the pixels within the bounding box region for the respective image.
    """
    
    x1, y1, x2, y2 = bbox.astype(int)
    variances = []

    for image in images:
        # Extract only the region inside the bounding box
        box_region = image[y1:y2, x1:x2]
        
        # Calculate variance and mean once
        box_variance = np.var(box_region)
        box_mean = np.mean(box_region)

        variances.append(box_variance / box_mean)
  
    return variances

def check_variance_condition(var_X0, var_X1, alpha):
    """
    Check if the variance condition is satisfied between two sets of variance values.

    This function compares the variance values of two sets of data and counts how many 
    entries in the first exceed a threshold defined by the second  
    multiplied by a factor related to alpha. It returns True if at least half of the 
    comparisons satisfy the condition.

    Parameters:
    -----------
    var_X0 : List[float]
        A list of variance values from the signal.

    var_X1 : List[float]
        A list of variance values from the thermal noise.

    alpha : float
        A scaling factor used to determine the threshold for comparison.

    Returns:
    --------
    bool
        True if at least half of the variance values in `var_X0` exceed the computed 
        threshold based on `var_X1`, False otherwise.
    """
    
    cnt = sum(
        1 for j in range(0, len(var_X0))
        if var_X0[j] > var_X1[j] * (2 / alpha - 1))
    
    return cnt >= np.ceil(len(var_X0) / 2)


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=1, label=None):
    """
    Draw bounding boxes on an image.

    This function takes an image and a list of bounding boxes and draws each box 
    on the image using the specified color and thickness. Optionally, it can also 
    add a label above each box for identification.

    Parameters:
    -----------
    image : np.ndarray
        The image on which the bounding boxes will be drawn. It should be a 
        numpy array representing an image.

    boxes : List[np.ndarray]
        A list of bounding boxes, where each box is defined by its coordinates 
        [x1, y1, x2, y2].

    color : Tuple[int, int, int], optional
        The color of the bounding box in RGB format. Default is green (0, 255, 0).

    thickness : int, optional
        The thickness of the bounding box lines. Default is 1.

    label : str, optional
        An optional label to be drawn above the bounding boxes. If provided, 
        the label will be displayed in the image.

    Returns:
    --------
    None
        This function modifies the input image in place by drawing the bounding boxes 
        and optionally adding labels.
    """
    
    for box in boxes:
        x1, y1, x2, y2 = box
        # Draw rectangle on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        # Optionally add a label
        if label:
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
