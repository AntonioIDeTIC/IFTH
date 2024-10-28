import albumentations as A
import cv2
import os
import yaml
import pybboxes as pbx
import random

with open("contants.yaml", 'r') as stream:
    CONSTANTS = yaml.safe_load(stream)


def is_image_by_extension(file_name):
    """
    Check if the given file has a recognized image extension.

    Args:
        file_name (str): Name of the file.

    Returns:
        bool: True if the file has a recognized image extension, False otherwise.

    """
    # List of common image extensions
    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']
    # Get the file extension
    file_extension = file_name.lower().split('.')[-1]
    # Check if the file has a recognized image extension
    return file_extension in image_extensions


def get_inp_data(img_file, i):
    """
    Get input data for image processing.

    Args:
        img_file (str): Name of the input image file.

    Returns:
        tuple: A tuple containing the image, ground truth bounding boxes, and augmented file name.

    """
    file_name = os.path.splitext(img_file)[0]
    aug_file_name = f"{file_name}_{i}_{CONSTANTS['transformed_file_name']}"
    image = cv2.imread(os.path.join(CONSTANTS["inp_img_pth"], img_file))
    lab_pth = os.path.join(CONSTANTS["inp_lab_pth"], f"{file_name}.txt")
    gt_bboxes = get_bboxes_list(lab_pth, CONSTANTS['CLASSES'])
    return image, gt_bboxes, aug_file_name


def get_album_bb_list(yolo_bbox, class_names):
    """
    Extracts bounding box information for a single object from YOLO format.

    Args:
        yolo_bbox (str): YOLO format string representing bounding box information.
        class_names (list): List of class names corresponding to class numbers.

    Returns:
        list: A list containing [x_center, y_center, width, height, class_name].
    """
    str_bbox_list = yolo_bbox.split()
    class_number = int(str_bbox_list[0])
    class_name = class_names[class_number]
    bbox_values = list(map(float, str_bbox_list[1:]))
    album_bb = bbox_values + [class_name]
    return album_bb


def get_album_bb_lists(yolo_str_labels, classes):
    """
    Extracts bounding box information for multiple objects from YOLO format.

    Args:
        yolo_str_labels (str): YOLO format string containing bounding box information for multiple objects.
        classes (list): List of class names corresponding to class numbers.

    Returns:
        list: A list of lists, each containing [x_center, y_center, width, height, class_name].
    """
    album_bb_lists = []
    yolo_list_labels = yolo_str_labels.split('\n')
    for yolo_str_label in yolo_list_labels:
        if yolo_str_label:
            album_bb_list = get_album_bb_list(yolo_str_label, classes)
            album_bb_lists.append(album_bb_list)
    return album_bb_lists


def get_bboxes_list(inp_lab_pth, classes):
    """
    Reads YOLO format labels from a file and returns bounding box information.

    Args:
        inp_lab_pth (str): Path to the YOLO format labels file.
        classes (list): List of class names corresponding to class numbers.

    Returns:
        list: A list of lists, each containing [x_center, y_center, width, height, class_name].
    """
    yolo_str_labels = open(inp_lab_pth, "r").read()

    if not yolo_str_labels:
        print("No object")
        return []

    lines = [line.strip() for line in yolo_str_labels.split("\n") if line.strip()]
    album_bb_lists = get_album_bb_lists("\n".join(lines), classes) if len(lines) > 1 else [get_album_bb_list("\n".join(lines), classes)]

    return album_bb_lists


def single_obj_bb_yolo_conversion(transformed_bboxes, class_names):
    """
    Convert bounding boxes for a single object to YOLO format.

    Parameters:
    - transformed_bboxes (list): Bounding box coordinates and class name.
    - class_names (list): List of class names.

    Returns:
    - list: Bounding box coordinates in YOLO format.
    """
    if transformed_bboxes:
        class_num = class_names.index(transformed_bboxes[-1])
        bboxes = list(transformed_bboxes)[:-1]
        bboxes.insert(0, class_num)
    else:
        bboxes = []
    return bboxes


def multi_obj_bb_yolo_conversion(aug_labs, class_names):
    """
    Convert bounding boxes for multiple objects to YOLO format.

    Parameters:
    - aug_labs (list): List of bounding box coordinates and class names.
    - class_names (list): List of class names.

    Returns:
    - list: List of bounding box coordinates in YOLO format for each object.
    """
    yolo_labels = [single_obj_bb_yolo_conversion(aug_lab, class_names) for aug_lab in aug_labs]
    return yolo_labels


def save_aug_lab(transformed_bboxes, lab_pth, lab_name):
    """
    Save augmented bounding boxes to a label file.

    Args:
        transformed_bboxes (list): List of augmented bounding boxes.
        lab_pth (str): Path to the output label directory.
        lab_name (str): Name of the label file.

    """
    lab_out_pth = os.path.join(lab_pth, lab_name)
    with open(lab_out_pth, 'w') as output:
        for bbox in transformed_bboxes:
            updated_bbox = str(bbox).replace(',', ' ').replace('[', '').replace(']', '')
            output.write(updated_bbox + '\n')


def save_aug_image(transformed_image, out_img_pth, img_name):
    """
    Save augmented image to an output directory.

    Args:
        transformed_image (numpy.ndarray): Augmented image.
        out_img_pth (str): Path to the output image directory.
        img_name (str): Name of the image file.

    """
    out_img_path = os.path.join(out_img_pth, img_name)
    cv2.imwrite(out_img_path, transformed_image)


def draw_yolo(image, labels, file_name):
    """
    Draw bounding boxes on an image based on YOLO format.

    Args:
        image (numpy.ndarray): Input image.
        labels (list): List of bounding boxes in YOLO format.

    """
    H, W = image.shape[:2]
    for label in labels:
        yolo_normalized = label[1:]
        box_voc = pbx.convert_bbox(tuple(yolo_normalized), from_type="yolo", to_type="voc", image_size=(W, H))
        cv2.rectangle(image, (box_voc[0], box_voc[1]),
                      (box_voc[2], box_voc[3]), (0, 0, 255), 1)
    cv2.imwrite(f"bb_image/{file_name}.jpeg", image)
    # cv2.imshow(f"{file_name}.png", image)
    # cv2.waitKey(0)

def get_pixel_augmented(image, pixel_tf):
    """
    Apply pixel-level augmentations to an input image.

    Parameters:
    - image (numpy.ndarray): Input image.
    - pixel_tf (list): List of pixel-level augmentations.

    Returns:
    - numpy.ndarray: Augmented image.
    """
    pixel_transform = A.Compose(pixel_tf)
    transformed_image = pixel_transform(image=image)['image']
    return transformed_image

def get_spatial_augmented(image, bboxes, spatial_tf):
    """
    Apply spatial augmentations to an input image and bounding boxes.

    Parameters:
    - image (numpy.ndarray): Input image.
    - bboxes (list): List of bounding boxes in YOLO format [x_center, y_center, width, height, class_name].
    - spatial_tf (list): List of spatial augmentations.

    Returns:
    - tuple: A tuple containing the augmented image and the transformed bounding boxes.
    """
    spatial_transform = A.Compose(spatial_tf, bbox_params=A.BboxParams(format='yolo'))
    transformed = spatial_transform(image=image, bboxes=bboxes)
    transformed_image, transformed_bboxes = transformed['image'], transformed['bboxes']
    return transformed_image, transformed_bboxes

# Example usage
def get_augmented_results(image, bboxes, pixel_tf, spatial_tf):
    """
    Apply data augmentation to an input image and bounding boxes.

    Parameters:
    - image (numpy.ndarray): Input image.
    - bboxes (list): List of bounding boxes in YOLO format [x_center, y_center, width, height, class_name].
    - pixel_tf (list): List of pixel-level augmentations.
    - spatial_tf (list): List of spatial augmentations.

    Returns:
    - tuple: A tuple containing the augmented image and the transformed bounding boxes.
    """
    image = get_pixel_augmented(image, pixel_tf)
    image, bboxes = get_spatial_augmented(image, bboxes, spatial_tf)
    return image, bboxes

# Define pixel-level augmentations
def get_pixel_transformations():
    blur_transformations = [
        A.Blur(blur_limit=3, p = 1.0),
        A.MedianBlur(blur_limit=3, p = 1.0),
        A.GaussianBlur(blur_limit=(1, 3), p = 1.0)
    ]
    noise_transformations = [
        A.GaussNoise(var_limit=(10.0, 15.0), mean=0, per_channel=True, p = 1.0),
        A.GlassBlur(sigma=0.7, max_delta=1.0, iterations=2, mode='fast', p = 1.0)
    ]
    
     # Randomly choose to apply blur or noise
    if random.uniform(0, 1) >= 0.5:
        selected_transformation = random.choice(blur_transformations)
    else:
        selected_transformation = random.choice(noise_transformations)
        
    pixel_tf = [
        selected_transformation,
        A.RandomGamma(gamma_limit=(80, 120), p = 0.5),
        A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(8, 8), p = 0.5),
        A.ImageCompression(quality_lower=75, p = 0.5)
    ]
    return pixel_tf

# Define spatial augmentations
def get_spatial_transformations(image):
    rotation_transformations = [
        A.Rotate(limit=(-90, 90), interpolation=1, border_mode=4, p = 1.0),
        A.ShiftScaleRotate(shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=(-90, 90), interpolation=1, border_mode=4, p = 1.0),
        # A.Affine (scale=None, translate_percent=None, translate_px=None, rotate=None, shear=(-15, 15), interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=False, rotate_method='largest_box', balanced_scale=False, p = 1.0)
    ]
    # crop_transformations = [
    #     A.RandomCrop(image.shape[0] // 2, image.shape[1] // 2, p = 1.0),
    #     A.CenterCrop(image.shape[0] // 2, image.shape[1] // 2, p = 1.0),
    # ]
    
    geo_transformations = [
        A.HorizontalFlip(p = 1.0),
        A.VerticalFlip(p = 1.0),
        A.D4(p = 1.0)
    ]
    spatial_tf = [
        random.choice(rotation_transformations),
        #random.choice(crop_transformations),
        random.choice(geo_transformations),
        #A.Downscale(scale_min=None, scale_max=None, scale_range=(0.75, 0.75), interpolation_pair={'upscale': cv2.INTER_NEAREST, 'downscale': cv2.INTER_LINEAR}, p=0.25),
        A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False, interpolation=1, p=0.5),
        A.Resize(image.shape[0], image.shape[1])
    ]
    return spatial_tf

def has_negative_element(matrix):
    """
    Check if there is a negative element in the 2D list of augmented bounding boxes.

    Args:
        matrix (list[list]): The 2D list.

    Returns:
        bool: True if a negative element is found, False otherwise.

    """
    return any(element < 0 for row in matrix for element in row)


def save_augmentation(trans_image, trans_bboxes, trans_file_name):
    """
    Saves the augmented label and image if no negative elements are found in the transformed bounding boxes.

    Parameters:
        trans_image (numpy.ndarray): The augmented image.
        trans_bboxes (list): The transformed bounding boxes.
        trans_file_name (str): The name for the augmented output.

    Returns:
        None
    """
    tot_objs = len(trans_bboxes)
    if tot_objs:
        # Convert bounding boxes to YOLO format
        trans_bboxes = multi_obj_bb_yolo_conversion(trans_bboxes, CONSTANTS['CLASSES']) if tot_objs > 1 else [single_obj_bb_yolo_conversion(trans_bboxes[0], CONSTANTS['CLASSES'])]
        if not has_negative_element(trans_bboxes):
            # Save augmented label and image
            save_aug_lab(trans_bboxes, CONSTANTS["out_lab_pth"], trans_file_name + ".txt")
            save_aug_image(trans_image, CONSTANTS["out_img_pth"], trans_file_name + ".jpeg")
            # Draw bounding boxes on the augmented image
            draw_yolo(trans_image, trans_bboxes, trans_file_name)
        else:
            print("Found Negative element in Transformed Bounding Box...")
    else:
        print("Label file is empty")
