from utils import *
import shutil

input_folder = './input'
output_folder = './output'
bb_folder = './bb_image'
 

# Create directories if they do not exist
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(bb_folder, exist_ok=True)
 

# Create subdirectories in train and valid folders
subdirectories = ['images', 'labels']
for subdir in subdirectories:
    os.makedirs(os.path.join(input_folder, subdir), exist_ok=True)
    os.makedirs(os.path.join(output_folder, subdir), exist_ok=True)
    
def run_yolo_augmentor():
    """
    Run the YOLO augmentor on a set of images.

    This function processes each image in the input directory, applies augmentations,
    and saves the augmented images and labels to the output directories.

    """
    imgs = [img for img in os.listdir(CONSTANTS["inp_img_pth"]) if is_image_by_extension(img)]
    print(imgs)
    for img_file in imgs:
        label_file = img_file.replace('.jpeg', '.txt')
        shutil.copy(os.path.join(input_folder, f'images/{img_file}'), os.path.join(output_folder, f'images/{img_file}'))
        shutil.copy(os.path.join(input_folder, f'labels/{label_file}'), os.path.join(output_folder, f'labels/{label_file}'))

    for k in range(1, 6):
        for i in range(1, 4):
            try:
                for img_num, img_file in enumerate(imgs):
                    print(f"{img_num+1}-image is processing...\n")
                    image, gt_bboxes, aug_file_name = get_inp_data(img_file, i)
                    pixel_tf = get_pixel_transformations()
                    spatial_tf = get_spatial_transformations(image)

                    aug_img, aug_label = get_augmented_results(image, gt_bboxes, pixel_tf, spatial_tf)

                    # aug_img, aug_label = get_augmented_results(image, gt_bboxes)
    
                    if len(aug_img) and len(aug_label):
                        save_augmentation(aug_img, aug_label, aug_file_name)
                    
            except:
                pass

if __name__ == "__main__":
    run_yolo_augmentor()