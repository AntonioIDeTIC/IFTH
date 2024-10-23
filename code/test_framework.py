import cv2
import numpy as np
import os
import utils
import time
import TA_detector

def main():

    # Get the current working directory
    path = os.getcwd()

    
    model_path = os.path.join('..', "models/yolo11_n/weights/best.pt")
    dataset_path = os.path.join('..', "datasets/FLAME-T/Point E/")


    core_detector = TA_detector.CoreDetector(model_path, conf=0.01)
    secondary_detector = TA_detector.SecondaryDetector()

    # Define the image path in the dataset
    img_path = os.path.join(path, f"{dataset_path}/images/")

    # Load image locations for processing using utility function (split into batches, 20 per batch)
    locations = utils.load_images_by_location(img_path, 20)  # Adjust batch size 'n' if necessary

    # Define the target cameras used in detection
    target_cameras = ["A35", "Tau", "Lepton"]

    # Define ranges for margin and alpha values for each camera using a dictionary. This was optimized using grid search.
    camera_parameters = {
        "Lepton": {
            "margin_range": 10,
            "alpha_range_local": 0.35,
            "alpha_range_global": 0.6
        },
        "A35": {
            "margin_range": 50, 
            "alpha_range_local": 0.25,  
            "alpha_range_global": 0.425  
        },
        "Tau": {
            "margin_range": 160,  
            "alpha_range_local": 0.15,  
            "alpha_range_global": 0.2  
        }
    }


    # Define ranges for margin and alpha values for each camera. This was optimized using grid search.
    # margin_range = [10, 50, 160]
    # alpha_range_local = [0.35, 0.25, 0.15]
    # alpha_range_global = [0.6, 0.425, 0.2]

    # Loop through each target camera for the detection process
    for i, t_c in enumerate(target_cameras):
        selected_parameters = camera_parameters[t_c]

        # Accessing the parameters for the selected camera
        margin = selected_parameters["margin_range"]
        alpha_range_local = selected_parameters["alpha_range_local"]
        alpha_range_global = selected_parameters["alpha_range_global"]

        # Create a list of cameras to exclude from processing, excluding the current target camera
        excluded_cameras = ["A35", "Tau", "Lepton"]
        excluded_cameras.remove(t_c)

        
        # Initialize detection metrics
        true_detections, false_detections = 0, 0

        # Initialize arrays for timing and fire labels
        id_time_arr = []
        fire_label = 0
        cnt = 0
        
        # Loop over batches of image locations
        for batch in locations:
            id_time = []
            # print(batch)
            # Load 8-bit images for processing (resize to 640x640) excluding certain cameras and missing images
            _8bit_images = [
                cv2.resize(cv2.imread(os.path.join(img_path, img), cv2.IMREAD_UNCHANGED), (640, 480), interpolation=cv2.INTER_LINEAR)
                for img in batch
                if "_missing" not in img and excluded_cameras[0] not in img and excluded_cameras[1] not in img
            ]
            

            # Load corresponding 16-bit TIFF images for processing, same resizing, and exclusion logic
            _16bit_images = [
                cv2.resize(cv2.imread(os.path.join(img_path.replace('images', 'tiff_images'), img.replace('.jpeg', '.tiff')), cv2.IMREAD_UNCHANGED), 
                (640, 480), interpolation=cv2.INTER_LINEAR)
                for img in batch
                if "_missing" not in img and excluded_cameras[0] not in img and excluded_cameras[1] not in img
            ]

            # Get corresponding image paths and label files for the batch
            image_paths = [
                os.path.join(img_path, b) 
                for b in batch 
                if "_missing" not in b and excluded_cameras[0] not in b and excluded_cameras[1] not in b
            ]
            
        
            # Replace the 'images' folder with 'labels' and '.jpeg' with '.txt' to find label files
            label_files = [image_path.replace('images', 'labels').replace('.jpeg', '.txt') for image_path in image_paths]
            # Load the labels using the utility function (reads the label files and matches them to images)
            labels = [utils.read_labels(label_file, image) for (label_file, image) in zip(label_files, _8bit_images)]
            
            # Check if labels exist and initialize the fire label counter based on the first batch of labels
            if labels:
                if cnt == 0:
                    for label in labels[0]:
                        if label[0] == 1:  # If the label corresponds to fire
                            fire_label += 1
                            
            # If there are no valid 8-bit or 16-bit images, skip the current batch
            if not _8bit_images or not _16bit_images:
                continue

            # Perform anomaly detection using the CoreDetector (YOLO-based detection model) on 8-bit images
            c_d_boxes, c_d_clss, c_d_probs, c_d_prediction_time, c_d_postprocessing_time, _3d_image = core_detector.detect_anomaly(_8bit_images)
        
            # Perform secondary detection using the SecondaryDetector on 16-bit images
            s_d_boxes, s_d_total_time = secondary_detector.detect_anomaly(batch, _16bit_images, n = 1)
            
            # Apply general non-maximum suppression (NMS) on the detected boxes from both detectors
            final_boxes = utils.general_nms_python(c_d_boxes, s_d_boxes)
            # Create another copy of the image to show boxes after NMS
            nms_img = _3d_image.copy()
            utils.draw_boxes(nms_img, final_boxes, color=(0, 0, 255), label="Thermal anomalie")

            # This should be used with s_d_boxes and c_d_boxes in order to get metrics such as mAP, F1, etc
            # confusion_matrix = utils.verify_detection(final_boxes, unified_label, iou_tresh = 0.5) 
            # print_confusion_matrix_one_class(confusion_matrix)
            # Unify labels from different sources (YOLO, secondary detector) for evaluation
            
            unified_labels = utils.unificate_labels(labels)
            
            # Initialize label class and bounding boxes for ground truth
            label_classes = [label[0] for label in unified_labels]
            label_boxes = [np.array(label[1]) for label in unified_labels]
            
            variance_valid_img = _3d_image.copy()

            # Loop through the final detected bounding boxes
            for j, box in enumerate(final_boxes):
                start = time.time()

                box = box.astype(int)  # Convert bounding box coordinates to integer
                
                # Calculate variances in the bounding boxes (X0 is inside the detection box)
                X0_var = utils.calculate_variances_in_boxes(_16bit_images, box)
                
                # Calculate the local frame variance outside the detection box using a margin
                X1_local_var = utils.calculate_frame_variance_without_detection(_16bit_images, box, margin=margin) 
                X1_global_var = utils.calculate_background_variance_without_detection(_16bit_images, box) 

                # Check if the variance condition holds with a margin alpha value
                if utils.check_variance_condition(X0_var, X1_local_var, alpha=alpha_range_local):
                    id_time.append(time.time() - start)

                    # Track whether a true detection (fire) is found
                    found_true_detection = False
                    
                    # Loop through ground truth label classes and boxes to compare with the detection box
                    for label_cls, label_box in zip(label_classes, label_boxes):
                        iou_value = utils.box_iou(label_box, box)  # Calculate IoU between ground truth and detected box
                        if iou_value >= 0.5 and label_cls == 1:  # Check if IoU is above 0.5 and label is fire (class 1)
                            true_detections += 1
                            found_true_detection = True
                            utils.draw_boxes(variance_valid_img, [box], color=(0, 255, 0), label="Identified fire")
                            break
                        else:
                            utils.draw_boxes(variance_valid_img, [box], color=(0, 0, 255), label="Thermal anomaly")

                    # If no true detection was found, increment false detections
                    if not found_true_detection:
                        false_detections += 1

                else:
                    id_time.append(time.time() - start)
                    utils.draw_boxes(variance_valid_img, [box], color=(0, 0, 255), label="Thermal anomaly")
            
            
            # # Create a blank space (black strip) between the images
            height, width, _ = nms_img.shape  # Get the height of the images
            spacing = np.zeros((height, 40, 3), dtype=np.uint8)  # Create a 20px wide black strip with the same height
            # Concatenate images with the blank space in between
            visual = cv2.hconcat([nms_img, spacing, variance_valid_img])  # Merge with spacing
            # Show the final merged window
            cv2.imshow("Detections", visual)

            while True:
                # Wait for user input, check if "Q" or "Esc" is pressed
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # Press "Q" to skip to the next image
                    break  # Exit the inner while loop and continue to the next image in the for loop
                
                elif key == 27:  # Press "Esc" to exit the whole program (keycode 27 for "Esc")
                    cv2.destroyAllWindows()
                    exit()  # Exit the entire program

            # Close any remaining OpenCV windows after the loop
            cv2.destroyAllWindows()

            # Append the average identification time for the batch
            id_time_arr.append(np.nanmean(id_time))     

        # Calculate identification accuracy
        acc = true_detections / (true_detections + (fire_label - true_detections) + false_detections + 1e-6)  # Add small constant to avoid division by zero
        print(f"{t_c} camera identification accuracy: {acc:.3f} in {np.nanmean(id_time_arr)} secs")

if __name__ == "__main__":
    main()