import numpy as np
import time
from skimage import measure
import torch
import cv2
from ultralytics import YOLO, RTDETR
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import ObjectDetector

# Main class for performing thermal anomaly detection
class CoreDetector:
    """
    A class used for detecting thermal anomalies using the object detection model and applying 
    Non-Maximum Suppression (NMS) to refine the detected bounding boxes.
    
    Attributes
    ----------
    model : Object Detection model
        The model used for object detection.
    conf : float
        The confidence threshold for detection.
    nms_threshold : float
        The IoU threshold for applying NMS to remove overlapping boxes.

    Methods
    -------
    detect_anomaly(image_batch)
        Detects anomalies in the input image(s) using obj detection and NMS.
    _process_image(image_batch)
        Preprocesses the input image, performs MIP operator for batch images, and normalizes 16-bit images.
    _ultralytics_detection(image)
        Performs object detection using YOLO and returns the bounding boxes, classes, and probabilities.
    _torch_detection(image)
        Performs object detection using torch implementation and returns the bounding boxes, classes, and probabilities.
    _apply_nms(bboxes, classes, pscores)
        Applies Non-Maximum Suppression (NMS) to refine bounding boxes based on IoU threshold.
    """
    
    def __init__(self, model_path, model_id, conf, nms_threshold=0.00001):
        """
        Initializes the CoreDetector class with the model, confidence threshold, and NMS IoU threshold.
        
        Parameters
        ----------
        model_path : str
            The path to the model weights.
        conf : float
            The confidence threshold for object detection.
        nms_threshold : float, optional
            The IoU threshold for Non-Maximum Suppression, by default 0.00001.
        """

        self.model_id = model_id
        print(self.model_id)
        if self.model_id == 'YOLO':
            self.model = YOLO(model_path) # YOLO CASE
        elif self.model_id == 'RTDETR' or self.model_id == 'YOLO-RTDETR':
            self.model = RTDETR(model_path) # RTDETR CASE or yolov8n-rtdetr CASE
        else:
            ################## Faster RCNN ################################
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights = None, num_classes=2, weights_backbone = None)
            # get the number of input features 
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # define a new head for the detector with required number of classes
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
            self.model = ObjectDetector.TorchModel(self.model , model_type = 'pytorch')
            self.model.load_weights(model_path)

        
        self.conf = conf
        self.nms_threshold = nms_threshold

    def detect_thermal_anomaly(self, image_batch):
        """
        Detects thermal anomalies in the input image(s) using the model and applies Non-Maximum Suppression (NMS).
        
        Parameters
        ----------
        image_batch : list of np.ndarray or np.ndarray
            A batch of images (list of np.ndarray) or a single image (np.ndarray) to detect anomalies in.
        
        Returns
        -------
        final_boxes : np.ndarray
            The final bounding boxes after applying NMS.
        final_classes : np.ndarray
            The detected classes for each bounding box.
        final_scores : np.ndarray
            The confidence scores for each bounding box.
        prediction_time : float
            The time taken by the YOLO model to perform detection.
        postprocessing_time : float
            The time taken to apply postprocessing such as NMS.
        image : np.ndarray
            The preprocessed input image used for detection.
        """
        preprocessing_start = time.time()

        # Preprocess image
        image = self._process_image(image_batch)

  
        if self.model_id == 'YOLO' or self.model_id == 'RTDETR' or self.model_id == 'YOLO-RTDETR':
            boxes, clss, probs, prediction_time = self._ultralytics_detection(image)
        else:
            boxes, clss, probs, prediction_time = self._torch_detection(image)



        # Apply Non-Maximum Suppression (NMS)
        final_boxes, final_classes, final_scores = self._apply_nms(boxes, clss, probs)

        postprocessing_time = (time.time() - preprocessing_start) - prediction_time

        return final_boxes, final_classes, final_scores, prediction_time, postprocessing_time, image

    def _process_image(self, image_batch):
        """
        Processes the input image by performing the Maximum Intensity Projection (MIP) operator if it's a batch of images.
        Converts 16-bit images to 8-bit using cv2.normalize and ensures a 3-channel format if necessary.
        
        Parameters
        ----------
        image_batch : list of np.ndarray or np.ndarray
            A batch of images (list of np.ndarray) or a single image (np.ndarray) to be processed.
        
        Returns
        -------
        image : np.ndarray
            The preprocessed image, normalized and converted to 3-channel format if necessary.
        """
        if isinstance(image_batch, list) and all(isinstance(i, np.ndarray) for i in image_batch):
            image_stack = np.stack(image_batch, axis=0)
            image = np.max(image_stack, axis=0)  # MIP operator
        else:
            image = image_batch
            
        # Check if the image is 16-bit and normalize it to 8-bit if necessary
        if image.dtype == np.uint16:  # Check if image is 16-bit
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if image.shape[-1] != 3:
            # Stack grayscale image to create a 3-channel image
            # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))            
            # image = clahe.apply(image)
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        return image

    def _ultralytics_detection(self, image):
        """
        Performs object detection using the YOLO or RTDETR model on the input image.
        
        Parameters
        ----------
        image : np.ndarray
            The input image to perform detection on.
        
        Returns
        -------
        boxes : np.ndarray
            The detected bounding boxes in the image.
        clss : np.ndarray
            The detected classes for each bounding box.
        probs : np.ndarray
            The confidence scores for each bounding box.
        prediction_time : float
            The time taken by the model to perform the detection.
        """
        start = time.time()
        result = self.model.predict(image, imgsz=640, conf=self.conf, verbose=False)[0]
        prediction_time = time.time() - start

        if result:
            boxes = result.boxes.xyxy.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            probs = result.boxes.conf.cpu().numpy()
        else:
            boxes, clss, probs = None, None, None

        return boxes, clss, probs, prediction_time

    
    def _torch_detection(self, image):
        """
        Performs object detection using the torch model on the input image.
        
        Parameters
        ----------
        image : np.ndarray
            The input image to perform detection on.
        
        Returns
        -------
        boxes : np.ndarray
            The detected bounding boxes in the image.
        clss : np.ndarray
            The detected classes for each bounding box.
        probs : np.ndarray
            The confidence scores for each bounding box.
        prediction_time : float
            The time taken by the model to perform the detection.
        """
        start = time.time()
        boxes, clss, probs = self.model.predict(image, conf=self.conf)
        prediction_time = time.time() - start
        
        return boxes, clss, probs, prediction_time

    def _apply_nms(self, bboxes, classes, pscores):
        """
        Applies Non-Maximum Suppression (NMS) to the detected bounding boxes to remove overlapping boxes.
        
        Parameters
        ----------
        bboxes : np.ndarray
            The detected bounding boxes.
        classes : np.ndarray
            The classes of the detected bounding boxes.
        pscores : np.ndarray
            The confidence scores for the detected bounding boxes.
        
        Returns
        -------
        final_bboxes : np.ndarray
            The filtered bounding boxes after applying NMS.
        final_classes : np.ndarray
            The filtered classes after applying NMS.
        final_pscores : np.ndarray
            The filtered confidence scores after applying NMS.
        """
        if bboxes is None or (isinstance(bboxes, np.ndarray) and np.all(bboxes == None)):
            return np.array([]), np.array([0]), np.array([0])
    
        # Convert to float numpy arrays
        bboxes = np.array(bboxes).astype('float')
        pscores = np.array(pscores).astype('float')
        classes = np.array(classes).astype('float')

        x_min = bboxes[:, 0]
        y_min = bboxes[:, 1]
        x_max = bboxes[:, 2]
        y_max = bboxes[:, 3]

        # Sort by confidence scores (pscores) in descending order
        sorted_idx = pscores.argsort()[::-1]
        bbox_areas = (x_max - x_min + 1) * (y_max - y_min + 1)

        filtered = []
        while len(sorted_idx) > 0:
            rbbox_i = sorted_idx[0]
            filtered.append(rbbox_i)

            # Calculate overlaps (IoU) with remaining boxes
            overlap_xmins = np.maximum(x_min[rbbox_i], x_min[sorted_idx[1:]])
            overlap_ymins = np.maximum(y_min[rbbox_i], y_min[sorted_idx[1:]])
            overlap_xmaxs = np.minimum(x_max[rbbox_i], x_max[sorted_idx[1:]])
            overlap_ymaxs = np.minimum(y_max[rbbox_i], y_max[sorted_idx[1:]])

            overlap_widths = np.maximum(0, (overlap_xmaxs - overlap_xmins + 1))
            overlap_heights = np.maximum(0, (overlap_ymaxs - overlap_ymins + 1))
            overlap_areas = overlap_widths * overlap_heights

            # Intersection over Union (IoU) calculation
            ious = overlap_areas / (bbox_areas[rbbox_i] + bbox_areas[sorted_idx[1:]] - overlap_areas)

            # Filter out boxes with IoU greater than threshold
            delete_idx = np.where(ious > self.nms_threshold)[0] + 1
            delete_idx = np.concatenate(([0], delete_idx))

            # Remove selected indices
            sorted_idx = np.delete(sorted_idx, delete_idx)

        # Return filtered bounding boxes, classes, and scores
        return bboxes[filtered].astype('int'), classes[filtered].astype('int'), pscores[filtered]


# Class for secondary detection processing
class SecondaryDetector:
    """
    A class used for performing secondary detection processing on thermal images.
    
    Methods
    -------
    detect_anomaly(image_batch_files, images, n)
        Detects anomalies in the input image batch by processing images and applying thresholding.
    _determine_threshold(image_batch_files)
        Determines the maximum threshold based on the type of image batch provided.
    _compute_regions(image_batch_files, binary_image, original_image, n)
        Computes the top `n` regions with maximum pixel values in the original image based on the binary image.
    _temporal_processing(images)
        Processes a stack of images using temporal processing to generate a composite image.
    """
    
    def detect_thermal_anomaly(self, image_batch_files, images, n):
        """
        Detects thermal anomalies in the input image batch by preprocessing images, 
        applying thresholding, and computing the top regions.

        Parameters
        ----------
        image_batch_files : list of str
            A list of file paths for the input image batch.
        images : list of np.ndarray
            A list of input images to process.
        n : int
            The number of top regions to return based on pixel intensity.

        Returns
        -------
        boxes : np.ndarray
            The bounding boxes of detected regions.
        processing_time : float
            The total time taken for preprocessing and detection.
        """
        # Step 1: Preprocessing
        preprocessing_start = time.time()
        composite_image = self._temporal_processing_np(images)
        
        # Step 2: Thresholding based on image batch
        max_pixel_value = np.max(composite_image)
        threshold_value = self._determine_threshold(image_batch_files[0]) * max_pixel_value
        
        _, binary_img = cv2.threshold(composite_image, threshold_value, 255, cv2.THRESH_BINARY)
        binary_img = np.asarray(binary_img, dtype=np.uint8)
        preprocessing_time = time.time() - preprocessing_start

        # Step 3: Detection
        detection_start = time.time()
        boxes = self._compute_regions(image_batch_files[0], binary_img, composite_image, n)
        detection_time = time.time() - detection_start
        
        return boxes, preprocessing_time + detection_time

    def _determine_threshold(self, image_batch_files):
        """
        Determines the maximum threshold based on the image batch type.

        Parameters
        ----------
        image_batch_files : str
            The filename or identifier of the image batch.

        Returns
        -------
        float
            The determined threshold value for the specific image batch type.
        """
        if "Tau" in image_batch_files:
            return 0.4
        elif "A35" in image_batch_files:
            return 0.1
        else:
            return 0.1
    
    def _compute_regions(self, image_batch_files, binary_image, original_image, n):
        """
        Computes the top `n` regions with maximum pixel values in the original image.
        Applies area restrictions based on the image batch type.

        Parameters
        ----------
        image_batch_files : str
            The filename or identifier of the image batch.
        binary_image : np.ndarray
            The binary image obtained from thresholding the composite image.
        original_image : np.ndarray
            The original image from which regions are computed.
        n : int
            The number of top regions to return.

        Returns
        -------
        np.ndarray
            An array of bounding boxes corresponding to the detected regions.
        """
        if len(binary_image.shape) == 3 and binary_image.shape[2] == 3:
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        
        if np.std(binary_image) == 0:
            return np.array([])

        labeled_image = measure.label(binary_image, connectivity=2)
        regions = measure.regionprops(labeled_image, intensity_image=original_image)
        
        if not regions:
            return np.array([])

        # Determine area constraints based on image batch
        if "Tau" in image_batch_files:
            max_area_allowed, min_area_allowed = 4096, 2
        elif "A35" in image_batch_files:
            max_area_allowed, min_area_allowed = 512, 2
        else:
            max_area_allowed, min_area_allowed = 128, 4
        
        regions_list = [
            (np.max(region.intensity_image), region.bbox)
            for region in regions
            if min_area_allowed <= region.area <= max_area_allowed
        ]
        
        regions_list = sorted(regions_list, key=lambda x: x[0], reverse=True)
        if len(regions_list) < n:
            regions_list += [(0, (0, 0, 0, 0))] * (n - len(regions_list))

        top_n_regions = regions_list[:n]
        bounding_boxes = [[min_col, min_row, max_col, max_row] 
                          for _, (min_row, min_col, max_row, max_col) in top_n_regions]
        
        return np.array(bounding_boxes)

    def _temporal_processing_np(self, images):
        """
        Equivalent implementation for _temporal_processing using only NumPy.
        """
        # Convert images to a numpy array (ensure float32 for consistency)
        images = np.asarray(images, dtype=np.float32)
        
        weight_nad = 0.55
        weight_variance = 0.25
    
        # weight_nad, weight_variance = weights
    
        if weight_variance == 0:
            # nad_images = []
            if images.shape[0] > 1 and weight_nad > 0:
                # Compute absolute differences between consecutive images along axis 0
                nad_diff = np.abs(np.diff(images, axis=0))
                # for i in range(len(images) - 1):
                #     nad_diff = np.abs(images[i] - images[i + 1])
                #     nad_images.append(nad_diff)
                max_nad_image = np.max(nad_diff, axis=0)
            else:
                max_nad_image = np.std(images, axis=0)
            combined_image = weight_nad * max_nad_image
    
        elif weight_nad == 0:
            variance_image = np.var(images, axis=0)
            combined_image = weight_variance * variance_image
            
        else:
            # nad_images = []
            if images.shape[0] > 1 and weight_nad > 0:
                # Compute absolute differences between consecutive images along axis 0
                nad_diff = np.abs(np.diff(images, axis=0))
                # for i in range(len(images) - 1):
                #     nad_diff = np.abs(images[i] - images[i + 1])
                #     nad_images.append(nad_diff)
                max_nad_image = np.max(nad_diff, axis=0)
            else:
                max_nad_image = np.std(images, axis=0)
            variance_image = np.var(images, axis=0)
            combined_image = weight_nad * max_nad_image + weight_variance * variance_image
    
        # Clip combined image to the valid range.
        combined_image = np.clip(combined_image, 0, 65535)
        max_image = np.max(images, axis=0)
        final_img = np.clip(0.5 * (max_image + combined_image), 0, 65535)
        
        return final_img.astype(np.uint16)

    def _temporal_processing(self, images):
        """
        Processes a stack of images using temporal processing with PyTorch,
        allowing CUDA optimization if available. Returns a composite image
        combining Normalized Absolute Differences (NAD), variance, and standard deviation.

        Parameters
        ----------
        images : list of np.ndarray
            A list of input images to be processed.

        Returns
        -------
        np.ndarray
            The composite image obtained from the temporal processing of the input images.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = np.asarray(images, dtype=np.float32)
        image_stack = torch.stack([torch.from_numpy(img) for img in images], dim=0).float().to(device)

        # Compute Normalized Absolute Differences (NAD)
        nad_images = [
            torch.abs(image_stack[i] - image_stack[i + 1]) 
            for i in range(len(images) - 1)
        ]
        max_nad_image = torch.max(torch.stack(nad_images), dim=0)[0] if nad_images else torch.std(image_stack, dim=0)

        variance_image = torch.var(image_stack, dim=0)

        # Weighted combination of processed images
        weight_nad = 0.75
        weight_variance = 0.25
        
        combined_image = (
            weight_nad * max_nad_image +
            weight_variance * variance_image 
        )

        combined_image = torch.clamp(combined_image, 0, 65535).type(torch.float32)
        max_image = torch.max(image_stack, dim=0)[0].type(torch.float32)

        final_img = (max_image * 0.5 + combined_image * 0.5).type(torch.float32).cpu().numpy().astype(np.uint16)

        return final_img
