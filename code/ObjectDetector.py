import os
import time
import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
    
# General Object Detection Class
class TorchModel:
    def __init__(self, model, optimizer = None, scheduler = None, device=None, model_type = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_type = model_type
        self.device = device if device else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.nms_threshold = 0.01

    def train(self, data_loader, num_epochs, save_path, val_loader=None):
        os.makedirs(f"{save_path}/training_data", exist_ok=True)
        os.makedirs(f"{save_path}/images", exist_ok=True)
        os.makedirs(f"{save_path}/evaluation_data", exist_ok=True)

        best_loss = float('inf')
        best_AP50 = 0
        training_losses = []
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            sum_loss_components = {}
            iteration_count = 0

            self.model.train()
            i = 0
            for images, targets in data_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                # print(loss_dict)
                losses = sum(loss_dict.values())
                epoch_loss += losses.item()

                for key in loss_dict.keys():
                    if key not in sum_loss_components:
                        sum_loss_components[key] = 0.0
                    sum_loss_components[key] += loss_dict[key].item()

                iteration_count += 1

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                if i % 10 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {losses.item()}")
                i += 1

            avg_loss_components = {key: value / iteration_count for key, value in sum_loss_components.items()}

            # Write CSV headers for loss components dynamically based on current losses
            if epoch == 0:
                with open(f"{save_path}/training_data/average_loss_components.csv", "w") as f:
                    loss_keys = avg_loss_components.keys()
                    f.write("epoch," + ",".join([f"avg_{key}" for key in loss_keys]) + "\n")

            # Write averaged loss components to CSV
            with open(f"{save_path}/training_data/average_loss_components.csv", "a") as f:
                loss_component_str = ",".join([f"{avg_loss_components.get(key, 0.0):.4f}" for key in avg_loss_components.keys()])
                f.write(f"{epoch},{loss_component_str}\n")

            avg_train_loss = epoch_loss / len(data_loader)
            training_losses.append(avg_train_loss)

            self.scheduler.step()

            epoch_time = (time.time() - start_time) / 3600
            print(f"Epoch [{epoch}/{num_epochs}] completed in {epoch_time:.2f} hours with avg training loss {avg_train_loss:.4f}")

            if val_loader is not None:
                mAP, AP50 = self._evaluate(epoch, val_loader, save_path, f'./dataset_v3_640/valid/corrected_labels.json')
            
            if epoch == 0:
                with open(f"{save_path}/training_data/epoch_times.csv", "a") as f:
                    f.write("epoch,elapsed time,avg_train_loss,val_mAP,val_AP50\n")

            with open(f"{save_path}/training_data/epoch_times.csv", "a") as f:
                f.write(f"{epoch},{epoch_time:.4f},{avg_train_loss:.4f},{mAP:.4f},{AP50:.4f}\n")
                
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                self._save_loss_model(save_path)

            if AP50 > best_AP50:
                best_AP50 = AP50
                self._save_AP_model(save_path)
                
    def load_weights(self, load_path):
        
        if 'pth' in load_path:
            self.model = torch.load(f"{load_path}", weights_only=False, map_location=torch.device(self.device))
        else:
            import onnxruntime as ort
            self.model = ort.InferenceSession(load_path)
            
    def predict(self, image, conf):
        """
        Perform inference on a single image and return predictions in YOLO format,
        filtered by confidence score.
        """
        # Normalize the image
        image = self._preprocess_image(image)

            
        if self.model_type == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                # print(f"Image device: {image.device}, Model device: {next(self.model.parameters()).device}")

                # Perform inference
                output = self.model([image])[0]
                # print("Inference completed.")
                if not any(tensor.numel() == 0 for tensor in output.values()):
                    # Extract boxes, classes, and scores
                    boxes = np.array([box.int().cpu().numpy() for box in output['boxes']])
                    classes = np.array([int(label.cpu().numpy()) for label in output['labels']])
                    scores = np.array([float(score.cpu().numpy()) for score in output['scores']])
                else:
                    return None, None, None
            
          
        else:
            input_name = self.model.get_inputs()[0].name  # Get input name from model
            output = self.model.run(None, {input_name: image})  # Run inference
            # Check for empty output
            if any(arr.size == 0 for arr in output):
                return None, None, None

            # Extract boxes, classes, and scores
            boxes = output[0].astype(int)
            classes = output[1]
            scores = output[2]
                    
        # Filter predictions based on confidence threshold
        valid_indices = scores >= conf
        boxes = boxes[valid_indices]
        classes = classes[valid_indices]
        scores = scores[valid_indices]

        boxes, classes, scores = boxes.tolist(), classes.tolist(), scores.tolist()
        return boxes, classes, scores
    
    def export_onnx(self, onnx_path):
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)

        # Export the model to ONNX format
        torch.onnx.export(
            self.model,            # Model to export
            dummy_input,           # Example input for tracing
            onnx_path,             # File to save the ONNX model
            export_params=True,    # Store the trained parameter weights
            do_constant_folding=True,  # Fold constant nodes for optimization
            input_names=['input'],    # Input tensor name
            output_names=['boxes', 'labels', 'scores'],  # Custom output names
        )



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

    def _save_loss_model(self, save_path):
        torch.save(self.model.state_dict(), f"{save_path}/best_loss_model_state_dict.pth")
        torch.save(self.model, f"{save_path}/best_loss_model_full.pth")
        
    def _save_AP_model(self, save_path):
        torch.save(self.model.state_dict(), f"{save_path}/best_AP_model_state_dict.pth")
        torch.save(self.model, f"{save_path}/best_AP_model_full.pth")

    def _evaluate(self, epoch, val_loader, save_path, gt_annotations_path):
        self.model.eval()

        results = []
        collage_rows = []
        max_images = 9  # Limit the collage to 9 images
        max_images_per_row = 3  # Number of images per row
        collage_size = 640  # Height of each image in the collage
        black_image = np.zeros((collage_size, collage_size, 3), dtype=np.uint8)  # Placeholder for padding

        with torch.no_grad():
            image_count = 0
            row_images = []  # Temporary holder for images in the current row

            for images, targets in val_loader:
                images = [image.to(self.device) for image in images]
                outputs = self.model(images)

                for j, output in enumerate(outputs):
                    # Parse boxes, labels, and scores
                    boxes = np.array([box.int().cpu().numpy() for box in output['boxes']])
                    labels = np.array([int(label.cpu().numpy()) for label in output['labels']])
                    scores = np.array([float(score.cpu().numpy()) for score in output['scores']])
                    
                    final_boxes, final_classes, final_scores = self._apply_nms(boxes, labels, scores)
                    
                    # Collect results for COCO evaluation (processed for all images)
                    for box, score, label in zip(final_boxes, final_scores, final_classes):
                        results.append({
                            "image_id": targets[j]['image_id'].item(),
                            "category_id": int(label),
                            "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                            "score": float(score)
                        })
                        
                    # Generate collage for first few images
                    if image_count < max_images:
                        img = self._prepare_image_for_collage(images[j], collage_size, final_boxes, final_classes, final_scores)
                        row_images.append(img)

                        # Add row to collage if filled
                        if len(row_images) == max_images_per_row:
                            collage_rows.append(np.hstack(row_images))
                            row_images = []

                        image_count += 1

            # Add remaining images in the row (pad if necessary)
            if row_images and image_count <= max_images:
                while len(row_images) < max_images_per_row:
                    row_images.append(black_image)
                collage_rows.append(np.hstack(row_images))

            # Save collage
            if collage_rows:
                collage = np.vstack(collage_rows)
                collage_save_path = f"{save_path}/images/val_epoch_{epoch}_collage.jpg"
                cv2.imwrite(collage_save_path, collage)
                print(f"Collage saved at: {collage_save_path}")

        # Save detection results for COCO evaluation
        dt_results_path = f"{save_path}/evaluation_data/detections_epoch_{epoch}.json"
        with open(dt_results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Detection results saved at: {dt_results_path}")

        # Skip COCO evaluation if no detections were made
        if len(results) == 0:
            print(f"No detections were made during evaluation of epoch {epoch}. Skipping COCO evaluation.")
            return 0, 0

        # Perform COCO evaluation
        metrics = self._evaluate_with_coco(gt_annotations_path, dt_results_path, save_path, epoch)

        print(f"Epoch [{epoch}] - mAP: {metrics['mAP']:.4f}, AP50: {metrics['AP50']:.4f}")
        
        return metrics['mAP'], metrics['AP50']

    def _prepare_image_for_collage(self, image, collage_size, boxes, labels, scores):
        """
        Prepare a single image for the collage with annotations.
        """
        img = image.cpu().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw detections on the image
        for box, label, score in zip(boxes, labels, scores):
            box = box.astype(int)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(img, f"Class: {label}, Score: {score:.2f}",
                        (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

        # Resize image for collage
        return cv2.resize(img, (collage_size, collage_size))

    def _evaluate_with_coco(self, gt_annotations_path, dt_results_path, save_path, epoch):
        # Load ground truth annotations
        coco_gt = COCO(gt_annotations_path)

        # Load detection results
        coco_dt = coco_gt.loadRes(dt_results_path)
 
        # Initialize COCO evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            'mAP': coco_eval.stats[0],
            'AP50': coco_eval.stats[1],
        }

        return metrics

    def _preprocess_image(self, image):
        """
        Normalize and preprocess the image for inference.
        """
        # Ensure the image is in float32 and normalize to [0, 1]
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        image /= 255.0

        if self.model_type == 'pytorch':
            # Convert image to a PyTorch tensor if not already
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float()

            # If the image is in HWC format, convert to CHW
            if image.ndimension() == 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)

            # Ensure the image is on the same device as the model
            image = image.to(self.device)
        
        else:
            # Ensure the image is in CHW format
            if image.shape[-1] == 3:  # Assuming the image is in HWC format
                image = image.transpose(2, 0, 1)  # Convert from HWC to CHW

            # Add batch dimension if not present
            if image.ndim == 3:
                image = np.expand_dims(image, axis=0)

        return image

        
    