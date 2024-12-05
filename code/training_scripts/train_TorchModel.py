import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from pycocotools.coco import COCO
import torchinfo
import ObjectDetector


def summary(model):
    # Torchvision Faster RCNN models are enclosed within a tuple ().
    if type(model) == tuple:
        model = model[0]
    device = 'cpu'
    batch_size = 4
    channels = 3
    img_height = 640
    img_width = 640
    torchinfo.summary(
        model, 
        device=device, 
        input_size=[batch_size, channels, img_height, img_width],
        row_settings=["var_names"]
    )
    

# Custom COCO Dataset class
def get_transform():
    transforms = [T.ToTensor()]
    return T.Compose(transforms)

class COCODataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(sorted(self.coco.getCatIds()))}

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path)

        boxes, labels = [], []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            xmax, ymax = xmin + w, ymin + h
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.cat_id_to_label[ann['category_id']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([img_id])}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

# Example Usage
if __name__ == "__main__":
    import time
    train_imgs_path = './dataset/train/images'
    valid_imgs_path = './dataset/valid/images'
    train_ann_file = './dataset/train/corrected_labels.json'
    valid_ann_file = './dataset/valid/corrected_labels.json'

    dataset = COCODataset(train_imgs_path, train_ann_file, get_transform())
    dataset_test = COCODataset(valid_imgs_path, valid_ann_file, get_transform())

    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    data_loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
   
    # dataset.ids = dataset.ids[:600]
    # dataset_test.ids = dataset_test.ids[:150]
    
    num_classes = len(dataset.cat_id_to_label) + 1
    
    ################### Faster RCNN ################################

    model = fasterrcnn_mobilenet_v3_large_fpn(weights = None, num_classes=num_classes, weights_backbone = None,  trainable_backbone_layers = 6)
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    
    optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.001,
    weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(data_loader),  
        epochs=20
    )
    

    model_path = './FasterRCNN_FPN_MobileNet'

    detector = ObjectDetector.ObjectDetector(model, optimizer, lr_scheduler)

    detector.train(data_loader, num_epochs=100, save_path=model_path, val_loader=data_loader_test)
        
        