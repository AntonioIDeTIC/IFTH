from ultralytics import YOLO
import os
import torch
from ultralytics import RTDETR

seed_value = 42

# Set the seed for generating random numbers in PyTorch
torch.manual_seed(seed_value)

# If you are using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.


#
path = os.getcwd()

# model = RTDETR("yolov8n-rtdetr.yaml")
model = YOLO("yolov8n.yaml")
# model = RTDETR("/rtdetr-n.yaml")



# Train the model
model.train(data=path + "...//data.yaml", epochs=100, imgsz = 640, plots=True, save=True, batch = 16, augment = False, single_cls=True)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set 
