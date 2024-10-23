# IFTH
 <p align="justify"> 
This is the official repository of "Integrated Framework for Fire Detection and Identification Using Thermal Imaging" (recently submitted). In this repository, you can find all the annotations discussed in the article, in addition to the images that have been proposed by the team.
</p>

<p align="justify"> 
It should be noted that the FLIR and $M{^3}FD$ images are open access and can be downloaded from the links provided in the paper. The annotations made respect the original name of these images.
</p>

## 💻 Materials
<p align="justify">
All the code necessary to replicate our work is available in this repository. The datasets are available via Mendeley Data. For the Nvidia Jetson Nano, we used the docker image available at: https://docs.ultralytics.com/es/guides/nvidia-jetson/#quick-start-with-docker 
</p>

## 🔧 Dependencies and Installation 
* Python == 3.10.8
* opencv-python-headless == 4.10.0.84
* numpy == 1.26.1
* ultralytics == 8.3.19
* scikit-image == 0.22.0
* scikit-learn == 1.3.2
* tabulate == 0.9.0
* colorama == 0.4.6

## 🚀 Code
<p align="justify">
The functions developed in this work can be found in the code folder. The utils.py file implements basic functions such as the IoU calculation used in this work, as well as the preprocessing routine. Take into account that the FLAME-T dataset must be downloaded and inside the "datasets" folder to run the example.
</p>

<p align="justify">
The test_framework.py file shows an example of thermal anomaly detection and fire identification. The result shows with a red color the class corresponding to thermal anomalies, and with a green color the identified fire.
</p>

<p align="center" width="100%">
    <img width="100%" src="images/output_example.png"> 
</p>

##  BibTeX
<!-- @InProceedings{aa,
    author    = {aa},
    title     = {aa},
    date      = {2024}
} -->

## 📜 License
This project is released under the GPL-3.0 license.

## 📧 Contact
If you have any questions, please email antonio.galvan@ulpgc.es.
