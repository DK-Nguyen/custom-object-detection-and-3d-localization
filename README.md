# Custom object detection and 3D localization
## Overview
A software to automate the process of constructing the synthetic dataset
in COCO format, training state-of-the-art deep neural networks (DNNs) from the framework 
[Detectron2](https://github.com/facebookresearch/detectron2), then doing depth 
estimation for the scene. 
The results are detected objects with corresponding [x, y, z] coordinates relative to the camera.

The whole pipeline could be seen in Figure 1 
![](./images/pipelineV2.png)*Figure 1: Data processing and training pipeline*

An output is shown in Figure 2
![](./images/output.png)*Figure 2: An example output*

## Requirements 
`torch`>=1.8  
`detectron2`

## How to run 
Change the main configurations in the file `config.yaml`.  
Configurations for each process (preparing the dataset, training the models on a dataset, and reconstructing 3D positions) are in the `yaml` files in the directories `dataset`, `dataset_model` and `reconstruct_3d`.  
After you are done with the configurations, run `python main.py`.
