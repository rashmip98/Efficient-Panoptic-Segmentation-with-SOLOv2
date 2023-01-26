# Location Based Efficient Panoptic Segmentation 

Panoptic segmentation is a scene understanding problem that combines the prediction from both instance and semantic segmentation into a general unified output.
This project implements a location-based panoptic segmentation model, modifying the state-of-the-art EfficientPS architecture by using SOLOv2 as the instance segmentation head instead of a Mask-RCNN.

## System Requirements
* Linux 
* Python 3.7
* PyTorch 1.7
* CUDA 10.2
* GCC 7 or 8

## Dependencies
Install the following frameworks

- [EfficientNet-Pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) for the backbone
- [detectron2](https://github.com/facebookresearch/detectron2) for the instance head
- [In-Place Activated BatchNorm](https://github.com/mapillary/inplace_abn)
- [COCO 2018 Panoptic Segmentation Task API (Beta version)](https://github.com/cocodataset/panopticapi) to compute panoptic quality metric

## Install Dependencies
### For EfficientPS
- Install [Albumentation](https://albumentations.ai/)
```
pip install -U albumentations
```
- Install [Pytorch lighting](https://www.pytorchlightning.ai/)
```
pip install pytorch-lightning
```
- Install [Inplace batchnorm](https://github.com/mapillary/inplace_abn)
```
pip install inplace-abn
```
- Install [EfficientNet Pytorch](https://github.com/lukemelas/EfficientNet-PyTorch)
```
pip install efficientnet_pytorch
```
- Install [Detecron 2 dependencies](https://github.com/facebookresearch/detectron2)
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
- Install [Panoptic api](https://github.com/cocodataset/panopticapi)
```
pip install git+https://github.com/cocodataset/panopticapi.git
```
### For SOLOv2
Install the dependencies by running
```
pip install pycocotools
pip install numpy
pip install scipy
pip install torch==1.5.1 torchvision==0.6.1
pip install mmcv
```

## Dataset Preparation

1. Download the GtFine and leftimg8bit files of the Cityscapes dataset from https://www.cityscapes-dataset.com/ and unzip the `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` into `data/cityscapes`
2. The dataset needs to be converted into coco format using the conversion tool in mmdetection:
* Clone the repository using `git clone https://github.com/open-mmlab/mmdetection.git`
* Enter the repository using `cd mmdetection`
* Install cityscapescripts using `pip install cityscapesscripts`
* Run the script as 
```
python tools/dataset_converters/cityscapes.py \
    data/cityscapes/ \
    --nproc 8 \
    --out-dir data/cityscapes/annotations
```
3. Create the panoptic images json file:
* Clone the repository using `git clone https://github.com/mcordts/cityscapesScripts.git`
* Install it using `pip install git+https://github.com/mcordts/cityscapesScripts.git`
* Run the script using `python cityscapesScripts/cityscapesscripts/preparation/createPanopticImgs.py`

Now the folder structure for the dataset should look as follows:
```
EfficientPS
└── data
    └── cityscapes
        ├── annotations
        ├── train
        ├── cityscapes_panoptic_val.json
        └── val
```
## How to train

### SOLOv2
- Go into the SOLOv2 folder using `cd SOLOv2`
- Modify `config.yaml` to change the paths
- Run `python setup.py develop`
- Run `train.py`

### EfficientPS
- Go into the SOLOv2 folder using `cd ..` and `cd EfficientPS`
- Run `train_net.py`

## How to run inference
1. Go into the SOLOv2 folder using `cd SOLOv2`
2. Run `python eval.py`. This will save the SOLOv2 masks in `EfficientPS/solo_outputs`
3. Now go into the EfficientPS folder using `cd ..` and `cd EfficientPS`
4. Run the combined evaluation using `python solo_fusion.py`

The results will be saved in `EfficientPS/Outputs`

### Why SOLOv2?
![image](https://user-images.githubusercontent.com/38180831/203141810-3c0e51b8-7a79-46ff-b0de-532efb184231.png)

## EfficientPS Architecture

The original EfficientPS paper: [here](https://arxiv.org/abs/2004.02307)\
Code from the authors of EfficientPS: [here](https://github.com/DeepSceneSeg/EfficientPS)

![image](https://user-images.githubusercontent.com/38180831/203141883-79fd1093-eb06-4be8-8ddc-b9c8e63a911e.png)

### Why EfficientPS?

Early research explored various techniques for Instance segmentation and Semantic segmentation separately. Initial panoptic segmentation methods heuristically combine predictions from state-of-the-art instance segmentation network and semantic segmentation network in a post-processing step. However, they suffered from large computational overhead, redundancy in learning and discrepancy between the predictions of each network.\
Recent works implemented top-down manner with shared components or in a bottom-up manner sequentially. This again did not utilize component sharing and suffered from low computational efficiency, slow runtimes and subpar results.\
EfficientPS:
- Shared backbone: EfficientNet
- Feature aligning semantic head, modified Mask R-CNN
- Panoptic fusion module: dynamic fusion of logits based on mask confidences
- Jointly optimized end-to-end, Depth-wise separable conv, Leaky ReLU
- 2 way FPN : semantically rich multiscale features

### Novelty of this approach

We replace the Mask-RCNN architecture from the instance head with a SOLOv2 architecture in order to improve the instance segmentation of the EfficientPS model.\
The Mask-RCNN losses now will be replaced by SOLOv2’s Focal Loss for semantic category classification and DiceLoss for mask prediction.\
This approach of using a location-based instance segmentation for panoptic segmentation will improve upon the performance metrics.



## Results
![image](https://user-images.githubusercontent.com/38180831/203146468-67a3e8cc-0a21-493e-be43-26655b6615b3.png)

![image](https://user-images.githubusercontent.com/38180831/203145055-325e047d-db78-437c-b103-bc42593e2c6f.png)
![image](https://user-images.githubusercontent.com/38180831/203145086-789ef0b7-25c7-4269-b468-a5673fecf22f.png)

