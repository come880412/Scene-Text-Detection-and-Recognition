# Scene-Text-Detection-and-Recognition (Pytorch)
- Competition URL: https://tbrain.trendmicro.com.tw/Competitions/Details/19 (Private 6th place)

# 1.Proposed Method
## The models
Our model comprises two parts: Scene text detection and Scene text recognition. the descriptions of these two models are as follow:

- **Scene text detection** \
We employ YoloV5 [1] to detect the ROI (Region Of Interest) from an image and Resnet50 [2] to implement the ROI transformation. This algorithm transforms the coordinates detected by YoloV5 to the proper location, which fits the text well. YoloV5 can detect all ROIs that might be strings while ROI transformation can make the bbox more fit the region of the string. The visualization result is illustrated below, where the bbox of the dark green is ROI detected by YoloV5 and the bbox of the red is ROI after ROI transformation.
<p align="center">
<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/ROI_transformation_visualization.png" width=50% height=50%>
</p>

- **Scene text recognition** \
We employ ViT [3] to recognize the string of bbox detected by YoloV5 since our task is not a single text recognition. The transformer-based model achieves the state-of-the-art performance in Natural Language Processing (NLP). In this task, we predict the string according to the given image. Therefore, through the attention mechanism, we can make the model pay attention to the words that need to be output at the moment. The model architecture is demonstrated below.
<p align="center">
<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/ViT.jpg" width=50% height=50%>
</p>

The whole training process is shown in the figure below.
<p align="center">
<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/Training_procedure.jpg" width=20% height=20%>
</p>

## Data augmentation
- **Random Scale Resize** \
We found that the sizes of the images in the training data are different. Therefore, if we resize the small image to the large, the features will almost be lost. We apply the random scale resize algorithm to obtain the low-resolution image from the high-resolution image. The visualization results are demonstrated as follows.

| Original image | 72x72 --> 224x224 | 96x96 --> 224x224 | 121x121 --> 224x224 | 146x146 --> 224x224 | 196x196 --> 224x224 |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/Original.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/72_72.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/96_96.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/121_121.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/146_146.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/196_196.png" width=50% height=50%>|

- **ColorJitter** \
In the training phase, the model's input is RGB channel. To enhance the reliability of the model, we appply the collorjitter algorithm to make the model see the images with different contrast, brightness, saturation and hue value. And this kind of method is also widely used in image classification. The visualization results are demonstrated as follows.

| Input image | brightness=0.5 | contrast=0.5 | saturation=0.5 | hue=0.5 | brightness=0.5  contrast=0.5  saturation=0.5  hue=0.5 |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/Original.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/brightness.png" width=35% height=35%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/contrast.png" width=40% height=40%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/saturation.png" width=37.5% height=37.5%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/hue.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/colorjitter.png" width=27.5% height=27.5%>|

- **Random Rotaion** \
After we observe the training data, we find that most of the images in training data are square-shaped (original image), while some of the testing data is a little skewed. The visualization results are demonstrated as follows.

| Original image | Random Rotation | Random Horizontal Flip | Both |
|:----------:|:----------:|:----------:|:----------:|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/Original.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/Random_Rotation.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/Random_Horizontal_Flip.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/Random_Rotation_Horizontal.png" width=50% height=50%>|

# Demo
- **Predicted results** \
Before we recognize the string bbox detected by YoloV5, we filter out the bbox with a size less than 45\*45. Because the image resolution of a bbox with a size less than 45\*45 is too low to recognize the correct string.

| Input image | Scene Text detection | Scene Text recognition |
|:----------:|:----------:|:----------|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/Scene_Text_Detection/yolov5-master/example/img_21009.jpg" width=60% height=60%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/detection/img_21009.jpg" width=60% height=60%>|驗車<br>委託汽車代檢<br>元力汽車公司<br>新竹區監理所|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/Scene_Text_Detection/yolov5-master/example/img_21017.jpg" width=60% height=60%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/detection/img_21017.jpg" width=60% height=60%>|3c配件<br>玻璃貼<br>專業包膜|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/Scene_Text_Detection/yolov5-master/example/img_21026.jpg" width=60% height=60%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/detection/img_21026.jpg" width=60% height=60%>|台灣大哥大<br>myfone<br>新店中正<br>加盟門市|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/Scene_Text_Detection/yolov5-master/example/img_21030.jpg" width=60% height=60%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/detection/img_21030.jpg" width=60% height=60%>|西門町<br>楊<br>排骨酥麵<br>非常感謝<br>tvbs食尚玩家<br>蘋果日報<br>壹週刊<br>財訊<br>錢櫃雜誌<br>聯合報<br>飛碟電台<br>等報導<br>排骨酥專賣店<br>西門町<br>楊<br>排骨酥麵<br>排骨酥麵<br>嘉義店|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/Scene_Text_Detection/yolov5-master/example/img_21023.jpg" width=60% height=60%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/detection/img_21023.jpg" width=60% height=60%>|永晟<br>電動工具行<br>492913338|

- **Attention maps of ViT** \
We also visualize the attention maps in ViT, to check whether the model focus on the correct location of the image. The visualization results are demonstrated as follows.


| Original image | Attention map |
|:--------------------:|:--------------------:|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/ori_img_1.png" width=70% height=70%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/att_img_1.png" width=70% height=70%>|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/ori_img_2.png" width=70% height=70%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/att_img_2.png" width=70% height=70%>|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/ori_img_3.png" width=70% height=70%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/att_img_3.png" width=70% height=70%>|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/ori_img_4.png" width=70% height=70%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/att_img_4.png" width=70% height=70%>|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/ori_img_6.png" width=20% height=20%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/att_img_6.png" width=25% height=25%>|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/ori_img_7.png" width=60% height=60%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/att_img_7.png" width=70% height=70%>|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/ori_img_8.png" width=60% height=60%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/att_img_8.png" width=70% height=70%>|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/ori_img_10.png" width=60% height=60%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/att_img_10.png" width=70% height=70%>|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/ori_img_11.png" width=60% height=60%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/att_map/att_img_11.png" width=70% height=70%>|

# Competition Results
- Public Scores \
We coduct extensive experiments, The results are demonstrated below. From the results, we can see the improvement of the results by adding each module at each stage. At first, we only employ YoloV5 to detect all the ROI of images, and the detection result is not good enough. We also compare the result of ViT with data augmentation or not, the results show that our data augmentation is effective to solve this task (compare the last row and the sixth row). In addition, we filter out the bbox with a size less than 45\*45 since the resolution of bbox is too low to recognize the correct strings.

| Models(Detection/Recognition) | Final score | Precision | Recall |
|:----------:|:----------:|:----------:|:----------:|
| YoloV5(L) / ViT(aug) |                                                                    0.60926|       0.7794|       0.9084|
| YoloV5(L) + <br>ROI_transformation(Resnet50) / ViT(aug)    |                              0.73148|       0.9261|       0.9017|
| YoloV5(L) + <br>ROI_transformation(Resnet50) + <br>reduce overlap bbox / ViT(aug)|        0.78254|       0.9324|       0.9072|
| YoloV5(L) + <br>ROI_transformation(SEResnet50) + <br>reduce overlap bbox / ViT(aug)|      0.78527|       0.9324|       0.9072|
| YoloV5(L) + <br>ROI_transformation(SEResnet50) + <br>reduce overlap bbox / ViT(aug) + filter bbox(40 \* 40)|      0.79373|       0.9333|       0.9029|
| YoloV5(L) + <br>ROI_transformation(SEResnet50) + <br>reduce overlap bbox / ViT(aug) + filter bbox(45 \* 45)|      **0.79466**|       **0.9335**|       **0.9011**|
| YoloV5(L) + <br>ROI_transformation(SEResnet50) + <br>reduce overlap bbox / ViT(aug) + filter bbox(50 \* 50)|      0.79431|       0.9338|       0.8991|
| YoloV5(L) + <br>ROI_transformation(SEResnet50) + <br>reduce overlap bbox / ViT(no aug) + filter bbox(45 \* 45)|      0.73802|       0.9335|       0.9011|

- Private Scores

| Models(Detection/Recognition) | Final score | Precision | Recall |
|:----------:|:----------:|:----------:|:----------:|
| YoloV5(L) + <br>ROI_transformation(SEResnet50) + <br>reduce overlap bbox / ViT(aug) + filter bbox(40 \* 40)|      0.7828|       0.9328|       0.8919|
| YoloV5(L) + <br>ROI_transformation(SEResnet50) + <br>reduce overlap bbox / ViT(aug) + filter bbox(45 \* 45)|      **0.7833**|       **0.9323**|       **0.8968**|
| YoloV5(L) + <br>ROI_transformation(SEResnet50) + <br>reduce overlap bbox / ViT(aug) + filter bbox(50 \* 50)|      0.7830|       0.9325|       0.8944|

# 4.Computer Equipment
- System: Windows10、Ubuntu20.04
- Pytorch version: Pytorch 1.7 or higher
- Python version: Python 3.6
- Testing:  
CPU: AMR Ryzen 7 4800H with Radeon Graphics
RAM: 32GB  
GPU: NVIDIA GeForce RTX 1660Ti 6GB  

- Training:  
CPU: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz  
RAM: 256GB  
GPU: NVIDIA GeForce RTX 3090 24GB * 2

# Getting Started
- Clone this repo to your local
``` bash
git clone https://github.com/come880412/Scene-Text-Detection-and-Recognition.git
cd Scene-Text-Detection-and-Recognition
```
### Download pretrained models
- **Scene Text Detection** \
Please download pretrained models from [Scene_Text_Detection](https://drive.google.com/drive/folders/1MJmNxNav8SFE83jlbe5snycYQ7j2tbcn?usp=sharing). There are three folders, "ROI_transformation", "yolo_models" and "yolo_weight". First, please put the weights in "ROI_transformation" to the path `./Scene_Text_Detection/Tranform_card/models/`. Second, please put all the models in "yolo_models" to the current path. Finally, please put the weight in "yolo_weight" to the path `./Scene_Text_Detection/yolov5-master/runs/train/expl/weights/`.

- **Scene Text Recogniton** \
Please download pretrained models from [Scene_Text_Recognition](https://drive.google.com/drive/folders/1DBO-L-00EA00rAZgV1dIBLhR-Q4kOmmA?usp=sharing). There are two files in this foler, "best_accuracy.pth" and "character.txt". Please put the files to the path `./Scene_Text_Recogtion/saved_models/`.

### Inference
- You should first download the pretrained models and change your path to `./Scene_Text_Detection/yolov5-master/`
```bash
$ python3 Text_detection.py
```
- The result will be saved in the path `'../output/'`. Where the folder "example" is the images detected by YoloV5 and after ROI transformation, the file "example.csv" records the coordinates of the bbox, starting from the upper left corner of the coordinates clockwise, respectively (x1, y1), (x2, y2), (x3, y3), and (x4, y4), and the file "exmaple_45.csv" is the predicted result.
- If you would like to visualize the bbox detected by yoloV5, you can use the function `public_crop()` in the script `../../data_process.py` to extract the bbox from images.

### Training
- You should first download the dataset provided by [official](https://tbrain.trendmicro.com.tw/Competitions/Details/19), then put the data in the path `'./dataset/'`. After that, you could use the following script to transform the original data to the training format.
```bash
$ python3 data_process.py
```
- Scene_Text_Detection \
There are two models for the Scene_Text_Detection task, ROI transformation and YoloV5. You could use the follow script to train these two models.
```bash
$ cd ./Scene_Text_Detection/yolov5-master # YoloV5
$ python3 train.py

$ cd ../Tranform_card/ # ROI Transformation
$ python3 Trainer.py
```

- Scene_Text_Recognition
```bash
$ cd ./Scene_Text_Recogtion # ViT for text recognition
$ python3 train.py
```

# References
[1] https://github.com/ultralytics/yolov5 \
[2] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py \
[3] https://github.com/roatienza/deep-text-recognition-benchmark \
[4] https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/ \
[5] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).  
