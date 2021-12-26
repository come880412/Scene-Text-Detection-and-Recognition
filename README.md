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
- Random Scale Resize
We found that the sizes of the images in the training data are different. Therefore, if we resize the small image to the large, the features will almost be lost. We apply the random scale resize algorithm to obtain the low-resolution image from the high-resolution image. The visualization results are demonstrated as follows.

| Original image | 72x72 --> 224x224 | 96x96 --> 224x224 | 121x121 --> 224x224 | 146x146 --> 224x224 | 196x196 --> 224x224 |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/Original.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/72_72.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/96_96.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/121_121.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/146_146.png" width=50% height=50%>|<img src="https://github.com/come880412/Scene-Text-Detection-and-Recognition/blob/main/images/data_augmentation/196_196.png" width=50% height=50%>|



# References
[1] https://github.com/ultralytics/yolov5 \
[2] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py \
[3] https://github.com/roatienza/deep-text-recognition-benchmark \
