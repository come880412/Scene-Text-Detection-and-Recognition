# Scene-Text-Detection-and-Recognition (Pytorch)
- Competition URL: https://tbrain.trendmicro.com.tw/Competitions/Details/19 (Private 6th place)

# 1.Proposed Method
## The models
Our model comprises two parts: Scene text detection and Scene text recognition. the descriptions of these two models are as follow:

- **Scene text detection** \
We employ YoloV5 [1] to detect the ROI (Region Of Interest) from an image and Resnet50 [2] to implement the ROI transformation. This algorithm transforms the coordinates detected by YoloV5 to the proper location, which fits the text well. YoloV5 can detect all ROIs that might be strings while ROI transformation can make the bbox more fit the region of the string. The visualization result is illustrated below.
<!-- src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/Mosaic2.jpg" width=50% height=50%>|<img  -->



# References
[1] https://github.com/ultralytics/yolov5
[2] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
