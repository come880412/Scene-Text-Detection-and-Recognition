'''
Modified Date: 2021/10/23

Author: Li-Wei Hsiao

mail: nfsmw308@gmail.com
'''
import argparse, cv2, time, json, csv, copy, math
from pathlib import Path
from PIL import Image
import numpy as np
from numpy import random
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
import torchvision
import tqdm
import torch.nn.functional as F


import os

from ViT.model import Model
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, public_data
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

class SELayerX(nn.Module):
    
    def __init__(self, inplanes):
        super(SELayerX, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, int(inplanes / 16), kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(int(inplanes / 16), inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out

class SEBottleneckX(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(SEBottleneckX, self).__init__()
        score = 2 
        self.conv1 = nn.Conv2d(inplanes, planes * score, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * score)

        self.conv2 = nn.Conv2d(planes * score, planes * score, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * score)
        
        self.conv3 = nn.Conv2d(planes * score, planes * score*2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * score*2)
        self.selayer = SELayerX(planes * score*2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.selayer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEResNeXt(nn.Module):
    def __init__(self, block, layers, cardinality=32, num_classes=1000):
        super(SEResNeXt, self).__init__()
        self.cardinality = cardinality
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        layer4 = self.layer4(x)

        x = self.avgpool(layer4)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

class TokenLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, opt):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.SPACE = '[s]'
        self.GO = '[GO]'
        #self.MASK = '[MASK]'

        #self.list_token = [self.GO, self.SPACE, self.MASK]
        self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + list(opt.character)

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = opt.batch_max_length + len(self.list_token)

    def encode(self, text):
        """ convert text-label into text-index.
        """
        length = [len(s) + len(self.list_token) for s in text]  # +2 for [GO] and [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            txt = [self.GO] + list(t) + [self.SPACE]
            txt = [self.dict[char] for char in txt]
            batch_text[i][:len(txt)] = torch.LongTensor(txt)  # batch_text[:, 0] = [GO] token
        return batch_text.cuda()

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

def Text_recognition(opt):
    opt.character_dir = os.path.join(opt.ViT_load, 'character.txt')
    output_csv = f'./{opt.output_path}/example_{opt.filter_bbox}.csv'
    save_csv = f'{opt.output_path}/example.csv'

    """Prepare the character list"""
    opt.character = ''
    with open(opt.character_dir, encoding='utf-8') as f:
        data = f.read().splitlines()
        for line in data:
            opt.character = opt.character + line

    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)

    ViT = torch.nn.DataParallel(Model(opt)).cuda()
    ViT.load_state_dict(torch.load(os.path.join(opt.ViT_load, 'best_accuracy.pth')))
    ViT.eval()

    public_dataset = public_data(opt.data_path, save_csv, opt.filter_bbox)
    public_loader = torch.utils.data.DataLoader(
            public_dataset, batch_size=32,
            shuffle=False, 
            num_workers=4)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        with torch.no_grad():
            for idx,(image, save) in enumerate(tqdm.tqdm(public_loader)):
                image = image.cuda()
                batch_size = image.shape[0]

                preds = ViT(image, seqlen=converter.batch_max_length)
                
                _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
                preds_index = preds_index.view(-1, converter.batch_max_length)
                length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).cuda()
                preds_str = converter.decode(preds_index[:, 1:], length_for_pred)

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                confidence_score_list = []
                save = np.array(save).T
                for idx, (pred, pred_max_prob) in enumerate(zip(preds_str, preds_max_prob)):

                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]
                    try:
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    except:
                        confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                    confidence_score_list.append(confidence_score)
                    pred = '###' if pred == '' else pred
                    writer.writerow([save[idx][0], save[idx][1], save[idx][2], save[idx][3], save[idx][4], save[idx][5], save[idx][6], save[idx][7], save[idx][8], pred])

def detect():
    source, weights, view_img, save_txt, imgsz = opt.data_path, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    save_dir = f'{opt.output_path}/example'
    save_csv = f'{opt.output_path}/example.csv'
    os.makedirs(save_dir, exist_ok=True)

    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # object detection (yolov5)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # classification
    tranform_cood = torch.load(opt.ROI_transformation)
    tranform_cood.cuda().eval()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    eval_save_txt = []
    eval_save_csv = []

    for path, img, im0s, vid_cap in dataset:
        draw_img = cv2.imread(path)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        
        t1 = time_synchronized()
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = Path(path), '', im0s
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                labels = []
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    xyxy += [names[int(cls)]]
                    labels += [[int(i.cpu().data.numpy()) if type(i) != str else i for i in xyxy ]]

                    # plot rectangle
                    color = colors[int(cls)] or [random.randint(0, 255) for _ in range(3)]
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    # cv2.rectangle(draw_img, c1, c2, (0,0,255), thickness=1, lineType=cv2.LINE_AA)

                # if len(labels) != 1:
                #     if np.std(np.array(labels)[:,0].astype('int')) < np.std(np.array(labels)[:,1].astype('int')):
                #         labels = sorted(labels, key=lambda x:x[1])
                #     else:
                #         labels = sorted(labels, key=lambda x:x[0])

                for idx, xyxy in enumerate( labels ):
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    Xsize = int(im0.shape[1]*0.005)
                    Ysize = int(im0.shape[0]*0.005)
                    Xmin, Ymin, Xmax, Ymax = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    # Xmin = max(Xmin - size, 0)
                    # Xmax = min(Xmax + size, im0.shape[1])
                    # Ymin = max(Ymin - size, 0)
                    # Ymax = min(Ymax + size, im0.shape[1])
                    Xmin = Xmin - Xsize if Xmin - Xsize > 0 else 0
                    Xmax = Xmax + Xsize if Xmax + Xsize < im0.shape[1] else im0.shape[1]
                    Ymin = Ymin - Ysize if Ymin - Ysize > 0 else 0
                    Ymax = Ymax + Ysize if Ymax + Ysize < im0.shape[0] else im0.shape[0]
                    img = im0[Ymin:Ymax, Xmin:Xmax]
                    # cv2.imshow('out',img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    img = Image.fromarray(np.uint8(img))
                    img = torchvision.transforms.Resize((224,224))(img)
                    img = torchvision.transforms.ToTensor()(img)
                    img = torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])(img)
                    img = img.view(-1, img.size(0), img.size(1), img.size(2)).cuda()
                    pred = tranform_cood(img)
                    pred = nn.Sigmoid()(pred)
                    X1, Y1, X2, Y2, X3, Y3, X4, Y4 = pred[0].cpu().data.numpy()
                    out = im0[Ymin:Ymax, Xmin:Xmax].copy()


                    # X1, X4, Y1, Y2 = np.maximum( np.array([X1, X4, Y1, Y2]) - 0.03, 0 )
                    # X2, X3, Y3, Y4 = np.minimum( np.array([X2, X3, Y3, Y4]) + 0.04, 1 )
                    # X1, X4, Y1, Y2 = [0 if i < 0 else i for i in [X1, X4, Y1, Y2]]
                    # X2, X3, Y3, Y4 = [1 if i > 1 else i for i in [X2, X3, Y3, Y4]]

                    # if out.shape[1] < out.shape[0]:
                    #     Y1, Y2 = np.maximum( np.array([Y1, Y2]) - 0.03, 0 )
                    #     Y3, Y4 = np.minimum( np.array([Y3, Y4]) + 0.04, 1 )
                    # else:
                    #     X1, X4 = np.maximum( np.array([X1, X4]) - 0.03, 0 )
                    #     X2, X3 = np.minimum( np.array([X2, X3]) + 0.04, 1 )

                    X1, X4, Y1, Y2 = [0 if i < 0 else i for i in [X1, X4, Y1, Y2]]
                    X2, X3, Y3, Y4 = [1 if i > 1 else i for i in [X2, X3, Y3, Y4]]

                    X1, X2, X3, X4 = (np.array([X1, X2, X3, X4])*out.shape[1]).astype('int')
                    Y1, Y2, Y3, Y4 = (np.array([Y1, Y2, Y3, Y4])*out.shape[0]).astype('int')
                    # cv2.drawContours(out, np.array([ [[[X1, Y1]],[[X2, Y2]],[[X3, Y3]],[[X4, Y4]]] ]), -1, (0,255,0), 1)
                    # cv2.imshow('out',out)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # cv2.imwrite(save_dir + '/' + path.split('\\')[-1][:-4] + "_%s.png"%(str(idx).zfill(2)), out)

                    X1, X2, X3, X4 = np.array([X1, X2, X3, X4]) + int(Xmin)
                    Y1, Y2, Y3, Y4 = np.array([Y1, Y2, Y3, Y4]) + int(Ymin)
                    X1, X2, X3, X4 = [0 if i < 0 else i for i in [X1, X2, X3, X4]]
                    Y1, Y2, Y3, Y4 = [0 if i < 0 else i for i in [Y1, Y2, Y3, Y4]]

                    eval_save_txt.append([path[:-4] + "_%s.png"%(str(idx).zfill(2))])
                    eval_save_csv.append([path[:-4].split('\\')[-1], X1, Y1, X2, Y2, X3, Y3, X4, Y4]) # windows '\\' linux '\'
                    cv2.drawContours(draw_img, np.array([ [[[X1, Y1]],[[X2, Y2]],[[X3, Y3]],[[X4, Y4]]] ]), -1, (0,255,0), 3)
                # cv2.imshow('draw_img',draw_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir, path.split('\\')[-1]), draw_img) # windows '\\' linux '\'
                del(labels)

                t2 = time_synchronized()
                print('%sDone. (%.3fs) ' % (s, t2 - t1))

    np.savetxt(save_csv, eval_save_csv, fmt='%s', delimiter=',', newline='\n', header='', footer='', comments='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument('--weights', nargs='+', type=str, default='./runs/train/expl_String1024/weights/best.pt', help='path to the model of YoloV5')
    parser.add_argument('--ROI_transformation', default='../Tranform_card/models/best_ser502.pkl', help='Path to the model of ROI_transformation ')
    parser.add_argument('--ViT_load', default='../../Scene_classification/saved_models', help='Path to the model of ViT')
    parser.add_argument('--data_path', default="./example", help='Path to the testing images')
    parser.add_argument('--output_path', default="../output", help='Path to the output images and csv')

    # Yolo Setting
    parser.add_argument('--img-size', type=int, default = 1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # ViT Setting
    parser.add_argument('--batch_max_length', type=int, default = 40, help='maximum predict text')
    parser.add_argument('--filter_bbox', type=int, default = 45, help='filter out the bbox less than')
    parser.add_argument('--TransformerModel', default="vitstr_base_patch16_224", help='Which vit/deit transformer model')
    
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
        Text_recognition(opt)