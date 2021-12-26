import numpy as np
import pandas as pd
import copy, tqdm, os, random, cv2, math, glob
import torch
import json
from PIL import Image
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ExponentialLR, StepLR, CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from torch.nn import Parameter

from math import cos, sin, radians

size = 224
train_transforms = transforms.Compose([ 
                                    # transforms.Resize((size,size)),  # (250,250)
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

test_transforms = transforms.Compose([  
                                        # transforms.Resize((size,size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

class TextData(Dataset):
    def __init__(self, path, istrain= False, transform=None):
        self.transform = train_transforms if istrain else test_transforms
        self.istrain = istrain
        imgs = glob.glob(path + '/*.png')[:100]
        txts = [ img[:-4] + '.txt' for img in imgs]
        targets = [ np.loadtxt(txt) for txt in txts]
        self.img_base = [ [img, target[:8]] for img, target in zip(imgs, targets)]

    def __getitem__(self, index):
        path, label = self.img_base[index]
        img = Image.open(path).convert("RGB")
        if self.istrain:
            img = np.array(img)
            if random.random() > 0.5:
                rsize = random.randint(96, 196)
                img = cv2.resize(cv2.resize(img, (rsize, rsize), cv2.INTER_LINEAR), (size, size), cv2.INTER_LINEAR)
            else:
                img = cv2.resize(img, (size, size), cv2.INTER_LINEAR)
            x1, y1, x2, y2, x3, y3, x4, y4 = (label*size).astype('int')
            M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), random.sample([0,90,180,270, random.randint(1, 20)], 1)[0], 1) #第一個參數是旋轉中心，第二個是旋轉角度，第三個是縮放比例
            img = cv2.warpAffine(img, M, img.shape[:2]) # 第三個參數是轉換後圖片大小
            label = np.dot(M, np.array([ [x1, x2, x3, x4], [y1, y2, y3, y4], [1,1,1,1] ])).astype('int')
            label[label < 0 ] = 0
            label[label > size ] = size
            x1, x2, x3, x4, y1, y2, y3, y4 = label.reshape(8)/size
            label = np.array([x1, y1, x2, y2, x3, y3, x4, y4])
            # cv2.drawContours(img, np.array([ [[[x1, y1]],[[x2, y2]],[[x3, y3]],[[x4, y4]]] ]), -1, (0,0,255), 1)
            # cv2.imshow('out', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img = Image.fromarray(np.uint8(img))
        else:
            img = np.array(img)
            img = cv2.resize(img, (size, size), cv2.INTER_LINEAR)
            img = Image.fromarray(np.uint8(img))
        img2 = self.transform(img)
        if self.istrain:
            return img2, label
        else:
            return img2, label, path

    def __len__(self):
        return len(self.img_base)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
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
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
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
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.avgpool(layer4)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)

        embedding = self.fc(x)
        x = self.bn2(embedding)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x, embedding

def resnet50(num_classes, pretrained=False):
    model = ResNet(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes)
    if pretrained:
        dict_ = models.resnet50(pretrained = pretrained)
        dict_ = dict_.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in dict_.items():
            if k in model_dict and k.split('.')[0] != 'fc':
                pretrained_dict.update({k: v})
            else:
                break
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

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
    
class SEBottleneckX101(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(SEBottleneckX101, self).__init__()
        score = 4
        self.conv1 = nn.Conv2d(inplanes, planes * score, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * score)

        self.conv2 = nn.Conv2d(planes * score, planes * score, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * score)
        
        self.conv3 = nn.Conv2d(planes * score, planes * score, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * score)
        self.selayer = SELayerX(planes * score)

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
        self.fc = nn.Linear(512 * block.expansion, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

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

        embedding = self.fc(x)
        x = self.bn2(embedding)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x, embedding

def se_resnext50(num_classes, pretrained=False):
    """Constructs a SE-ResNeXt-50 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNeXt(SEBottleneckX, [3, 4, 6, 3], num_classes = num_classes)
    if pretrained:
        dict_ = models.resnext50_32x4d(pretrained = pretrained)
        dict_ = dict_.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in dict_.items():
            if k in model_dict and k.split('.')[0] != 'fc':
                pretrained_dict.update({k: v})
            else:
                break
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def se_resnext101(num_classes, pretrained=False):
    """Constructs a SE-ResNeXt-101 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNeXt(SEBottleneckX101, [3, 4, 23, 3], num_classes = num_classes)
    if pretrained:
        dict_ = models.resnext101_32x8d(pretrained = pretrained)
        dict_ = dict_.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in dict_.items():
            if k in model_dict and k.split('.')[0] != 'fc':
                pretrained_dict.update({k: v})
            else:
                break
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

if __name__ in "__main__":
    save = './models'
    os.makedirs(save, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    epochs = 120
    batch_size = 32
    optimizerName = 'adam'
    print("Load dataset...")
    train_data = TextData(path = '../../../dataset/TextTranform/train/', istrain = True, transform = True)
    valid_data = TextData(path = '../../../dataset/TextTranform/Val/', istrain = False, transform = True)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 0)
    vaildloader = torch.utils.data.DataLoader(valid_data, batch_size = int(batch_size/2), shuffle = False, num_workers = 0)

    print("Load model...")
    model = models.resnet50(pretrained = True)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 8)
    # model = torch.load('./models/shopee_33.8043_27.pkl')
    
    # model = resnet50(num_classes = len(char_to_index), pretrained = True)
    # model = se_resnext50(num_classes = len(char_to_index), pretrained = True)
    # model = se_resnext101(num_classes = len(char_to_index), pretrained = True)
    # model = torch.load("modelsArcEfficientNetB5Only_sgd/Text_92.7875_51.pkl")
    model = model.cuda()
    
    criterion = torch.nn.SmoothL1Loss()

    if optimizerName == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9, weight_decay = 5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 5e-4)

    # scheduler = StepLR(optimizer, step_size = 80, gamma = 0.1)

    print("Training...")
    running_loss = 0
    loss_min = float('inf')
    for epoch in range(1, epochs):
        model.train()
        if epoch in [30, 80]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * 0.1
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']
        train_bar = tqdm.tqdm(trainloader)
        for idx,(inputs, labels) in enumerate(train_bar):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            logps = model(inputs)
            logps = nn.Sigmoid()(logps)
            loss = criterion(logps*size, labels*size)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_description( 
                            "Epoch: %d/%d  |"%(epoch, epochs) + \
                            "Batch: %s/%d  |"%(str(idx).zfill(len(str(len(trainloader)))), len(trainloader)) +\
                            "lr: %0.6f  |"%(learning_rate) + \
                            "Train loss: %4.5f |"%(running_loss)
                            )

        running_loss = 0
        model.eval()
        with torch.no_grad():
            valid_loss=0
            valid_bar = tqdm.tqdm(vaildloader)
            for idx,(inputs, labels, _) in enumerate(valid_bar):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                logps = nn.Sigmoid()(logps)
                batch_loss = criterion(logps*size, labels*size)
                valid_loss += batch_loss.item()
                
                _, pred = torch.max(logps, 1)
                valid_bar.set_description('[Test]  Loss: {:.4f}'.format(valid_loss))
            
            if valid_loss < loss_min:
                torch.save(model,'%s/best.pkl'%(save))
                # torch.save(model.state_dict(),'%s/Text_%.4f_%d.pkl'%('./models', accuracy, epoch))
                loss_min = copy.copy(valid_loss)
        # scheduler.step()
