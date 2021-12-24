import os
import json
import torchvision as tv
import random

import cv2

from PIL import Image
import numpy as np
from torch.utils.data import Dataset

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl                                                                                                                                                                                                          [1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def read_json(file_name):
    with open(file_name, encoding="utf-8") as handle:
        out = json.load(handle)
    return out

train_transform = tv.transforms.Compose([
    tv.transforms.Resize((224,224)),
    tv.transforms.RandomRotation(random.randint(1, 8)),
    # tv.transforms.GaussianBlur(7,3),
    # tv.transforms.RandomAffine(15),
    tv.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Resize((224,224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class RawDataset(Dataset):

    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.mode = mode
        if mode == 'train':
            self.transform = train_transform
            json_file = read_json(opt.train_data)
        elif mode == 'val':
            self.transform = val_transform
            json_file = read_json(opt.valid_data)
        self.image_path_list = [(os.path.join(opt.data_dir, val['image_id']), val['caption']) for val in json_file['annotations']]
        self.image_path_list = self.image_path_list[:20]
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        image_path, label = self.image_path_list[index]
        
        if self.mode == 'train':
            img = cv2.imread(image_path)
            # cv2.imwrite('./ori.png', cv2.resize(img, (224,224)))
            # cv2.imwrite('./96_96.png', cv2.resize(img, (96,96)))
            img = cv2.resize(img, (random.randint(96, 192),random.randint(96, 192)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        else:
            img = Image.open(image_path).convert('RGB')  # for color image
        img = self.transform(img)

        return (img, label)

class public_data(Dataset):
    def __init__(self, data_path, csv_path, filter_pixel):
        self.transform = val_transform
        public_csv = np.loadtxt(csv_path, delimiter=',', dtype=np.str)
        self.image_list = []
        self.save = []
        for public_data in public_csv:
            image_name = public_data[0]
            image_path = os.path.join(data_path, image_name + '.jpg')

            x1, y1 = int(public_data[1]), int(public_data[2])
            x2, y2 = int(public_data[3]), int(public_data[4])
            x3, y3 = int(public_data[5]), int(public_data[6])
            x4, y4 = int(public_data[7]), int(public_data[8])
            # fliter out small bounding box
            if (x2 - x1) <= filter_pixel and (y4-y1) <= filter_pixel:
                continue

            points = [[x1, y1], [x2,y2], [x3,y3], [x4,y4]]
            pts = np.array(points)
            self.image_list.append([image_path, pts])
            self.save.append([image_name, str(x1), str(y1), str(x2), str(y2), str(x3), str(y3), str(x4), str(y4)])

    def __getitem__(self, index):
        image_path, pts = self.image_list[index]
        image = cv2.imread(image_path)

        cropped = four_point_transform(image, pts)
        # cv2.imwrite('./ori_img.png', cropped)
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped = Image.fromarray(cropped)
        image = self.transform(cropped)

        return image, self.save[index]
    
    def __len__(self):
        return len(self.image_list)
