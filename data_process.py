'''
train: ../dataset/train
public: ../dataset/public

資料庫:
https://blog.csdn.net/qq_42246695/article/details/115164584#ICDARReCTS_11
Polygon to-rectangle conversion:
https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
'''
import json, os, random, tqdm, shutil
import numpy as np
import re
import cv2

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def Cal_area_2poly(img, original_bbox, prediction_bbox):
    im = np.zeros(img.shape[:2], dtype = "uint8")
    im1 = np.zeros(img.shape[:2], dtype = "uint8")
    original_grasp_mask = cv2.fillPoly(im, original_bbox.reshape((-1,4,2)), 255)
    prediction_grasp_mask = cv2.fillPoly(im1, prediction_bbox.reshape((-1,4,2)), 255)
    masked_and = cv2.bitwise_and(original_grasp_mask, prediction_grasp_mask, mask=im)
    masked_or = cv2.bitwise_or(original_grasp_mask, prediction_grasp_mask)
    or_area = np.sum(np.float32(np.greater(masked_or, 0))) # 有沒有比0大
    and_area = np.sum(np.float32(np.greater(masked_and,0)))
    IOU = and_area/or_area
    return IOU

def Tranform_train():
    trainJsonPath = '../dataset/train/json/'
    trainImgPath = '../dataset/train/img/'
    trainSample = '../dataset/TextTranform/'
    os.makedirs(trainSample, exist_ok = True)
    json_files = os.listdir(trainJsonPath)
    random.shuffle(json_files)
    for json_files, folder in [(json_files[:int(len(json_files)*0.8)], 'train'), (json_files[int(len(json_files)*0.8):], 'Val')]:
        os.makedirs(trainSample + folder, exist_ok = True)
        for file in tqdm.tqdm(json_files):
            with open(trainJsonPath + file, encoding="utf-8") as f:
                data = json.load(f)
            img = cv2.imread(trainImgPath + file.split('.')[0] + '.jpg')
            
            for idx, value in enumerate(data['shapes']):
                labels = []
                label = value['label']
                points = np.array(value['points'])
                group_id = value['group_id']
                shape_type = value['shape_type']
                # cv2.imwrite( trainSample + folder + '/0/' + file.split('.')[0] + '.png', img)
                if group_id in [0,2,3,4]:
                    x1, y1 = min(points[:, 0]), min(points[:, 1])
                    x3, y3 = max(points[:, 0]), max(points[:, 1])
                    x = (x1 + x3) / 2
                    y = (y1 + y3) / 2
                    w = abs(x1 - x3)
                    h = abs(y1 - y3)
                    x = x/img.shape[1]
                    y = y/img.shape[0]
                    w = w/img.shape[1]
                    h = h/img.shape[0]
                    size = 5
                    x1 = x1 - size if x1 - size > 0 else 0
                    x3 = x3 + size if x3 + size < img.shape[1] else img.shape[1]
                    y1 = y1 - size if y1 - size > 0 else 0
                    y3 = y3 + size if y3 + size < img.shape[0] else img.shape[0]

                    X1, Y1 = points[0]
                    X2, Y2 = points[1]
                    X3, Y3 = points[2]
                    X4, Y4 = points[3]
                    X1, X2, X3, X4 = (X1 - x1, X2 - x1, X3 - x1, X4 - x1)
                    Y1, Y2, Y3, Y4 = (Y1 - y1, Y2 - y1, Y3 - y1, Y4 - y1)
                    X1, X2, X3, X4 = (X1/(x3-x1), X2/(x3-x1), X3/(x3-x1), X4/(x3-x1))
                    Y1, Y2, Y3, Y4 = (Y1/(y3-y1), Y2/(y3-y1), Y3/(y3-y1), Y4/(y3-y1))
                    

                    labels = [[X1, Y1, X2, Y2, X3, Y3, X4, Y4, x, y, w, h, 1]]
                    out = img[y1:y3, x1:x3]
                    # cv2.drawContours(img, np.array([points[:, np.newaxis, :]]), -1, (0,0,255), 3)
                    # cv2.imshow('out',out)
                    # cv2.drawContours(out, np.array([ [[[X1, Y1]],[[X2, Y2]],[[X3, Y3]],[[X4, Y4]]] ]), -1, (0,255,0), 3)
                    # cv2.imshow('out mask',out)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    
                    cv2.imwrite( trainSample + folder + '/' + file.split('.')[0] + f'_{idx}.png', out)
                    labels = np.array(labels)
                    if labels.ndim == 1:
                        labels = labels[np.newaxis, :]
                    np.savetxt(trainSample + folder + '/' + file.split('.')[0] + f'_{idx}.txt', np.array(labels), fmt='%0.5f', delimiter=' ', newline='\n')
                
                elif group_id in [255, 5]:
                    labels = [[0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5, 1, 1, 0]]

                    x1, y1 = min(points[:, 0]), min(points[:, 1])
                    x3, y3 = max(points[:, 0]), max(points[:, 1])
                    x = (x1 + x3) / 2
                    y = (y1 + y3) / 2
                    w = abs(x1 - x3)
                    h = abs(y1 - y3)
                    x = x/img.shape[1]
                    y = y/img.shape[0]
                    w = w/img.shape[1]
                    h = h/img.shape[0]
                    size = 5
                    x1 = x1 - size if x1 - size > 0 else 0
                    x3 = x3 + size if x3 + size < img.shape[1] else img.shape[1]
                    y1 = y1 - size if y1 - size > 0 else 0
                    y3 = y3 + size if y3 + size < img.shape[0] else img.shape[0]
                    out = img[y1:y3, x1:x3]

                    # cv2.drawContours(img, np.array([points[:, np.newaxis, :]]), -1, (0,0,255), 3)
                    # cv2.imshow('out',out)
                    # cv2.drawContours(out, np.array([ [[[0, 0]],[[1, 0]],[[1, 1]],[[0, 1]]] ]), -1, (0,255,0), 3)
                    # cv2.imshow('out mask',out)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    
                    cv2.imwrite( trainSample + folder + '/' + file.split('.')[0] + f'_{idx}.png', out)
                    labels = np.array(labels)
                    if labels.ndim == 1:
                        labels = labels[np.newaxis, :]
                    np.savetxt(trainSample + folder + '/' + file.split('.')[0] + f'_{idx}.txt', np.array(labels), fmt='%0.5f', delimiter=' ', newline='\n')

def ViT_train_data():
    group_id_list = [0, 1, 2, 3, 4] #0:中文字串 1: 中文字元 2:英數字串 3:中英數混和字串 4:中文單字字串 5:其他 255:Don't care
    train_json_path = '../dataset/train/json'
    train_img_path = '../dataset/train/img'
    save_image_path = '../dataset/train_crop/'
    save_file_path = '../dataset'
    train_ratio = 0.99

    os.makedirs(save_image_path, exist_ok=True)

    # Read json_file list
    json_file_list = os.listdir(train_json_path)
    json_file_list.sort()
    
    # Split data into train/val
    number_of_training = len(json_file_list)
    random_num_list = np.random.choice(number_of_training, number_of_training, replace=False)
    train_index = random_num_list[:int(number_of_training*train_ratio)]
    val_index = random_num_list[int(number_of_training*train_ratio):]

    character_list = []
    train_save_dict = {'annotations':[]}
    val_save_dict = {'annotations':[]}
    
    for jsonfile in tqdm.tqdm(json_file_list):
        # read json information
        train_json_file = os.path.join(train_json_path, jsonfile)
        img_name = jsonfile.split('.')[0]
        image_id = int(img_name.split('_')[1])
        
        # read image
        img_path = os.path.join(train_img_path, img_name + '.jpg')
        img = cv2.imread(img_path)

        # write information
        with open(train_json_file, encoding='utf-8') as f:
            data = json.load(f)
            shapes = data['shapes']

            # iterate all the data
            for i in range(len(shapes)):
                group_id = int(shapes[i]['group_id'])

                if group_id in group_id_list:
                    saved_label = ''

                    # preprocess the bounding box and save it
                    points = shapes[i]['points']
                    pts = np.array(points)
                    out = four_point_transform(img, pts)
                    cv2.imwrite("%s/%s_%d.png"%(save_image_path, img_name, i), out)

                    """ save character """
                    label = shapes[i]['label']
                    # filter out '#'
                    if '#' in label:
                        continue

                    if len(label): # prevent the label from zero-length
                        # Extract all the characters in label and save them
                        for j in range(len(label)):
                            character = label[j]

                            # transform all the characters in lower case
                            if character.isalpha(): 
                                character = character.lower()

                            # check for punctuation
                            test_str = re.search(r"\W", character)
                            if test_str == None:
                                if character not in character_list:
                                    character_list.append(character)
                                saved_label+= character
                            
                    if image_id in train_index:
                        train_save_dict['annotations'].append({'image_id':img_name + '_%d.png' % i, 'caption': saved_label})
                    else:
                        val_save_dict['annotations'].append({'image_id':img_name + '_%d.png' % i, 'caption': saved_label})

    with open(os.path.join(save_file_path, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_save_dict, f, ensure_ascii=False)
    with open(os.path.join(save_file_path, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_save_dict, f, ensure_ascii=False)
    print('training data:', len(train_save_dict['annotations']))
    print('validation data:', len(val_save_dict['annotations']))
    np.savetxt(os.path.join(save_file_path, 'character.txt'), character_list, fmt='%s', encoding='utf-8')

def public_crop():
    public_csv_path = './yolo/dataset/privateL1024_0404_se.csv'
    public_img_path = '../dataset/private'
    save_path = '../dataset/publicL1024_0404_se'

    os.makedirs(save_path, exist_ok=True)
    public_csv = np.loadtxt(public_csv_path, delimiter=',', dtype=np.str)
    count = 0
    
    for idx, public_data in enumerate(tqdm.tqdm(public_csv)):
        image_name = public_data[0]
        if idx == 0:
            tmp = image_name
        if image_name != tmp:
            count = 0
            tmp = image_name
        x1, y1 = int(public_data[1]), int(public_data[2])
        x2, y2 = int(public_data[3]), int(public_data[4])
        x3, y3 = int(public_data[5]), int(public_data[6])
        x4, y4 = int(public_data[7]), int(public_data[8])
        image_path = os.path.join(public_img_path, image_name + '.jpg')
        img = cv2.imread(image_path)

        points = [[x1, y1], [x2,y2], [x3,y3], [x4,y4]]
        pts = np.array(points)
        out = four_point_transform(img, pts)
        croped_resize = cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite("%s/%s_%d.png"%(save_path, image_name, count), croped_resize)
        count += 1

def yoloV5_train_bbox():
    trainJsonPath = '../dataset/train/json/'
    trainImgPath = '../dataset/train/img/'
    trainSample = '../dataset/TextYolo/'
    trainTxt = f'{trainSample}/TrainBBox.txt'
    validTxt = f'{trainSample}/ValidBBox.txt'
    os.makedirs(trainSample, exist_ok = True)
    json_files = os.listdir(trainJsonPath)
    random.shuffle(json_files)
    for json_files, savePath, folder in [(json_files[:int(len(json_files)*0.8)], trainTxt, 'train'), (json_files[int(len(json_files)*0.8):], validTxt, 'Val')]:
        imgs = []
        os.makedirs(trainSample + folder + '/images/', exist_ok = True)
        os.makedirs(trainSample + folder + '/labels/', exist_ok = True)
        
        for file in tqdm.tqdm(json_files):
            with open(trainJsonPath + file, encoding="utf-8") as f:
                data = json.load(f)
            img = cv2.imread(trainImgPath + file.split('.')[0] + '.jpg')
            imgs.append('./images/'+ folder + '/images/' + file.split('.')[0] + '.jpg')
            src = trainImgPath + file.split('.')[0] + '.jpg'
            des = trainSample + folder + '/images/' + file.split('.')[0] + '.jpg'
            shutil.copy(src, des)

            labels = []
            for idx, value in enumerate(data['shapes']):
                label = value['label']
                points = np.array(value['points'])
                group_id = value['group_id']
                shape_type = value['shape_type']
                
                X1, Y1 = min(points[:, 0]), min(points[:, 1])
                X3, Y3 = max(points[:, 0]), max(points[:, 1])
                x = (X1 + X3) / 2
                y = (Y1 + Y3) / 2
                w = abs(X1 - X3)
                h = abs(Y1 - Y3)
                x = x/img.shape[1]
                y = y/img.shape[0]
                w = w/img.shape[1]
                h = h/img.shape[0]
                if group_id in [0,2,3,4]:
                    labels += [[1., x,y,w,h]]
            #     x0 = int((x - w/2)*img.shape[1])
            #     x1 = int((x + w/2)*img.shape[1])
            #     y0 = int((y - h/2)*img.shape[0])
            #     y1 = int((y + h/2)*img.shape[0])
            #     cv2.drawContours(img, np.array([points[:, np.newaxis, :]]), -1, (0,255,0), 3)
            #     cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 3, cv2.LINE_AA)
            # img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)))
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            labels = np.array(labels)
            if labels.ndim == 1:
                labels = labels[np.newaxis, :]
            np.savetxt(trainSample + folder + '/labels/' + file.split('.')[0] + '.txt', np.array(labels), fmt='%0.5f', delimiter=' ', newline='\n')
            del labels
            del img
        np.savetxt(savePath, np.array(imgs), fmt='%s', delimiter=' ', newline='\n')

def yoloV5_test_bbox():
    for path in os.listdir('../dataset/TextYolo_CharEnNum/train/images/'):
        img = cv2.imread('../dataset/TextYolo_CharEnNum/train/images/' + path)
        labels = np.loadtxt('../dataset/TextYolo_CharEnNum/train/labels/' + path[:-4] + '.txt')
        if labels.ndim == 1:
            labels = labels[np.newaxis, :]
        print('../dataset/TextYolo_CharEnNum/train/images/' + path)
        red_color = (0, 0, 255) # BGR
        for label, x,y,w,h in labels:
            x0 = int((x - w/2)*img.shape[1])
            x1 = int((x + w/2)*img.shape[1])
            y0 = int((y - h/2)*img.shape[0])
            y1 = int((y + h/2)*img.shape[0])
            cv2.rectangle(img, (x0, y0), (x1, y1), red_color, 1, cv2.LINE_AA)
        img = cv2.resize(img, (int(img.shape[1]*2), int(img.shape[0]*2)))
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def data_augmentation():
    from torchvision import transforms
    from PIL import Image

    img = cv2.imread("./img_1002_00.png")
    cv2.imwrite('./ori.jpg', cv2.resize(img, (224,224)))
    cv2.imwrite('./72_72.jpg', cv2.resize(cv2.resize(img, (72,72)), (224,224)))
    cv2.imwrite('./96_96.jpg', cv2.resize(cv2.resize(img, (96,96)), (224,224)))
    cv2.imwrite('./121_121.jpg', cv2.resize(cv2.resize(img, (121,121)), (224,224)))
    cv2.imwrite('./146_146.jpg', cv2.resize(cv2.resize(img, (146,146)), (224,224)))
    cv2.imwrite('./171_171.jpg', cv2.resize(cv2.resize(img, (171,171)), (224,224)))
    cv2.imwrite('./196_196.jpg', cv2.resize(cv2.resize(img, (196,196)), (224,224)))
    
    img = Image.open("./img_1002_00.png").convert("RGB")
    img = transforms.Resize((224,224))(img)

    img10 = transforms.RandomRotation(10)(img)
    img10.save('./Rotation10.jpg')

    img10 = transforms.RandomHorizontalFlip()(img)
    img10.save('./Horizontal.jpg')

    img10 = transforms.RandomRotation(10)(img)
    img10 = transforms.RandomHorizontalFlip()(img10)
    img10.save('./Rotation10 + Horizontal.jpg')

    imgB = transforms.ColorJitter(brightness=0.5)(img)
    imgB.save('./brightness.jpg')
    imgC = transforms.ColorJitter(contrast=0.5)(img)
    imgC.save('./contrast.jpg')
    imgS = transforms.ColorJitter(saturation=0.5)(img)
    imgS.save('./saturation.jpg')
    imgH = transforms.ColorJitter(hue=0.5)(img)
    imgH.save('./hue.jpg')
    imgall = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(img)
    imgall.save('./ColorJitter.jpg')

if __name__ in "__main__":
    Tranform_train() # For ROI transformation training
    yoloV5_train_bbox()
    ViT_train_data()
    # yoloV5_test_bbox() 
    # data_augmentation() # Visualize the image after data augmentation
    # public_crop() # Visualize the bounding box extracted by YoloV5