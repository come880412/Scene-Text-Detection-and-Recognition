import os, tqdm, cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from Trainer import TextData


if __name__ in "__main__":
    save = './plot'
    os.makedirs(save, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    epochs = 120
    batch_size = 32
    optimizerName = 'sgd'
    print("Load dataset...")
    valid_data = TextData(path = '../../dataset/TextTranform2/Val/', istrain = False, transform = True)
    vaildloader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle = False, num_workers = 0)

    print("Load model...")
    # model = models.resnet50(pretrained = True)
    # fc_features = model.fc.in_features
    # model.fc = nn.Linear(fc_features, 8)
    model = torch.load('./models/best.pkl')
    model = model.cuda()

    print("Validation...")
    model.eval()
    with torch.no_grad():
        for idx,(inputs, labels, paths) in enumerate(tqdm.tqdm(vaildloader)):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            preds = nn.Sigmoid()(logps)

            for n, (path, pred, target) in enumerate( zip(paths, preds, labels) ):
                X1, Y1, X2, Y2, X3, Y3, X4, Y4 = pred.cpu().data.numpy()
                x1, y1, x2, y2, x3, y3, x4, y4 = target.cpu().data.numpy()
                img = cv2.imread(path, 1)
                X1, X2, X3, X4, x1, x2, x3, x4 = X1*(img.shape[1]), X2*(img.shape[1]), X3*(img.shape[1]), X4*(img.shape[1]), x1*(img.shape[1]), x2*(img.shape[1]), x3*(img.shape[1]), x4*(img.shape[1])
                Y1, Y2, Y3, Y4, y1, y2, y3, y4 = Y1*(img.shape[0]), Y2*(img.shape[0]), Y3*(img.shape[0]), Y4*(img.shape[0]), y1*(img.shape[0]), y2*(img.shape[0]), y3*(img.shape[0]), y4*(img.shape[0])
                X1, X2, X3, X4, x1, x2, x3, x4 = int(X1), int(X2), int(X3), int(X4), int(x1), int(x2), int(x3), int(x4)
                Y1, Y2, Y3, Y4, y1, y2, y3, y4 = int(Y1), int(Y2), int(Y3), int(Y4), int(y1), int(y2), int(y3), int(y4)
                # print([[x1, y1]],[[x2, y2]],[[x3, y3]],[[x4, y4]])
                # print([[X1, Y1]],[[X2, Y2]],[[X3, Y3]],[[X4, Y4]])
                cv2.drawContours(img, np.array([ [[[x1, y1]],[[x2, y2]],[[x3, y3]],[[x4, y4]]] ]), -1, (0,0,255), 1)
                cv2.drawContours(img, np.array([ [[[X1, Y1]],[[X2, Y2]],[[X3, Y3]],[[X4, Y4]]] ]), -1, (0,255,0), 1)
                # cv2.imshow('out',out)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(f'{save}/{idx}_{n}.png', img)