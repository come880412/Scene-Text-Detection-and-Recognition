'''
Modified Date: 2021/12/20

Author: Gi-Luen Huang

mail: come880412@gmail.com
'''

from dataset import public_data
import torch
from utils import get_args, TokenLabelConverter, Visualize_attmap
from model import Model
import csv
import numpy as np
import torch.nn.functional as F
import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opt = get_args()

model_path = './saved_models/complex_new3/'
output_csv = './publicL1024_0404_se.csv'
data_path = '../../dataset/public'
csv_path = '../yolo/dataset/publicL1024_0404_se.csv'
filter_pixel = 45
opt.character_dir = os.path.join(model_path, 'character.txt')
batch = 64

if __name__ == '__main__':
    """Prepare the character list"""
    opt.character = ''
    with open(opt.character_dir, encoding='utf-8') as f:
        data = f.read().splitlines()
        for line in data:
            opt.character = opt.character + line

    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)

    # model = torch.nn.DataParallel(Model(opt)).to(device)
    model = Model(opt).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'best_accuracy.pth')))
    model.eval()

    public_dataset = public_data(data_path, csv_path, filter_pixel)
    public_loader = torch.utils.data.DataLoader(
            public_dataset, batch_size=batch,
            shuffle=False, 
            num_workers=4)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        with torch.no_grad():
            for idx,(image, save) in enumerate(tqdm.tqdm(public_loader)):
                image = image.cuda()
                batch_size = image.shape[0]

                preds, att_map = model(image, seqlen=converter.batch_max_length)
                # Visualize_attmap(att_map.cpu())
                
                _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
                preds_index = preds_index.view(-1, converter.batch_max_length)
                length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
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
