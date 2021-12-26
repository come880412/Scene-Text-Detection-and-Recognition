import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        return batch_text.to(device)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def get_device(verbose=True):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if verbose:
        print("Device:", device)
    return device

def Visualize_attmap(attention_matrix):
    img = Image.open('./ori_img.png')
    mask = cv2.resize(attention_matrix[0, 1:].reshape(14, 14).numpy(), img.size)
    plt.imshow(mask, cmap='jet')
    plt.imshow(img, alpha = 0.4)
    plt.axis('off')
    plt.savefig('./att_img.png')
    plt.show()


def get_args(is_train=True):
    parser = argparse.ArgumentParser(description='STR')
    # for train
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--n_epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('--data_dir', default='../../dataset/train_crop', help='path to training dataset')
    parser.add_argument('--character_dir', default='../../dataset/character.txt', help='path to character list')
    parser.add_argument('--train_data', default='../../dataset/train.json', help='path to training json')
    parser.add_argument('--valid_data', default='../../dataset/val.json', help='path to validation json')
    parser.add_argument('--manualSeed', type=int, default=412, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers. Use -1 to use all cores.', default=8)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--sgd', action='store_true', help='Whether to use SGD (default is Adadelta)')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=40, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=224, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=224, help='the width of the input image')
    parser.add_argument('--rgb', default=True, help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    
    """ Model Architecture """
    choices = ["vitstr_tiny_patch16_224", "vitstr_small_patch16_224", "vitstr_base_patch16_224", "vitstr_tiny_distilled_patch16_224", "vitstr_small_distilled_patch16_224"]
    parser.add_argument('--TransformerModel', default=choices[2], help='Which vit/deit transformer model', choices=choices)
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the number of input channel of Feature extractor')

    parser.add_argument('--gpu_id', type=str, default='0', help='gpu_id')
    # use cosine learning rate decay
    parser.add_argument('--scheduler', action='store_true', help='Use lr scheduler')
   
    args = parser.parse_args()
    return args
