"""
@Author: LucasX
@Time: 2021/01/17
@Desc: extract compressed deep features via residual conv auto encoder
"""
import os
import sys
import argparse
import pickle

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage import io

sys.path.append('../')
from models.resconvdae import *

args = argparse.ArgumentParser()
args.add_argument('-ckpt', help='checkpoint of pretrained ResDAE', type=str, default='./model/ResConvDAE.pth')
args.add_argument('-img_dir', help='image directory', type=str, default='/home/xulu/DataSet/Face/SCUT-FBP/Crop')
args.add_argument('-save_to_dir', help='image directory', type=str, default='./features')
args = vars(args.parse_args())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def extract_features_with_dae(img_f, model):
    """
    extract deep features with ResConvDAE
    :param img_f:
    :param model:
    :return:
    """
    model.eval()
    img = io.imread(img_f)
    img = Image.fromarray(img.astype(np.uint8))

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = preprocess(img)
    img.unsqueeze_(0)
    img = img.to(device)
    output = model.encoder(img)

    feat = output.to("cpu").detach().numpy().astype(np.float)[0].transpose([1, 2, 0]).ravel()

    os.makedirs(args['save_to_dir'], exist_ok=True)
    with open('./{}/gen_{}'.format(args['save_to_dir'], img_f.split(os.path.sep)[-1].split('.')[0] + '.pkl'),
              'wb') as f:
        pickle.dump(feat, f)
        print(f'extract deep features for {os.path.basename(img_f)} successfully...')


if __name__ == '__main__':
    resconvdae = ResConvDAE()
    model_name = resconvdae.__class__.__name__
    resconvdae = resconvdae.float()
    resconvdae = resconvdae.to(device)
    print('Start extracting deep features...')
    resconvdae.load_state_dict(torch.load(args['ckpt']))
    for img_f in os.listdir(args['img_dir']):
        extract_features_with_dae(os.path.join(args['img_dir'], img_f), resconvdae)
