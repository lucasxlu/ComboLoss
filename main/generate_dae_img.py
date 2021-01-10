"""
@Author: LucasX
@Time: 2021/01/09
@Desc: generate reconstructed images
"""
import os
import sys
import argparse

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
args.add_argument('-save_to_dir', help='image directory', type=str, default='./gen_img')
args = vars(args.parse_args())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_img_with_dae(img_f, model):
    """
    generate reconstructed image with ResConvDAE
    :param img_f:
    :param model:
    :return:
    """
    model.eval()
    img = io.imread(img_f)
    img = Image.fromarray(img.astype(np.uint8))

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = preprocess(img)
    img.unsqueeze_(0)
    img = img.to(device)
    output = model(img)

    output = output.to("cpu").detach().numpy().astype(np.float)[0].transpose([1, 2, 0])
    output[:, :, 0] *= 0.229
    output[:, :, 0] += 0.485
    output[:, :, 1] *= 0.224
    output[:, :, 1] += 0.456
    output[:, :, 2] *= 0.225
    output[:, :, 2] += 0.406
    output *= 255.0
    output = output.clip(0, 255)
    output = Image.fromarray(np.uint8(output), mode='RGB')

    os.makedirs(args['save_to_dir'], exist_ok=True)
    output.save('./{}/gen_{}'.format(args['save_to_dir'], img_f.split(os.path.sep)[-1]))
    print(f'Reconstructed image for {os.path.basename(img_f)} has been generated...')


if __name__ == '__main__':
    resconvdae = ResConvDAE()
    model_name = resconvdae.__class__.__name__
    resconvdae = resconvdae.float()
    resconvdae = resconvdae.to(device)
    print('Start testing %s...' % model_name)
    resconvdae.load_state_dict(torch.load(args['ckpt']))
    for img_f in os.listdir(args['img_dir']):
        generate_img_with_dae(os.path.join(args['img_dir'], img_f), resconvdae)
