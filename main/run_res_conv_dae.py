import copy
import os
import sys
import time
import math

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import spatial
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

sys.path.append('../')
from models import ssim
from models.resconvdae import *
from models.losses import ReconstructionLoss
from data.data_loaders import load_scutfbp, load_hotornot, load_scutfbp5500_64, load_scutfbp5500_cv
from util.file_util import mkdirs_if_not_exist
from config.cfg import cfg


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, inference=False):
    """
    train model
    :param model:
    :param dataloaders:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param inference:
    :return:
    """
    print(model)
    model_name = model.__class__.__name__
    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() and cfg['use_gpu'] else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}

    for k, v in dataset_sizes.items():
        print('Dataset size of {0} is {1}...'.format(k, v))

    if not inference:
        print('Start training %s...' % model_name)
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_ssim = 0.0
        best_cosine_similarity = 0.0
        best_l2_dis = float('inf')

        for epoch in range(num_epochs):
            print('-' * 100)
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    if torch.__version__ <= '1.1.0':
                        scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_ssim = 0.0
                running_l2_dis = 0.0
                running_cos_sim = 0.0

                # Iterate over data.
                # for data in dataloaders[phase]:
                for i, data in enumerate(dataloaders[phase], 0):

                    inputs = data['image']
                    inputs = inputs.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        loss = criterion(outputs, inputs)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.sum().backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.sum() * inputs.size(0)
                    running_cos_sim += 1 - spatial.distance.cosine(outputs.to('cpu').detach().numpy().ravel(),
                                                                   inputs.to('cpu').detach().numpy().ravel())
                    running_l2_dis += np.linalg.norm(
                        outputs.to('cpu').detach().numpy().ravel() - inputs.to('cpu').detach().numpy().ravel())
                    running_ssim += ssim.ssim(outputs, inputs)

                if phase == 'train':
                    if torch.__version__ >= '1.1.0':
                        scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_l2_dis = running_l2_dis / dataset_sizes[phase]
                epoch_cos_sim = running_cos_sim / dataset_sizes[phase]
                epoch_ssim = running_ssim / dataset_sizes[phase]

                print('{} Loss: {:.4f} L2_Distance: {} Cosine_Similarity: {} SSIM: {}'
                      .format(phase, epoch_loss, epoch_l2_dis, epoch_cos_sim, epoch_ssim))

                # deep copy the model
                if phase == 'val' and epoch_l2_dis <= best_l2_dis:
                    best_l2_dis = epoch_l2_dis
                    best_model_wts = copy.deepcopy(model.state_dict())

                    model.load_state_dict(best_model_wts)
                    model_path_dir = './model'
                    mkdirs_if_not_exist(model_path_dir)
                    state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
                    torch.save(state_dict, './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best L2_Distance: {:4f}'.format(best_l2_dis))

        # load best model weights
        model.load_state_dict(best_model_wts)
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        torch.save(state_dict, './model/%s.pth' % model_name)
    else:
        print('Start testing %s...' % model.__class__.__name__)
        model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model_name)))

    model.eval()

    cos_sim, l2_dist, ssim_ = 0.0, 0.0, 0.0

    with torch.no_grad():
        for data in dataloaders['test']:
            images = data['image']
            images = images.to(device)
            outputs = model(images)

            cos_sim += 1 - spatial.distance.cosine(outputs.to('cpu').detach().numpy().ravel(),
                                                   images.to('cpu').detach().numpy().ravel())
            l2_dist += np.linalg.norm(
                outputs.to('cpu').detach().numpy().ravel() - images.to('cpu').detach().numpy().ravel())
            ssim_ += ssim.ssim(outputs, images)

    print('*' * 200)
    print('Avg L2 Distance of {0} on test set: {1}'.format(model_name, l2_dist / dataset_sizes['test']))
    print('Avg CosineSimilarity of {0} on test set: {1}'.format(model_name, cos_sim / dataset_sizes['test']))
    print('Avg SSIM of {0} on test set: {1}'.format(model_name, ssim_ / dataset_sizes['test']))
    print('*' * 200)


def main(model, data_name):
    """
    train model
    :param model:
    :param data_name: SCUT-FBP/HotOrNot/SCUT-FBP5500/SCUT-FBP5500CV
    :return:
    """
    # criterion = ReconstructionLoss()
    criterion = nn.MSELoss()

    optimizer_ft = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

    if data_name == 'SCUT-FBP':
        print('start loading SCUTFBPDataset...')
        dataloaders = load_scutfbp()
    elif data_name == 'HotOrNot':
        print('start loading HotOrNotDataset...')
        dataloaders = load_hotornot(cv_split_index=cfg['cv_index'])
    elif data_name == 'SCUT-FBP5500':
        print('start loading SCUTFBP5500Dataset...')
        dataloaders = load_scutfbp5500_64()
    elif data_name == 'SCUT-FBP5500CV':
        print('start loading SCUTFBP5500Dataset Cross Validation...')
        dataloaders = load_scutfbp5500_cv(cfg['cv_index'])
    else:
        print('Invalid data name. It can only be SCUT-FBP or HotOrNot...')
        sys.exit(0)

    train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer_ft,
                scheduler=exp_lr_scheduler, num_epochs=cfg['epoch'], inference=False)


def ext_res_dae_feat(img, res_dae):
    """
    extract deep features from Residual Deep AutoEncoder's encoder module
    :param img:
    :param res_dae:
    :return:
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if isinstance(img, str):
        img = Image.open(img)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = preprocess(img)
    img.unsqueeze_(0)
    img = img.to(device)
    encoder = res_dae.module.encoder if torch.cuda.device_count() > 1 else res_dae.encoder
    feat = encoder(img).to("cpu").detach().numpy().ravel()

    return feat


if __name__ == '__main__':
    res_conv_dae = ResConvDAE()
    main(res_conv_dae, 'SCUT-FBP5500')

    # resConvDAE = ResConvDAE()
    # model_name = resConvDAE.__class__.__name__
    # resConvDAE = resConvDAE.float()
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #
    # resConvDAE = resConvDAE.to(device)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     resConvDAE = nn.DataParallel(resConvDAE)
    # print('[INFO] loading pretrained weights for %s...' % model_name)
    # resConvDAE.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model_name)))
    # resConvDAE.eval()
    #
    # img_dir = '/home/xulu/DataSet/SCUT-FBP/Crop'
    # for img_f in os.listdir(img_dir):
    #     # generate_img_with_dae(os.path.join(img_dir, img_f))
    #     feat = ext_res_dae_feat(os.path.join(img_dir, img_f), resConvDAE)
    #     print(feat)
