"""
train and eval ComboLoss
"""
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, accuracy_score
from torch import nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import models
from pytorchcv.model_provider import get_model as ptcv_get_model

sys.path.append('../')
from models.nets import ComboNet
from models.losses import CombinedLoss, SmoothHuberLoss
from data.data_loaders import load_scutfbp, load_hotornot, load_scutfbp5500_64, load_scutfbp5500_cv
from util.file_util import mkdirs_if_not_exist
from config.cfg import cfg


def train_regressor(model, dataloaders, criterion, optimizer, scheduler, num_epochs, inference=False):
    """
    train regressor
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

    dataset_sizes = {x: dataloaders[x].dataset for x in ['train', 'val', 'test']}

    if not inference:
        print('Start training %s...' % model_name)
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_record = {
            'pc': 0.0,
            'epoch': 0
        }

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
                epoch_gt = []
                epoch_pred = []

                # Iterate over data.
                # for data in dataloaders[phase]:
                for i, data in enumerate(dataloaders[phase], 0):

                    inputs = data['image']
                    scores = data['score']
                    classes = data['class']

                    inputs = inputs.to(device)
                    scores = scores.to(device).float()
                    classes = classes.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        outputs = outputs.view(-1)

                        loss = criterion(outputs, scores)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    epoch_gt += scores.to('cpu').detach().numpy().ravel().tolist()
                    epoch_pred += outputs.to('cpu').detach().numpy().ravel().tolist()

                if phase == 'train':
                    if torch.__version__ >= '1.1.0':
                        scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]

                epoch_mae = round(mean_absolute_error(np.array(epoch_gt).flatten(), np.array(epoch_pred).flatten()), 4)
                epoch_rmse = round(
                    np.math.sqrt(mean_squared_error(np.array(epoch_gt).flatten(), np.array(epoch_pred).flatten())), 4)
                epoch_pc = round(np.corrcoef(np.array(epoch_gt).flatten(), np.array(epoch_pred).flatten())[0, 1], 4)

                print('[{}] Loss: {:.4f} MAE: {} RMSE: {} PC: {}'
                      .format(phase, epoch_loss, epoch_mae, epoch_rmse, epoch_pc))

                # deep copy the model
                if phase == 'val' and epoch_pc > best_record['pc']:
                    best_record['pc'] = epoch_pc
                    best_record['epoch'] = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

                    model.load_state_dict(best_model_wts)
                    mkdirs_if_not_exist('./model')
                    torch.save(model.state_dict(),
                               './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('+' * 100)
        print('Epoch {} achieves best PC: {:4f}'.format(best_record['epoch'], best_record['pc']))
        print('+' * 100)

        # load best model weights
        model.load_state_dict(best_model_wts)
        mkdirs_if_not_exist('./model')
        torch.save(model.state_dict(), './model/%s.pth' % model_name)
    else:
        print('Start testing %s...' % model.__class__.__name__)
        model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model_name)))

    model.load_state_dict(torch.load('./model/{0}_best_epoch-{1}.pth'.format(model_name, best_record['epoch'])))
    model.eval()

    total = 0
    y_pred = []
    y_true = []
    filenames = []

    with torch.no_grad():
        for data in dataloaders['test']:
            images = data['image']
            images = images.to(device)

            filenames += data['filename']
            outputs = model(images)

            total += images.size(0)

            y_pred += outputs.to("cpu").detach().numpy().tolist()
            y_true += data['score'].detach().numpy().tolist()

    mae = round(mean_absolute_error(np.array(y_true).ravel(), np.array(y_pred).ravel()), 4)
    rmse = round(np.math.sqrt(mean_squared_error(np.array(y_true).ravel(), np.array(y_pred).ravel())), 4)
    pc = round(np.corrcoef(np.array(y_true).ravel(), np.array(y_pred).ravel())[0, 1], 4)

    print('===============The Mean Absolute Error of {0} is {1}===================='.format(model_name, mae))
    print('===============The Root Mean Square Error of {0} is {1}===================='.format(model_name, rmse))
    print('===============The Pearson Correlation of {0} is {1}===================='.format(model_name, pc))

    col = ['filename', 'gt', 'pred']
    df = pd.DataFrame([[filenames[i], y_true[i], y_pred[i][0]] for i in range(len(y_true))],
                      columns=col)
    df.to_excel("./{0}.xlsx".format(model_name), sheet_name='Output', index=False)
    print('Output Excel has been generated~')


def train_classifier(model, criterion, optimizer, scheduler, dataloaders, num_epochs, inference=False):
    """
    train classifier
    :param model:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param dataloaders:
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

    dataset_sizes = {x: dataloaders[x].__len__() for x in ['train', 'val', 'test']}

    if not inference:
        print('Start training %s...' % model_name)
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_record = {
            'acc': 0.0,
            'epoch': 0
        }

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    if torch.__version__ <= '1.1.0':
                        scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                tmp_y_pred = []
                tmp_y_true = []

                # Iterate over data.
                for i, data in enumerate(dataloaders[phase], 0):
                    inputs = data['image']
                    scores = data['score']
                    classes = data['class']

                    inputs = inputs.to(device)
                    classes = classes.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs.data, 1)

                        loss = criterion(outputs, classes)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    # running_corrects += torch.sum(preds == classes.data)
                    tmp_y_true += data['class'].detach().numpy().tolist()
                    tmp_y_pred += preds.to("cpu").detach().numpy().tolist()

                if phase == 'train':
                    if torch.__version__ >= '1.1.0':
                        scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = accuracy_score(tmp_y_true, tmp_y_pred) * 100
                # epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}%'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_record['acc']:
                    best_record['acc'] = epoch_acc
                    best_record['epoch'] = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Epoch {} achieves best Acc: {:4f}'.format(best_record['epoch'], best_record['acc']))

        # load best model weights
        model.load_state_dict(best_model_wts)

    else:
        print('Start testing %s...' % model_name)
        model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model_name)))

    model.eval()

    total = 0
    y_pred = []
    y_true = []
    filenames = []
    probs = []

    with torch.no_grad():
        for data in dataloaders['test']:
            images = data['image'].to(device)

            filenames += data['filename']
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)
            # get TOP-K output labels and corresponding probabilities
            topK_prob, topK_label = torch.topk(outputs, 1)
            probs += topK_prob.to("cpu").detach().numpy().tolist()

            _, predicted = torch.max(outputs.data, 1)

            total += images.size(0)
            y_true += data['class'].detach().numpy().tolist()
            y_pred += predicted.to("cpu").detach().numpy().tolist()

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print('===============The Accuracy of {0} is {1}===================='.format(model_name, accuracy))
    print('===============The Precision of {0} is {1}===================='.format(model_name, precision))
    print('===============The Recall of {0} is {1}===================='.format(model_name, recall))

    col = ['filename', 'gt', 'pred', 'prob']
    df = pd.DataFrame([[filenames[i], y_true[i], y_pred[i], probs[i][0]] for i in range(len(y_true))],
                      columns=col)
    df.to_excel("./{0}.xlsx".format(model_name), sheet_name='Output', index=False)
    print('Output Excel has been generated~')


def train_combinator(model, dataloaders, criterion, optimizer, scheduler, num_epochs, inference=False):
    """
    train combinator
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

    dataset_sizes = {x: dataloaders[x].__len__() for x in ['train', 'val', 'test']}

    if not inference:
        print('Start training %s...' % model_name)
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_record = {
            'pc': 0.0,
            'epoch': 0
        }

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
                epoch_gt = []
                epoch_pred = []

                # Iterate over data.
                # for data in dataloaders[phase]:
                for i, data in enumerate(dataloaders[phase], 0):

                    inputs = data['image']
                    scores = data['score']
                    classes = data['class']

                    inputs = inputs.to(device)
                    scores = scores.to(device).float()
                    classes = classes.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        regression_output, classification_output = model(inputs)
                        regression_output = regression_output.view(-1)
                        _, predicted = torch.max(classification_output.data, 1)

                        loss = criterion(regression_output, scores, F.softmax(classification_output, 1), predicted,
                                         classes)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    epoch_gt += scores.to('cpu').detach().numpy().ravel().tolist()
                    epoch_pred += regression_output.to('cpu').detach().numpy().ravel().tolist()

                if phase == 'train':
                    if torch.__version__ >= '1.1.0':
                        scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]

                epoch_mae = round(mean_absolute_error(np.array(epoch_gt).flatten(), np.array(epoch_pred).flatten()), 4)
                epoch_rmse = round(
                    np.math.sqrt(mean_squared_error(np.array(epoch_gt).flatten(), np.array(epoch_pred).flatten())), 4)
                epoch_pc = round(np.corrcoef(np.array(epoch_gt).flatten(), np.array(epoch_pred).flatten())[0, 1], 4)

                print('[{}] Loss: {:.4f} MAE: {} RMSE: {} PC: {}'
                      .format(phase, epoch_loss, epoch_mae, epoch_rmse, epoch_pc))

                # deep copy the model
                if phase == 'val' and epoch_pc > best_record['pc']:
                    best_record['pc'] = epoch_pc
                    best_record['epoch'] = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

                    model.load_state_dict(best_model_wts)
                    mkdirs_if_not_exist('./model')
                    torch.save(model.state_dict(),
                               './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('+' * 100)
        print('Epoch {} achieves best PC: {:4f}'.format(best_record['epoch'], best_record['pc']))
        print('+' * 100)

        # load best model weights
        model.load_state_dict(best_model_wts)
        mkdirs_if_not_exist('./model')
        torch.save(model.state_dict(), './model/%s.pth' % model_name)
    else:
        print('Start testing %s...' % model_name)
        model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model_name)))

    model.eval()

    total = 0
    y_pred = []
    y_true = []
    filenames = []

    with torch.no_grad():
        for data in dataloaders['test']:
            images = data['image']
            images = images.to(device)

            filenames += data['filename']
            regression_output, classification_output = model(images)
            probs = F.softmax(classification_output, dim=1)
            cls = torch.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float).T).to(device)  # for SCUT-FBP*
            # cls = torch.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float).T).to(device)  # for HotOrNot
            # expectation = torch.matmul(probs, cls.float()).view(-1).view(-1, 1)

            # output = (2 * regression_output + expectation) / 3
            output = regression_output
            total += images.size(0)

            y_pred += output.to("cpu").detach().numpy().tolist()
            y_true += data['score'].detach().numpy().tolist()

    mae = round(mean_absolute_error(np.array(y_true).ravel(), np.array(y_pred).ravel()), 4)
    rmse = round(np.math.sqrt(mean_squared_error(np.array(y_true).ravel(), np.array(y_pred).ravel())), 4)
    pc = round(np.corrcoef(np.array(y_true).ravel(), np.array(y_pred).ravel())[0, 1], 4)

    print('===============The Mean Absolute Error of {0} is {1}===================='.format(model_name, mae))
    print('===============The Root Mean Square Error of {0} is {1}===================='.format(model_name, rmse))
    print('===============The Pearson Correlation of {0} is {1}===================='.format(model_name, pc))

    col = ['filename', 'gt', 'pred']
    df = pd.DataFrame([[filenames[i], y_true[i], y_pred[i][0]] for i in range(len(y_true))],
                      columns=col)
    df.to_excel("./{0}.xlsx".format(model_name), sheet_name='Output', index=False)
    print('Output Excel has been generated~')


def main(model, data_name, model_type):
    """
    train model
    :param model:
    :param data_name: SCUT-FBP/HotOrNot/SCUT-FBP5500
    :param model_type: classifier/regressor
    :return:
    """
    xent_weight_list = None
    if data_name == 'SCUT-FBP':
        print('start loading SCUTFBPDataset...')
        dataloaders = load_scutfbp()
        xent_weight_list = [91.5, 1.0, 1.06, 5.72, 18.3]
    elif data_name == 'HotOrNot':
        print('start loading HotOrNotDataset...')
        dataloaders = load_hotornot(cv_split_index=cfg['cv_index'])
        xent_weight_list = [3.35, 1.0, 3.34]
    elif data_name == 'SCUT-FBP5500':
        print('start loading SCUTFBP5500Dataset...')
        dataloaders = load_scutfbp5500_64()
        xent_weight_list = [1.88, 1, 1.91, 99.38, 99.38]
    elif data_name == 'SCUT-FBP5500-CV':
        print('start loading SCUTFBP5500DatasetCV...')
        dataloaders = load_scutfbp5500_cv(cv_index=cfg['cv_index'])
        if cfg['cv_index'] == 1:
            xent_weight_list = [93.3, 1.98, 1.0, 1.91, 102.19]
        elif cfg['cv_index'] == 2:
            xent_weight_list = [105.9, 1.92, 1.0, 1.86, 92.09]
        elif cfg['cv_index'] == 3:
            xent_weight_list = [97.64, 1.97, 1.0, 1.92, 89.5]
        elif cfg['cv_index'] == 4:
            xent_weight_list = [96.68, 1.92, 1.0, 1.9, 106.35]
        elif cfg['cv_index'] == 5:
            xent_weight_list = [85.32, 1.94, 1.0, 1.9, 106.65]
    else:
        print('Invalid data name. It can only be [SCUT-FBP], [HotOrNot], [SCUT-FBP5500] or [SCUT-FBP5500-CV]...')
        sys.exit(0)

    if model_type == 'regressor':
        # criterion = nn.MSELoss()
        # criterion = nn.SmoothL1Loss()
        # criterion = nn.L1Loss()
        criterion = SmoothHuberLoss()

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        train_regressor(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
                        scheduler=exp_lr_scheduler, num_epochs=cfg['epoch'], inference=False)
    elif model_type == 'classifier':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        train_classifier(model, criterion, optimizer, scheduler=exp_lr_scheduler, dataloaders=dataloaders,
                         num_epochs=cfg['epoch'], inference=False)
    elif model_type == 'combinator':
        criterion = CombinedLoss(xent_weight=xent_weight_list)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        train_combinator(model, dataloaders, criterion, optimizer, scheduler=exp_lr_scheduler, num_epochs=cfg['epoch'],
                         inference=False)


if __name__ == '__main__':
    seresnext50 = ptcv_get_model("seresnext50_32x4d", pretrained=True)
    num_ftrs = seresnext50.output.in_features
    seresnext50.output = nn.Linear(num_ftrs, 1)

    # resnet18 = models.resnet18(pretrained=True)
    # num_ftrs = resnet18.fc.in_features
    # resnet18.fc = nn.Linear(num_ftrs, 1)

    main(ComboNet(num_out=5), 'SCUT-FBP', 'combinator')
    # main(seresnext50, 'SCUT-FBP5500', 'regressor')
