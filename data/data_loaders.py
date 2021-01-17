import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Lambda

sys.path.append('../')
from config.cfg import cfg
from data.datasets import ScutFBPDataset, HotOrNotDataset, SCUTFBP5500Dataset, SCUTFBP5500DatasetCV


def load_scutfbp():
    """
    load SCUTFBP Dataset
    :return:
    """
    df = pd.read_excel('../data/SCUT-FBP.xlsx', sheet_name='Sheet1')
    X_train, X_test, y_train, y_test = train_test_split(df['Image'].tolist(), df['Attractiveness label'],
                                                        test_size=0.2, random_state=cfg['random_seed'])

    resize_to = 224

    train_dataset = ScutFBPDataset(f_list=X_train, f_labels=y_train, transform=transforms.Compose([
        transforms.Resize(resize_to),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

    test_dataset = ScutFBPDataset(f_list=X_test, f_labels=y_test, transform=transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=50, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=50, pin_memory=True)

    return {'train': train_dataloader, 'val': test_dataloader, 'test': test_dataloader}


def load_hotornot(cv_split_index=1):
    """
    load HotOrNot Dataset
    :param cv_split_index:
    :return:
    """
    train_dataset = HotOrNotDataset(cv_split_index=cv_split_index, train=True, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(30),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

    test_dataset = HotOrNotDataset(cv_split_index=cv_split_index, train=False, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=50, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=50, pin_memory=True)

    return {'train': train_dataloader, 'val': test_dataloader, 'test': test_dataloader}


def load_scutfbp5500_64():
    """
    load Face Dataset for SCUT-FBP5500 with 6/4 split CV
    :return:
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomRotation(30),
            # transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_loader = torch.utils.data.DataLoader(SCUTFBP5500Dataset(train=True, transform=data_transforms['train']),
                                               batch_size=cfg['batch_size'], shuffle=True, num_workers=50,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(SCUTFBP5500Dataset(train=False, transform=data_transforms['test']),
                                              batch_size=cfg['batch_size'], shuffle=False, num_workers=50,
                                              pin_memory=True)

    return {'train': train_loader, 'val': test_loader, 'test': test_loader}


def load_scutfbp5500_cv(cv_index=1):
    """
    load SCUT-FBP5500 with Cross Validation Index
    :return:
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomRotation(30),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_loader = torch.utils.data.DataLoader(
        SCUTFBP5500DatasetCV(cv_index=cv_index, train=True, transform=data_transforms[
            'train']), batch_size=cfg['batch_size'], shuffle=True, num_workers=50, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        SCUTFBP5500DatasetCV(cv_index=cv_index, train=False, transform=data_transforms[
            'test']), batch_size=cfg['batch_size'], shuffle=False, num_workers=50, pin_memory=True, drop_last=False)

    return {'train': train_loader, 'val': test_loader, 'test': test_loader}


def load_reconstruct_scutfbp():
    """
    load SCUTFBP Dataset for Reconstruction
    :return:
    """
    df = pd.read_excel('../data/SCUT-FBP.xlsx', sheet_name='Sheet1')
    X_train, X_test, y_train, y_test = train_test_split(df['Image'].tolist(), df['Attractiveness label'],
                                                        test_size=0.2, random_state=cfg['random_seed'])

    resize_to = 224

    train_dataset = ScutFBPDataset(f_list=X_train, f_labels=y_train, transform=transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
    ]))

    test_dataset = ScutFBPDataset(f_list=X_test, f_labels=y_test, transform=transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
    ]))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=50, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=50, pin_memory=True)

    return {'train': train_dataloader, 'val': test_dataloader, 'test': test_dataloader}


def load_reconstruct_hotornot(cv_split_index=1):
    """
    load HotOrNot Dataset for Reconstruction
    :param cv_split_index:
    :return:
    """
    train_dataset = HotOrNotDataset(cv_split_index=cv_split_index, train=True, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]))

    test_dataset = HotOrNotDataset(cv_split_index=cv_split_index, train=False, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=50, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=50, pin_memory=True)

    return {'train': train_dataloader, 'val': test_dataloader, 'test': test_dataloader}


def load_reconstruct_scutfbp5500_64():
    """
    load Face Dataset for SCUT-FBP5500 with 6/4 split CV for Reconstruction
    :return:
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
    }

    train_loader = torch.utils.data.DataLoader(SCUTFBP5500Dataset(train=True, transform=data_transforms['train']),
                                               batch_size=cfg['batch_size'], shuffle=True, num_workers=50,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(SCUTFBP5500Dataset(train=False, transform=data_transforms['test']),
                                              batch_size=cfg['batch_size'], shuffle=False, num_workers=50,
                                              pin_memory=True)

    return {'train': train_loader, 'val': test_loader, 'test': test_loader}


def load_reconstruct_scutfbp5500_cv(cv_index=1):
    """
    load SCUT-FBP5500 with Cross Validation Index for Reconstruction
    :return:
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
    }

    train_loader = torch.utils.data.DataLoader(
        SCUTFBP5500DatasetCV(cv_index=cv_index, train=True, transform=data_transforms[
            'train']), batch_size=cfg['batch_size'], shuffle=True, num_workers=50, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        SCUTFBP5500DatasetCV(cv_index=cv_index, train=False, transform=data_transforms[
            'test']), batch_size=cfg['batch_size'], shuffle=False, num_workers=50, pin_memory=True, drop_last=False)

    return {'train': train_loader, 'val': test_loader, 'test': test_loader}
