#!/usr/bin/env python
# coding: utf-8

# In[9]:

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from tqdm import tqdm
import sys
import copy
import itertools
import pickle
import ast, csv
import datetime
# from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import partial
import warnings
import numpy as np
import random
from shapely.geometry import Polygon
from skimage import measure
from shapely.validation import make_valid
from skimage.io import imread, imsave
from skimage import measure
from skimage.io import imread
from glob import glob
from time import gmtime, strftime
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import skimage
import mmcv
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
import json
import shutil
from skimage import measure
from shapely import geometry
from shapely.validation import make_valid
import os
import random
import math
from torchvision.io import read_image
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from itertools import cycle
from torchvision.models.segmentation import fcn_resnet101 as resnet_101_fcn
import learn2learn as l2l
import segmentation_models_pytorch as SM
from sklearn.metrics import f1_score as score
import torchmetrics
import torchvision as vision
import argparse


def scale_8bit(data, cmin=None, cmax=None, high=255, low=0):
    """
    Converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255). If the input image already has 
    dtype uint8, no scaling is done.
    :param data: 16-bit image data array
    :param cmin: bias scaling of small values (def: data.min())
    :param cmax: bias scaling of large values (def: data.max())
    :param high: scale max value to high. (def: 255)
    :param low: scale min value to low. (def: 0)
    :return: 8-bit image data array
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        high = 255
    if low < 0:
        low = 0
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

def std_stretch_data(data, n=2.5):
    """Applies an n-standard deviation stretch to data."""

    mean, d = data.mean(), data.std() * n
    new_min = math.floor(max(mean - d, data.min()))
    new_max = math.ceil(min(mean + d, data.max()))
    
    data = np.clip(data, new_min, new_max)
    data = (data - data.min()) / (new_max - new_min)
    data = data*255
    return data.astype(np.uint8)

def std_stretch_all(img, std = 2.5, chanell_order = "last"):
    if chanell_order == "first":
        stacked = np.dstack((std_stretch_data(img[2,:,:], std), std_stretch_data(img[1,:,:], std),std_stretch_data(img[0, :,:], std)))
    else:
        stacked =  np.dstack((std_stretch_data(img[:,:,2], std), std_stretch_data(img[:,:,1], std),std_stretch_data(img[:,:,0], std)))
    return stacked

def read_images(im, scale=True, stretch=True):
    img = imread(im)
    if scale:
        img = scale_8bit(img)
    if stretch:
        img = std_stretch_all(img)
    return img
    
def read_labels(IM):
    Im = imread(IM)
    Im = Im.astype(np.uint8)
    return Im

def t_diff(a, b):
    t_diff = relativedelta(b, a) 
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

def find_area(contr):
    '''computes the areaa of each countour'''
    c = np.expand_dims(contr.astype(np.float32), 1)
    c = cv2.UMat(c)
    area = cv2.contourArea(c)
    return area



class CustomImageDataset(Dataset):
    def __init__(self, imgs, lbl, transform=None):
        self.imgs = imgs  # glob(root_dir + '/*.tif')
        self.lbls = lbl # [im.replace('images', 'labels') for im in self.imgs]
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lb_path = self.lbls[idx]
        image = read_images(img_path)
        if self.transform is not None:
            image = transform(image)
        image = torch.from_numpy(np.array([image[:,:,0], image[:,:,1], image[:,:,2]]))
        label = read_labels(lb_path)
        label = label[None,:]  # expand channel dimensions
        return image, label

def train_test_split(images,labels, valid_ratio=0.1, sampling='systematic'):
    assert len(images) == len(labels), 'images and labels in the folder are not the same'
    n = int(len(images)*valid_ratio)
    
    if sampling == 'systematic':
        interval = round(len(images)/n)
        sample_inds = list(range(0, len(images), interval))
        valid_image_samples = [images[idx] for idx in sample_inds if idx < len(images)]
        valid_label_samples = [labels[idx] for idx in sample_inds if idx < len(images)]
        assert len(valid_image_samples) == len(valid_label_samples), 'sample images and sample labels are not the same'
    elif sampling == 'random':
        full_inds = list(range(0, len(images)))
        random.seed(1)
        sample_inds = random.sample(full_inds, n)
        valid_image_samples = [images[idx] for idx in sample_inds if idx < len(images)]
        valid_label_samples = [labels[idx] for idx in sample_inds if idx < len(images)]

        assert len(valid_image_samples) == len(valid_label_samples), 'sample images and sample labels are not the same'
    else:
        raise ValueError(f'The sampling method {sampling} not known')
        
    train_images = list(set(images).symmetric_difference(set(valid_image_samples)))
    train_labels = list(set(labels).symmetric_difference(set(valid_label_samples)))
    
    assert len(train_images) == len(train_labels), 'training image and labels are not the same'
    
    return (train_images, train_labels), (valid_image_samples, valid_label_samples)  



def adaptation_split(images,labels, valid_ratio=0.1, train_ratio = 0.1, sampling='systematic'):
    assert len(images) == len(labels), 'images and labels in the folder are not the same'
    
    if sampling =='systematic':
        vl = int(len(images)*valid_ratio)
        tr = int(len(images)*train_ratio)
        tst = len(images) - (vl+tr)

        vl_inter = round(len(images)/vl)  
        vl_ind = list(range(0, len(images), vl_inter))     # validation index
                
        all_inds = [i for i in range(len(images)) if i not in vl_ind]

        tr_inter = round(len(images)/tr)
        tr_ind_f = list(range(1, len(all_inds), tr_inter))      # training index
        tr_ind = [all_inds[i] for i in tr_ind_f]

        all_inds = list(range(len(images)))
        tst_ind = list(set(all_inds).symmetric_difference(set(vl_ind+tr_ind)))  # test index

        valid_image = [images[idx] for idx in vl_ind if idx < len(images)]
        valid_label = [labels[idx] for idx in vl_ind if idx < len(images)]

        train_image = [images[idx] for idx in tr_ind if idx < len(images)]
        train_label = [labels[idx] for idx in tr_ind if idx < len(images)]

        test_image = [images[idx] for idx in tst_ind if idx < len(images)]
        test_label = [labels[idx] for idx in tst_ind if idx < len(images)]
    else:
        raise ValueError('The sample approch {sampling} is not implemented for adaptation data')

    return (train_image, train_label), (valid_image, valid_label) , (test_image, test_label)



class Adaptation_data_loader:
    def __init__(self, root, sampling='systematic', batch_size=8,v_size=0.1, tr_size = 0.1):
        self.root = root
        self.sampling = sampling
        self.batch_size = batch_size
        self.v_size=v_size
        self.tr_size = tr_size
        self.images = sorted(glob(root + '/images'+ '/*.tif'))
        self.labels = [im.replace('images', 'labels') for im in self.images]
        
        assert len(self.images) == len(self.labels), 'Warning, imgaes and labels have different length'
        
        # adaptation sampling
        train, valid, test = adaptation_split(self.images,
                                              self.labels, 
                                              valid_ratio=self.v_size, 
                                              train_ratio = self.tr_size, 
                                              sampling=self.sampling)
        # put sampled to the dataset
        self.train_samples = CustomImageDataset(train[0], train[1])
        self.valid_images = CustomImageDataset(valid[0], valid[1])
        self.test_images = CustomImageDataset(test[0], test[1])
        
#         self.train_dataset = CustomImageDataset(img, lbl) # after sampling
#         self.valid_dataset = CustomImageDataset(img, lbl) # after sampling
#         self.test_dataset = CustomImageDataset(img, lbl) # after sampling
        
    # put the dataset to the data loader
    def train_loader(self):
        return DataLoader(self.train_samples, self.batch_size, drop_last=True, num_workers=0, shuffle=True)
    
    def valid_loader(self):
        return DataLoader(self.valid_images, self.batch_size, drop_last=True, num_workers=0, shuffle=True)
    
    def test_loader(self):
        return DataLoader(self.test_images, self.batch_size, drop_last=True, num_workers=0, shuffle=True)


class custom_data_loader:
    def __init__(self, root_dir, batch_size=10, valid_ratio=0.1, sampling='systematic'):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        self.sampling = sampling
        self.folders = os.listdir(self.root_dir)
        self.image_folders = [root_dir + '/' + folder + '/'+ 'images' for folder in self.folders]
        self.task_datasets = []
        self.meta_loaders = []   # meta loader
        self.train_imgs = []
        self.train_lbls = []
        self.valid_imgs = []
        self.valid_lbls = []
        
        for image_folder in self.image_folders:
            images = sorted(glob(image_folder + '/*.tif'))
            labels = [im.replace('images', 'labels') for im in images]
            train, valid = train_test_split(images=images,
                                            labels=labels, 
                                            valid_ratio=self.valid_ratio, 
                                            sampling=self.sampling)
            
            task_dataset  = CustomImageDataset(train[0], train[1])
            
            self.task_datasets.append(task_dataset)
            
            self.train_imgs+=train[0] # append it for classical training and later use
            self.train_lbls += train[1] 
            self.valid_imgs += valid[0] # append it for both meta and classical training 
            self.valid_lbls += valid[1] 
            
        self.task_datasets.sort(key=len)  # sort the datasets according to length with acending order
        
        for i in range(len(self.task_datasets)): # this is for meta training 
            if i !=len(self.task_datasets)-1:
                self.meta_loaders.append(cycle(DataLoader(self.task_datasets[i], self.batch_size,drop_last=True, num_workers=0,shuffle=True)))                                      
            else:
                self.meta_loaders.append(DataLoader(self.task_datasets[i], self.batch_size, drop_last=True, num_workers=0, shuffle=True))
        
        self.valid_task = CustomImageDataset(self.valid_imgs, self.valid_lbls)
        self.valid_loader = DataLoader(self.valid_task, self.batch_size, drop_last=True,num_workers=0, shuffle=True)
        
        self.classic_task = CustomImageDataset(self.train_imgs, self.train_lbls)
        self.classic_loader = DataLoader(self.classic_task, self.batch_size, drop_last=True, num_workers=0, shuffle=True)
        
    def load_meta_dataset(self):
        return self.task_datasets
    
    def load_classic_dataset(self):
        return self.classic_task
    
    def load_valid_dataset(self):
        return self.valid_task

    def load_train_loader(self, partition='meta'):
        if partition == 'meta':
            return self.meta_loaders
        elif partition == 'classic':
            return self.classic_loader
        else:
            raise ValueError(f'partition type {partition} not known')

    def load_valid_loader(self):
        return self.valid_loader


def get_resnet_model(fam='resnet101', pretrained=True, num_class=1, train_backbone=True):
    
    assert fam in ['resnet50', 'resnet101'], 'The model family should be either resnet50/101'
    
    if fam == 'resnet50':
        model = resnet_50_fcn(pretrained = pretrained,
                              pretrained_backbone = train_backbone)
    elif fam == 'resnet101':
        model = resnet_101_fcn(pretrained = pretrained,
                               pretrained_backbone = train_backbone)
    if num_class != 21:
        model.classifier[4] = nn.Conv2d(
            in_channels=512,
            out_channels=num_class,
            kernel_size=1,
            stride=1)
        model.aux_classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=num_class,
            kernel_size=1,
            stride=1)
    return model



def getVGG19(backbon='vgg19', weights='imagenet', nchannels=3, nclass=1, activation = 'sigmoid'):
    vgg19 = SM.FPN(encoder_name = backbon,
                   encoder_weights = weights,
                   in_channels = nchannels,
                   decoder_dropout=0.0,
                   classes  = nclass,
                   activation = activation)
    return vgg19


def Unet(backbon='resnet101', weights=None, nchannels=3, nclass=1, activate = False, keep_identity = False):
    assert weights in [None, 'imagenet'], 'initialization should be either "None" or "imagenet"'
    unet = SM.Unet(encoder_name = backbon,
                   encoder_weights = weights,
                   in_channels = nchannels,
                   classes = nclass)
    if not keep_identity:
        if activate:
            unet.segmentation_head = nn.Sequential(nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.Sigmoid())
        else:
            unet.segmentation_head = nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    return unet

class DeepLab(nn.Module):
    def __init__(self, weights, activate=False):
        super(DeepLab, self).__init__()
        self.weights = weights
        self.activate = activate
        self.model = self.defineModel()
        self.sigmoid = nn.Sigmoid()
        
    def defineModel(self):
        if self.weights == 'imagenet':
            backbone_weights = True
        elif self.weights == None:
            backbone_weights = False
        else:
            raise ValueError('Weight initialization not known')
        # aux_loss == False just to drop auxiliary loss specified in the source code/document
        deepmodel = torchvision.models.segmentation.deeplabv3_resnet101(pretrained = False,
                                                                        progress = True,
                                                                        num_classes= 1,
                                                                        aux_loss = False,
                                                                        pretrained_backbone = backbone_weights)
        return deepmodel
    
    def forward(self, x):
        x = self.model(x)
        x = x['out'] 
        if self.activate:
            return self.sigmoid(x)
        else:
            return x


def set_seeds(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2,-1), activate=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
        self.activate = activate
        if self.activate:
            self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, y, tresh=0.5, loss=False):
        if self.activate:
            x = self.sigmoid(x)
        if tresh is None:
            x = torch.argmax(x, dim=1)
        else:
            x = (x>=tresh).float()*1
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        if loss:
            return 1 - dc
        else:
            return dc


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', input_type='activation'):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.input_type = input_type
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        implementation of focal loss. The input should be any convolution map without activation
        Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        """
        inputs = inputs.float()
        targets = targets.float()
        if self.input_type =='activation':
            p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class BinaryMetrics(nn.Module):
    """Caculates binary accuracy metrics, accuracy, dice, precision, recall and specificity.
    """
    def __init__(self, eps=1e-5, activation='none'):
        super(BinaryMetrics, self).__init__()
        self.eps = eps
        self.activation = activation

    def compute_metrics(self,pred, gt):
        output = pred.view(-1, )
        target = gt.view(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)
        tpr =  tp/(tp + fn + self.eps)    # true positive rate
        fpr =  fn/(fn + tn + self.eps)    # false positive rate 
        # True positive rate and false positive rates will be sued later for ROC curve plot
        
        # Intersection over Union
        intersetion = torch.sum(pred*gt, dim=(1,2,3))
        union = torch.sum(pred, dim=(1,2,3)) + torch.sum(gt, dim=(1,2,3))
        
        IOU = torch.mean((intersetion+self.eps)/(union-intersetion + self.eps))

        return pixel_acc, dice, precision, specificity, recall, IOU, tpr, fpr

    def forward(self,y_pred, y_true):
        if self.activation in [None, 'none']:
            activation_fn = lambda x: (x>0.5).float()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid_with_reshold":
            sigmoid_pred = nn.Sigmoid()(y_pred)
            activated_pred = (sigmoid_pred > 0.5).float()
        else:
            raise ValueError(f"activation {self.activation} not supported!")
        pixel_acc, dice, precision, specificity, recall, iou, tpr, fpr = self.compute_metrics(y_true, activated_pred)
        return [pixel_acc, dice, precision, specificity, recall, iou, tpr, fpr]



class ObjectAccuracy:
    def __init__(self, batched=True, iou_cut_off=0.5, logits=True, eps=0.00001):
        self.batched = batched
        self.logits = logits
        self.eps = eps
        self.names = ['rf_count','prd_count','TP', 'FP', 'FN', "F1", "CR", "CM", "QL"]
        self.totals = ['rf_count','prd_count','TP', 'FP', 'FN']
        self.means = ["F1", "CR", "CM", "QL"]
        self.mask_cut_off = 0.5
        self.iou_cut_off = iou_cut_off
        
    @torch.no_grad()
    def compute(self,pred, ref):
        assert pred.shape == ref.shape, 'shapes are not the same'
        if self.batched:
            if self.logits:
                pred = nn.Sigmoid()(pred)
                pred = (pred>=self.mask_cut_off).float()
            ref = torch.squeeze(ref)
            pred = torch.squeeze(pred)
            
            b = ref.shape[0]
            metrics = [self.object_metric(pred[i], ref[i], iou_thresh=self.iou_cut_off, eps=self.eps, compute_pr=True) for i in range(b)]
            
            summary = []
            for i in self.names:
                val = [a[i] for a in metrics]
                if i in self.totals:
                    val = np.nansum(val)
                else:
                    val = np.nanmean(val)
                summary.append(val)
        return summary
    
    def object_metric(self, P, R, iou_thresh=0.5, eps= 0.00001, compute_pr=True):
        '''
        P: predicted mask
        R: reference mask
        compute_pr: boolean, whether to compute precision and recall per single chip
        returns TP, FP and FN counts
        '''
        assert P.shape == R.shape, 'Predicted and reference shapes are not the same'

        R = self.pad(R) # pad
        R = measure.find_contours(R, 0.5)  # generate contours
        R = [make_valid(geometry.Polygon(pol)) for pol in R]  # convert to shapely polygon

        P = self.pad(P)
        P = measure.find_contours(P, 0.5)
        P = [make_valid(geometry.Polygon(pol)) for pol in P]

        # define empty TP, FP and FN containers
        TP = 0
        FP = 0
        FN = 0

        ref_count = len(R)
        pred_count = len(P)

        if len(P)>0 and len(R) == 0:  # when there is predition and no object reference
            FP = FP + len(P)
        elif len(P) == 0 and len(R)>0: # when there is no object in prediction but there is on reference
            FN = FN+len(R)
        elif len(P) == 0 and len(R) == 0: # when both prediction and reference have no any thing
            pass  #  do nothing as there is no any further prediction
        else:
            for i in range(len(P)):
                p = P[i]
                area = [p.intersection(robj).area for robj in R]  # intersection
                whole = [p.area + robj.area for robj in R]   # union +intersection
                union = [whole[i]-area[i] + self.eps for i in range(len(area))]   # union with safe division factor
                iou = [(area[i]+eps)/union[i] for i in range(len(area))]   # inntersection over union
                truth = [a>=iou_thresh for a in iou]   # tresholding

                if True in truth:
                    TP+=1
                else:
                    FP+=1
            for i in range(len(R)):
                r = R[i]
                area = [r.intersection(pobj).area for pobj in P]
                whole = [r.area + pobj.area for pobj in P]
                union = [whole[i]-area[i] + self.eps for i in range(len(area))]
                iou = [(area[i]+self.eps)/union[i] for i in range(len(area))]   # inntersection over union
                truth = [a<iou_thresh for a in iou]   # tresholding
                if True in truth:
                    FN += 1
        if not compute_pr:
            return {'ref_count':ref_count,'pred_count':pred_count, 'TP':TP, 'FP':FP, 'FN':FN}

        else:
            # constant for safe division
            corr = TP/(TP+FP + eps)  # prec
            comp = TP/(TP+FN + eps)  # reca
            qual = TP/(TP+FP+FN+eps) 
            of1 = 2*((corr*comp)/(corr+comp+eps))
            
            return {'rf_count':ref_count,'prd_count':pred_count,'TP':TP, 'FP':FP, 'FN':FN, "F1":of1, "CR":corr, "CM":comp, 'QL':qual}

    def pad(self, mask):
        return np.pad(mask.cpu(), ((1, 1), (1, 1)), 'minimum')
    
    def repad(self, mask):
        return mask[1:-1, 1:-1]


bce_fn = nn.BCEWithLogitsLoss()
dice_fn = DiceLoss()
def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred, y_true)  # y_pred.sigmoid(), y_true
    return 0.8*bce + 0.2*dice



@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.float().to(device), target.float().to(device)
        output = model(image)['out']
        loss = loss_fn(output, target)
        losses.append(loss.item())
    return np.array(losses).mean()


def core(t,p):
    assert p.shape == t.shape, 'shape mismatch'
    p = p.view(-1)
    t = t.view(-1)
    return score(t,p)

def F1_score(target,pred,treshold=0.5,binary=True):
    if binary:
        pred = torch.squeeze(pred)
        pred = (pred>treshold).float()*1 # convert to binary
        target = torch.squeeze(target)
    
    tot = [core(target[i].cpu(), pred[i].cpu()) for i in range(pred.shape[0])]
    tot = np.sum(tot)/pred.shape[0]
    
    return tot



def fast_adapt(learner, task_data, criterion, adaptation_steps, device): # not used for automation
    X, Y = task_data[0], task_data[1]
    x_support, y_support = X[:m].float().to(device), Y[:m].float().to(device)
    x_query, y_query = X[m:].float().to(device), Y[m:].float().to(device)
    for step in range(adaptation_steps):
        logits = learner(x_support)
        loss = criterion(logits['out'], y_support)
        learner.adapt(loss) 
    querry_preds = learner(x_query)
    querry_loss = criterion(querry_preds['out'], y_query)
    return querry_loss


class MAML:
    def __init__(self, root, weight_path, batch_size=8, adapt_step=1, adapt_lr=0.05, meta_lr=0.001,epochs=100):
        self.root = root
        self.batch_size = batch_size
        self.adapt_lr = adapt_lr
        self.meta_lr = meta_lr
        self.epochs = epochs
        self.adapt_step = adapt_step
        self.ql = int(self.batch_size/2)                                             # querry length
        self.weight_path = weight_path
        self.is_cuda = torch.cuda.is_available()
        self.device =  torch.device("cuda" if self.is_cuda else "cpu")
        self.DATA = custom_data_loader(self.root, self.batch_size)
        self.acc_fn = BinaryMetrics(activation="sigmoid_with_reshold").to(self.device) # binary accuracy metrics
        self.loss_fn = SigmoidFocalLoss().to(self.device)                              # sigmoid focal loss with logits
        self.n_tasks = len(os.listdir(root))                                           # number of tasks
   
    def trainMML(self, init_weight='imagenet'):
        assert init_weight in [None, 'imagenet'], 'initialization should be either "None" or "imagenet"'
        torch.cuda.empty_cache()
        set_seeds()
        if init_weight == None:
            end_fold = 'weight_random'
        elif init_weight == 'imagenet':
            end_fold = 'weight_imagenet'
        else:
            raise ValueError('path not known')
        save_dir = self.weight_path + '/train/maml/'+ end_fold
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=False)
        
        self.TL = []                         # meta train loss
        self.TA = [[],[],[],[],[],[],[],[]]  # [[]]*8
        self.VL = []                         # meta validation loss
        self.VA = [[],[],[],[],[],[],[],[]]  #[[]]*8 # validation accuracy
        
        
        # model = Unet(nclass=1, weights=init_weight, activate=False)
        model = DeepLab(init_weight)
        maml = l2l.algorithms.MAML(model, lr=self.adapt_lr, first_order=True, allow_nograd=True)
        maml.to(self.device)
      
        if isinstance(self.meta_lr, list) and len(self.meta_lr) == 2:
            max_lr = self.meta_lr[0]
            min_lr = self.meta_lr[1]
        else:
            max_lr = self.meta_lr
            min_lr = 0.000001
        opt = torch.optim.SGD(maml.parameters(), max_lr)
        
        meta_scheduler = CosineAnnealingLR(opt, T_max=max_lr, eta_min=min_lr)
        train_loader = self.DATA.load_train_loader()
        valid_loader = self.DATA.load_valid_loader()
        
        best_valid = 0
        for epoch in range(1, self.epochs+1):
            for j, data in enumerate(zip(*train_loader)): # this iterates through all data loaders(datasets) per epoch
                meta_train_loss = 0
                meta_train_acc = [0, 0, 0, 0, 0, 0, 0, 0]  # [0]*6
                control = 0                                # number of effective tasks
                for i in range(len(data)):                  #  iterate through each tasks
                    X, Y = data[i][0], data[i][1]            # images and labels respectively
                    x_support, y_support = X[:self.ql].float().to(self.device), Y[:self.ql].float().to(self.device)
                    x_query, y_query = X[self.ql:].float().to(self.device), Y[self.ql:].float().to(self.device)
                    learner = maml.clone()
                    for _ in range(self.adapt_step):              # number of adaptation steps
                        logits = learner(x_support)
                        loss = self.loss_fn(logits, y_support)
                    learner.adapt(loss)                            # inner loop update
                    query_logits = learner(x_query)
                    query_loss = self.loss_fn(query_logits, y_query)
                    querry_acc = self.acc_fn(query_logits, y_query)
                    
                    meta_train_loss += query_loss
                    for n in range(8):
                        meta_train_acc[n]+=querry_acc[n].item()
                    control += 1
                    print(f'inner_loss: {query_loss.item()}, inner_acc: {querry_acc[0].item()}')
                meta_train_loss = meta_train_loss / control
                opt.zero_grad()
                meta_train_loss.backward()                                  # retain_graph=True
                opt.step()
                
                with torch.no_grad():
                    self.TL.append(meta_train_loss.item())
                    meta_train_acc[:] = [a/control for a in meta_train_acc]
                    for n in range(8):
                        self.TA[n].append(meta_train_acc[n])
                    print(f'+++ {epoch}: step: {j}, meta-loss: {meta_train_loss.item()}, mat-accu:{meta_train_acc[0]} +++')
                    
            # change meta learning rate  
            meta_scheduler.step()
            # validate the model
            val_loss = 0
            val_acc = [0, 0, 0, 0, 0, 0, 0, 0]    # [0]*8
            control = 0
    
            for image, target in valid_loader:
                image, target = image.float().to(self.device), target.float().to(self.device)
                output = maml(image)
                vloss = self.loss_fn(output, target)
                vacc = self.acc_fn(output, target)
                val_loss+=vloss.item()
                for n in range(8):
                    val_acc[n]+=vacc[n].item()
                control+=1
            val_loss = val_loss/control
            val_acc[:] = [a/control for a in val_acc]
            
            self.VL.append(val_loss)
            for n in range(8):
                self.VA[n].append(val_acc[n])

            print(f'+++ {epoch}: step: {j}, valid-loss: {val_loss}, valid_acc {val_acc[0]} +++')
            
            if epoch == 1:
                name =save_dir + '/maml_checkpoint.pth'
                torch.save(maml.state_dict(), name)
                best_valid = self.VL[-1]
            else:
                if self.VL[-1]<=best_valid:
                    name = save_dir + '/maml_checkpoint.pth'   # self.weigth_path 
                    torch.save(maml.state_dict(), name)
                    best_valid = self.VL[-1]
                else:
                    pass
        
        touts = {'tloss':self.TL, 
                 'tacc':self.TA[0], 
                 'dice':self.TA[1], 
                 'precision':self.TA[2], 
                 'specificity':self.TA[3], 
                 'recall':self.TA[4],
                 'IOU':self.TA[5],
                 'tpr':self.TA[6],
                 'fpr':self.TA[7]}

        vouts = {'vloss':self.VL, 
                 'vacc':self.VA[0], 
                 'dice':self.VA[1], 
                 'precision':self.VA[2], 
                 'specificity':self.VA[3], 
                 'recall':self.VA[4],
                 'IOU':self.VA[5],
                 'tpr':self.VA[6],
                 'fpr':self.VA[7]}
        
        np.save(save_dir + '/train_summary.npy', touts)
        np.save(save_dir + '/valid_summary.npy', vouts)
            
    def trainCLASSIC(self, init_weight='imagenet'):
        
        assert init_weight in [None, 'imagenet'], 'initialization should be either "None" or "imagenet"'
        torch.cuda.empty_cache()
        set_seeds()
        
        if init_weight == None:
            end_fold = 'weight_random'
        elif init_weight == 'imagenet':
            end_fold = 'weight_imagenet'
        else:
            raise ValueError('path not known')
        save_dir = self.weight_path + '/train/classic/'+ end_fold
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=False)
    
        self.TL = []                         # meta train loss
        self.TA = [[],[],[],[],[],[],[],[]]  # [[]]*8
        self.VL = []                         # meta validation loss
        self.VA = [[],[],[],[],[],[],[],[]]  # [[]]*8 # validation accuracy
        
        # model = Unet(nclass=1, weights=init_weight, activate=False)
        model = DeepLab(init_weight)
        model.to(self.device)
        
        if isinstance(self.meta_lr, list) and len(self.meta_lr) == 2:
            max_lr = self.meta_lr[0]
            min_lr = self.meta_lr[1]
        else:
            max_lr = self.meta_lr
            min_lr = 0.000001
    
        opt = torch.optim.SGD(model.parameters(), max_lr)
        scheduler = CosineAnnealingLR(opt, T_max=max_lr, eta_min=min_lr)
        
        train_loader = self.DATA.load_train_loader(partition='classic')
        valid_loader = self.DATA.load_valid_loader()
        
        best_valid = 0
        for epoch in range(1, self.epochs+1):
            epoch_loss = 0
            epoch_acc = [0, 0, 0, 0, 0, 0, 0, 0]   # [0]*8
            control = 0
            for j, data in enumerate(train_loader): # this iterates through all data loaders(datasets) per epoch
                opt.zero_grad()
                X, Y = data[0].float().to(self.device), data[1].float().to(self.device)
                logits = model(X)
                loss = self.loss_fn(logits, Y)
                acc = self.acc_fn(logits, Y)
                loss.backward()
                opt.step()
                
                with torch.no_grad():
                    epoch_loss+=loss.item()
                    for n in range(8):
                        epoch_acc[n] += acc[n].item()        
                print(f'step_loss: {loss.item()}, step_acc: {acc[0].item()}')
                control+=1
                
            with torch.no_grad():
                epoch_loss = epoch_loss / control
                epoch_acc[:] = [aa/control for aa in epoch_acc]
                self.TL.append(epoch_loss)
                for n in range(8):
                    self.TA[n].append(epoch_acc[n])
                print(f'+++ {epoch}: step: {j}, train_loss: {epoch_loss}, train_acc:{epoch_acc[0]} +++')
                
            scheduler.step()
            val_loss = 0
            val_acc = [0, 0, 0, 0, 0, 0, 0, 0]  # [0]*8
            control = 0
            with torch.no_grad():
                for image, target in valid_loader:
                    image, target = image.float().to(self.device), target.float().to(self.device)
                    output = model(image)
                    vloss = self.loss_fn(output, target)
                    vacc = self.acc_fn(output, target)
                    val_loss +=vloss.item()
                    for n in range(8):
                        val_acc[n]+=vacc[n].item()
                    control+=1
            val_loss = val_loss/control
            val_acc[:] = [bb/control for bb in val_acc]
            
            self.VL.append(val_loss)
            for n in range(8):
                self.VA[n].append(val_acc[n])
            print(f'+++ {epoch}: step: {j}, valid_loss: {val_loss}, valid_acc: {val_acc[0]}')
            
            if epoch ==1:
                name = save_dir + '/classic_checkpoint.pth' # self.weigth_path 
                torch.save(model.state_dict(), name)
                best_valid = self.VL[-1]
            else:
                if self.VL[-1]<=best_valid:
                    name = save_dir + '/classic_checkpoint.pth'   # self.weigth_path
                    torch.save(model.state_dict(), name)
                    best_valid = self.VL[-1]
                else:
                    pass
                
        touts = {'tloss':self.TL, 
                 'acc':self.TA[0], 
                 'dice':self.TA[1], 
                 'precision':self.TA[2], 
                 'specificity':self.TA[3], 
                 'recall':self.TA[4],
                 'IOU':self.TA[5],
                 'tpr':self.TA[6],
                 'fpr':self.TA[7]}

        vouts = {'tloss':self.VL, 
                 'acc':self.VA[0], 
                 'dice':self.VA[1], 
                 'precision':self.VA[2], 
                 'specificity':self.VA[3], 
                 'recall':self.VA[4],
                 'IOU':self.VA[5],
                 'tpr':self.VA[6],
                 'fpr':self.VA[7]}
        np.save(save_dir + '/train_summary.npy', touts)
        np.save(save_dir + '/valid_summary.npy', vouts)

        
class Adapter:
    def __init__(self, root, tr_size=0.1, v_size=0.1, lr=0.01, batch_size=8, epochs=20, weight_path=None):
        self.root = root
        self.tr_size = tr_size
        self.v_size = v_size
        self.lr = lr
        self.epochs = epochs
        self.weight_path = weight_path
        self.batch_size = batch_size
        self.is_cuda = torch.cuda.is_available()
        self.device =  torch.device("cuda" if self.is_cuda else "cpu")
        self.acc_fn = BinaryMetrics(activation="sigmoid_with_reshold").to(self.device) # binary accuracy metrics
        self.loss_fn = SigmoidFocalLoss().to(self.device)
        self.Data = Adaptation_data_loader(root= root,
                                           sampling='systematic',
                                           batch_size=self.batch_size,
                                           v_size=self.v_size,
                                           tr_size = self.tr_size)
    
        self.TL = []
        self.TA = [[],[],[],[],[],[],[],[]] # [[]]*8
        self.VL = []
        self.VA = [[],[],[],[],[],[],[],[]] #[[]]*8

    def adapt(self, init_weight=None, model_type='maml', checkpoint=None):
        
        assert init_weight in [None, 'imagenet'], 'initialization should be either "None" or "imagenet" as trained weights are initiated'
        torch.cuda.empty_cache()
        set_seeds()
        
        if init_weight is None:
            end_fold = 'weight_random'
        elif init_weight == 'imagenet':
            end_fold = 'weight_imagenet'
        else:
            raise ValueError('path not known')
        
        if model_type == 'maml':
            sub_fold = 'maml'
            # model = Unet(nclass=1, activate=False)
            model = DeepLab(init_weight)
            model = l2l.algorithms.MAML(model, lr=self.lr, first_order=True, allow_nograd=True)
            
        elif  model_type == 'classic':
            sub_fold = 'classic'
            # model = Unet(nclass=1, activate=False)
            model = DeepLab(init_weight)
        else:
            raise ValueError('model type {model_type} is known')
        
        save_dir = self.weight_path + '/' + sub_fold + '/' + end_fold   # self.weight_path + '/' + 'adapt/'+ sub_fold + '/' + end_fold
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))
            print(f'Checkpoint loaded from {checkpoint}')
        else:
            raise ValueError('Weight path can not be "None" for adaptation phase')
            
        if isinstance(self.lr, list) and len(self.lr) == 2:
            max_lr = self.lr[0]
            min_lr = self.lr[1]
        else:
            max_lr = self.lr
            min_lr = 0.000001
        optim = torch.optim.SGD(model.parameters(), max_lr)
        # scheduler = CosineAnnealingLR(optim, T_max=max_lr, eta_min=min_lr)
        
        model.train()
        model.to(self.device)
        adapt_loader = self.Data.train_loader()  # adapt traner
        valid_loader = self.Data.valid_loader()  # adapt validtor
        
        best_valid = 0
        for epoch in range(1, self.epochs+1):
            epoch_loss = 0
            epoch_acc = [0, 0, 0, 0, 0, 0, 0, 0] # [0]*8
            control = 0
            
            for j, data in enumerate(adapt_loader): # this iterates through all data loaders(datasets) per epoch
                X, Y = data[0].float().to(self.device), data[1].float().to(self.device)
                optim.zero_grad()
                logits = model(X)
                loss = self.loss_fn(logits, Y)
                acc = self.acc_fn(logits, Y)
                loss.backward()
                optim.step()
                epoch_loss+=loss.item()
                
                with torch.no_grad():
                    for i in range(8):
                        epoch_acc[i]+=acc[i].item()
                control+=1
                print(f'Step-loss: {loss.item()}, step_acc: {acc[0].item()}')
            
            with torch.no_grad():
                epoch_loss = epoch_loss / control
                self.TL.append(epoch_loss)
                epoch_acc[:] = [aa/control for aa in epoch_acc]
                for n in range(8):
                    self.TA[n].append(epoch_acc[n])
                print(f'+++ {epoch}: step: {j}, train_loss: {epoch_loss}, train_acc: {epoch_acc[0]} +++')
                
            # scheduler.step()
            val_loss = 0
            val_acc = [0, 0, 0, 0, 0, 0, 0, 0] # [0]*8 
            control = 0

            with torch.no_grad():
                for image, target in valid_loader:
                    image, target = image.float().to(self.device), target.float().to(self.device)
                    output = model(image)
                    vloss = self.loss_fn(output, target)
                    vacc = self.acc_fn(output, target)
                    val_loss+=vloss.item()
                    for n in range(8):
                        val_acc[n]+=vacc[n].item()
                    control+=1
            val_loss = val_loss/control
            val_acc[:] = [bb/control for bb in val_acc]
            self.VL.append(val_loss)
            for n in range(8):
                self.VA[n].append(val_acc[n])
            print(f'+++ {epoch}: step: {j}, val_loss: {val_loss}, va_acc {val_acc[0]}')

            if epoch == 1:
                name = save_dir + '/{}_adapt_checkpoint.pth'.format(epoch)
                torch.save(model.state_dict(), name)
                best_valid = self.VL[-1]
            else:
                if self.VL[-1]<=best_valid:
                    name =  save_dir + '/{}_adapt_checkpoint.pth'.format(epoch)  # self.weigth_path
                    torch.save(model.state_dict(), name)
                    best_valid = self.VL[-1]
                else:
                    pass
                
        touts = {'ad_tloss':self.TL, 
                 'ad_acc':self.TA[0], 
                 'dice':self.TA[1], 
                 'precision':self.TA[2], 
                 'specificity':self.TA[3], 
                 'recall':self.TA[4],
                 'IOU':self.TA[5],
                 'tpr':self.TA[6],
                 'fpr':self.TA[7]}

        vouts = {'adv_loss':self.VL, 
                 'adv_acc':self.VA[0], 
                 'dice':self.VA[1], 
                 'precision':self.VA[2], 
                 'specificity':self.VA[3], 
                 'recall':self.VA[4],
                 'IOU':self.VA[5],
                 'tpr':self.VA[6],
                 'fpr':self.VA[7]}
        np.save(save_dir + '/train_summary.npy', touts)
        np.save(save_dir + '/valid_summary.npy', vouts)
        

class Tester:
    def __init__(self, root, out_path, batch_size=20, tr_size=0.1, v_size=0.1):
        self.root = root
        self.batch_size = batch_size
        
        self.is_cuda = torch.cuda.is_available()
        self.device =  torch.device("cuda" if self.is_cuda else "cpu")
        self.v_size = v_size
        self.tr_size = tr_size
        self.out_path = out_path
        self.lr = 0.0001 # this is just to remove default error, during test time its not used
        self.Data = Adaptation_data_loader(root=self.root,
                                           sampling='systematic',
                                           batch_size=self.batch_size,
                                           v_size=self.v_size,
                                           tr_size = self.tr_size)
        
        
        self.pix_ac_fn = BinaryMetrics(activation="sigmoid_with_reshold").to(self.device)
        self.obj_acc_fn = ObjectAccuracy()
        
    def test(self, init_weight=None, model_type='maml', checkpoint= None, report=True):
        
        assert init_weight in [None, 'imagenet'], 'initialization should be either "None" or "imagenet"'
        torch.cuda.empty_cache()
        set_seeds()
        
        if init_weight is None:
            end_fold = 'weight_random'
        elif init_weight == 'imagenet':
            end_fold = 'weight_imagenet'
        else:
            raise ValueError('path not known')
        
        if model_type == 'maml':
            sub_fold = 'maml'
            # model = Unet(nclass=1, activate=False)
            model = DeepLab(init_weight)
            model = l2l.algorithms.MAML(model, lr=self.lr, first_order=True, allow_nograd=True)
        elif  model_type == 'classic':
            sub_fold = 'classic'
            # model = Unet(nclass=1, activate=False)
            model = DeepLab(init_weight)
        else:
            raise ValueError('model type {model_type} is known')
        
        model.load_state_dict(torch.load(checkpoint)) # load weight adapted weight
        
        save_dir = self.out_path + '/' + sub_fold + '/' + end_fold
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        model.eval()
        model.to(self.device)
        test_loader = self.Data.test_loader()
        
        acc = [0, 0, 0, 0, 0, 0, 0, 0] # [0]*8
        obj = [0, 0, 0, 0, 0, 0, 0, 0, 0] # [0]*9
        control = 0
        
        for data in test_loader:
            X, Y = data[0].float().to(self.device), data[1].float().to(self.device)
            logits = model(X)
            pix_acc = self.pix_ac_fn(logits, Y) # iou function 
            obj_acc = self.obj_acc_fn.compute(logits, Y) # f1 function
            
            for n in range(len(pix_acc)):
                acc[n]+=pix_acc[n].item()
            for n in range(len(obj_acc)):
                obj[n]+=obj_acc[n].item()
            control+=1
        
        metric_names1 = ['pixel-acc',
                         'pixel-dice',
                         'pixel-precision',
                         'pixel-specificity',
                         'pixel-recall',
                         'pixel-iou',
                         'pixel-tpr',
                         'pixel-fpr']
        
        metric_names2 = ['ref_count',
                         'pred_count',
                         'object-TP',
                         'object-FP',
                         'object-FN',
                         'object-F1',
                         'object-CR',
                         'object-CM',
                         'object-QL']
        
        pix_metric = {}
        obj_metric = {}
        
        for j, val in enumerate(acc):
            pix_metric[metric_names1[j]] = acc[j]/control
        for j, val in enumerate(obj):
            if j<5:
                obj_metric[metric_names2[j]] = obj[j]
            else:
                obj_metric[metric_names2[j]] = obj[j]/control
                
        final_metric = pix_metric.update(obj_metric)
        
        if report:
            print(final_metric)
            
            # print('== Pixel accuracy ==')
            # for key in metric_names1:
            #     print(f'{key} : {final_metric[key]}')
            # print('== Object accuracy ==')
            # for key in metric_names2:
            #     print(f'{key} : {final_metric[key]}')
                
        np.save(save_dir + '/accuracy_summary.npy', final_metric)

class BaseTrainer:
    def __init__(self, root, weight_path, batch_size=20, lr=0.1,epochs=100, v_size=0.1, tr_size=0.1):
        self.root = root
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.v_size = v_size
        self.tr_size = tr_size
        self.weight_path = weight_path
        self.is_cuda = torch.cuda.is_available()
        self.device =  torch.device("cuda" if self.is_cuda else "cpu")
        self.acc_fn = BinaryMetrics(activation="sigmoid_with_reshold").to(self.device) # binary accuracy metrics
        self.loss_fn = SigmoidFocalLoss().to(self.device)                              # sigmoid focal loss with logit
        self.Data = Adaptation_data_loader(root= root,
                                           sampling='systematic',
                                           batch_size=self.batch_size,
                                           v_size=self.v_size,
                                           tr_size = self.tr_size)
           
    def trainbase(self, init_weight='imagenet'):
        
        assert init_weight in [None, 'imagenet'], 'initialization should be either "None" or "imagenet"'
        torch.cuda.empty_cache()
        set_seeds()
        
        if init_weight == None:
            end_fold = 'weight_random'
        elif init_weight == 'imagenet':
            end_fold = 'weight_imagenet'
        else:
            raise ValueError('path not known')
        save_dir = self.weight_path + '/train/'+ end_fold
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=False)
    
        self.TL = []                         # meta train loss
        self.TA = [[],[],[],[],[],[],[],[]]  # [[]]*8
        self.VL = []                         # meta validation loss
        self.VA = [[],[],[],[],[],[],[],[]]  # [[]]*8 # validation accuracy
        
        # model = Unet(nclass=1, weights=init_weight, activate=False)
        model = DeepLab(init_weight)
        model.to(self.device)
        
        if isinstance(self.lr, list) and len(self.lr) == 2:
            max_lr = self.lr[0]
            min_lr = self.lr[1]
        else:
            max_lr = self.lr
            min_lr = 0.000001
    
        opt = torch.optim.SGD(model.parameters(), max_lr)
        meta_scheduler = CosineAnnealingLR(opt, T_max=max_lr, eta_min=min_lr)  # needs reference 
        
        train_data = self.Data.train_loader()  # adapt traner
        valid_data = self.Data.valid_loader()  # adapt validtor
         
        best_valid = 0
        for epoch in range(1, self.epochs+1):
            epoch_loss = 0
            epoch_acc = [0, 0, 0, 0, 0, 0, 0, 0]   # [0]*8
            control = 0
            for j, data in enumerate(train_data): # this iterates through all data loaders(datasets) per epoch
                opt.zero_grad()
                X, Y = data[0].float().to(self.device), data[1].float().to(self.device)
                logits = model(X)
                loss = self.loss_fn(logits, Y)
                acc = self.acc_fn(logits, Y)
                loss.backward()
                opt.step()
                
                with torch.no_grad():
                    epoch_loss+=loss.item()
                    for n in range(8):
                        epoch_acc[n] += acc[n].item()        
                print(f'step_loss: {loss.item()}, step_acc: {acc[0].item()}')
                control+=1
                
            with torch.no_grad():
                epoch_loss = epoch_loss / control
                epoch_acc[:] = [aa/control for aa in epoch_acc]
                self.TL.append(epoch_loss)
                for n in range(8):
                    self.TA[n].append(epoch_acc[n])
                print(f'+++ {epoch}: step: {j}, train_loss: {epoch_loss}, train_acc:{epoch_acc[0]} +++')

            val_loss = 0
            val_acc = [0, 0, 0, 0, 0, 0, 0, 0]  # [0]*8
            control = 0
            with torch.no_grad():
                for image, target in valid_data:
                    image, target = image.float().to(self.device), target.float().to(self.device)
                    output = model(image)
                    vloss = self.loss_fn(output, target)
                    vacc = self.acc_fn(output, target)
                    val_loss +=vloss.item()
                    for n in range(8):
                        val_acc[n]+=vacc[n].item()
                    control+=1
            val_loss = val_loss/control
            val_acc[:] = [bb/control for bb in val_acc]
            
            self.VL.append(val_loss)
            for n in range(8):
                self.VA[n].append(val_acc[n])
            print(f'+++ {epoch}: step: {j}, valid_loss: {val_loss}, valid_acc: {val_acc[0]}')
            
            if epoch ==1:
                name = save_dir + '/checkpoint.pth' # self.weigth_path 
                torch.save(model.state_dict(), name)
                best_valid = self.VL[-1]
            else:
                if self.VL[-1]<=best_valid:
                    name = save_dir + '/checkpoint.pth'   # self.weigth_path
                    torch.save(model.state_dict(), name)
                    best_valid = self.VL[-1]
                else:
                    pass
                
        touts = {'tloss':self.TL, 
                 'acc':self.TA[0], 
                 'dice':self.TA[1], 
                 'precision':self.TA[2], 
                 'specificity':self.TA[3], 
                 'recall':self.TA[4],
                 'IOU':self.TA[5],
                 'tpr':self.TA[6],
                 'fpr':self.TA[7]}

        vouts = {'tloss':self.VL, 
                 'acc':self.VA[0], 
                 'dice':self.VA[1], 
                 'precision':self.VA[2], 
                 'specificity':self.VA[3], 
                 'recall':self.VA[4],
                 'IOU':self.VA[5],
                 'tpr':self.VA[6],
                 'fpr':self.VA[7]}
        np.save(save_dir + '/train_summary.npy', touts)
        np.save(save_dir + '/valid_summary.npy', vouts)
