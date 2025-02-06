#!/usr/bin/env python
# coding: utf-8

# In[9]:
###########################
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
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
import json
import shutil
from shapely import geometry
from shapely.validation import make_valid
import os
import gc
import random
import math
from torchvision.io import read_image
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from itertools import cycle
import segmentation_models_pytorch as smp
import learn2learn as l2l
import argparse

############ Utility functions ############

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

class DictMerger:
    def __init__(self, inplace=False):
        self.inplace=inplace
    def merge(self, x):
        '''Merges two dictionaries. Plrease note that if there is simmilar keys, it only takes the last dictionary key'''
        #if not set(list(x.keys())).isdisjoint(set(list(y.keys()))):
                #raise Warning('The dictionaries has simmilar keys which the merger will only take key values from second dict')
        if self.inplace:
            for i in range(1, len(x)):
                x[0].update(x[i])
            return x[0]
        else:
            merged = copy.deepcopy(x[0])
            for i in range(1, len(x)):
                merged.update(x[i])
            return merged

def set_seeds(seed=1): # to reduce the stochasticity from the cuda and other libraries
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    print(f'random seed set with seed value {seed}')
        
# Custom data laoders and samplers

class CustomImageDataset(Dataset):
    def __init__(self, imgs, lbl, transform=None, standardize=True):
        self.imgs = imgs                                             # glob(root_dir + '/*.tif')
        self.lbls = lbl                                              # [im.replace('images', 'labels') for im in self.imgs]
        self.transform = transform
        self.standardize = standardize
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lb_path = self.lbls[idx]
        image = read_images(img_path)
        if self.transform is not None:
            image = transform(image)
        if self.standardize:
            image = image
        image = torch.from_numpy(np.array([image[:,:,0], image[:,:,1], image[:,:,2]]))
        label = read_labels(lb_path)
        label = label[None,:]  # expand channel dimensions
        return image, label

class TensorDataset(Dataset):
    def __init__(self, imgs, lbl, transform=None, standardize=True):
        self.imgs = imgs                                             # glob(root_dir + '/*.tif')
        self.lbls = lbl                                              # [im.replace('images', 'labels') for im in self.imgs]
        self.transform = transform
        self.standardize = standardize
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lb_path = self.lbls[idx]
        image = torch.load(img_path)
        if self.transform is not None:
            image = transform(image)
        if self.standardize:
            image = image
        label = torch.load(lb_path)
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
        self.images = sorted(glob(root + '/images'+ '/*.pt'))
        self.labels = [im.replace('images', 'labels') for im in self.images]
        
        assert len(self.images) == len(self.labels), 'Warning, imgaes and labels have different length'
        
        # adaptation sampling
        train, valid, test = adaptation_split(self.images,
                                              self.labels, 
                                              valid_ratio=self.v_size, 
                                              train_ratio = self.tr_size, 
                                              sampling=self.sampling)
        # put sampled to the dataset
        self.train_samples = TensorDataset(train[0], train[1])  # TensorDataset CustomImageDataset
        self.valid_images = TensorDataset(valid[0], valid[1])
        self.test_images = TensorDataset(test[0], test[1])
        
        print(f'==Sample distribution==')
        print(f'adaptation images: {len(train[0])}, Adaptation labels: {len(train[1])}')
        print(f'validation images: {len(valid[0])}, validation labels: {len(valid[1])}')
        print(f'Test images: {len(test[0])}, Test labels: {len(test[1])}')
        
#         self.train_dataset = TensorDataset(img, lbl) # after sampling
#         self.valid_dataset = TensorDataset(img, lbl) # after sampling
#         self.test_dataset = TensorDataset(img, lbl) # after sampling
        
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
        self.meta_loaders = []   
        self.train_imgs = []
        self.train_lbls = []
        self.valid_imgs = []
        self.valid_lbls = []
        
        for image_folder in self.image_folders:
            images = sorted(glob(image_folder + '/*.pt'))
            labels = [im.replace('images', 'labels') for im in images]
            train, valid = train_test_split(images=images,
                                            labels=labels, 
                                            valid_ratio=self.valid_ratio, 
                                            sampling=self.sampling)
            
            task_dataset  = TensorDataset(train[0], train[1])  # # TensorDataset CustomImageDataset
            
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
        
        self.valid_task = TensorDataset(self.valid_imgs, self.valid_lbls)
        self.valid_loader = DataLoader(self.valid_task, self.batch_size, drop_last=True,num_workers=0, shuffle=True)
        
        self.classic_task = TensorDataset(self.train_imgs, self.train_lbls)
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

    
######## Model definition #####################
class DeepUnet(nn.Module):
    def __init__(self, arch='Unet', encoder_name='resnet101', in_channels=3, out_classes=1, encoder_weights='imagenet'):
        super(DeepUnet, self).__init__()
        self.model = smp.create_model(arch=arch,
                                      encoder_name=encoder_name,
                                      encoder_weights=encoder_weights,
                                      in_channels=in_channels,
                                      classes=out_classes)
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask


#### Pixel based accuracy class #####
class AccuracyMetric(nn.Module):
    def __init__(self, activation='sigmoid', treshold = 0.5):
        super(AccuracyMetric, self).__init__()
        '''Computes the pixel based accuracy metrics'''
        self.accuracy = smp.utils.metrics.Accuracy(threshold=treshold, activation=activation)
        self.iou = smp.utils.metrics.IoU(threshold=treshold, activation=activation)
        self.f1 = smp.utils.metrics.Fscore(threshold=treshold, activation=activation)
        self.recal = smp.utils.metrics.Recall(threshold=treshold, activation=activation)
        self.prec = smp.utils.metrics.Precision(threshold=treshold, activation=activation)
    def forward(self, pr, gt, metric='all'):
        if metric == 'accuracy':
            accuracy = self.accuracy(pr, gt)
            return {'accuracy': accuracy}
        elif metric == 'iou':
            iou = self.iou(pr, gt)
            return {'iou': iou}
        elif metric == 'f1':
            f1 = self.f1(pr, gt)
            return {'f1':f1}
        elif metric == 'recal':
            recal = self.recal(pr, gt)
            return {'recal':recal}
        elif metric == 'prec':
            prec = self.prec(pr, gt)
            return {'rec':prec}
        elif metric == 'all':
            accuracy = self.accuracy(pr, gt)
            iou = self.iou(pr, gt)
            f1 = self.f1(pr, gt)
            recal = self.recal(pr, gt)
            prec = self.prec(pr, gt)
            return {'acc': accuracy, 'iou': iou, 'f1s':f1, 'rec':recal, 'pre':prec}
        else:
            raise ValueError('the specified accuracy type is not known')

### Object based accuracy metrics implementation class  ###
class ObjectAccuracy:
    def __init__(self, batched=True, iou_cut_off=0.5, logits=True, eps=0.000001):
        self.batched = batched
        self.logits = logits
        self.eps = eps
        self.names = ['rf_count','prd_count','TP', 'FP', 'FN', "F1", "CR", "CM", "QL"]
        self.totals = ['rf_count','prd_count','TP', 'FP', 'FN']
        self.means = ["F1", "CR", "CM", "QL"]
        self.mask_cut_off = 0.5
        self.iou_cut_off = iou_cut_off
        
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
            
            summary = {}
            for i in self.names:
                val = [a[i] for a in metrics]
                if i in self.totals:
                    val = np.nansum(val)
                else:
                    val = np.nanmean(val)
                summary[i] = val
            return summary
    
    def object_metric(self, P, R, iou_thresh=0.5, eps= 0.00001, compute_pr=True):
        '''
        P: predicted mask
        R: reference mask
        compute_pr: boolean, whether to compute precision and recall per single chip
        returns TP, FP and FN counts
        '''
        assert P.shape == R.shape, 'Predicted and reference shapes are not the same'

        R = self.pad(R)                    # pad to control error at the edge pixels
        R = measure.find_contours(R, 0.5)  # generate contours
        R = [make_valid(geometry.Polygon(pol)) for pol in R]  # convert to shapely polygon

        P = self.pad(P)
        P = measure.find_contours(P, 0.5)
        P = [make_valid(geometry.Polygon(pol)) for pol in P]

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
            corr = TP/(TP+FP + eps)  # prec
            comp = TP/(TP+FN + eps)  # reca
            qual = TP/(TP+FP+FN+eps) 
            of1 = 2*((corr*comp)/(corr+comp+eps))
            
            return {'rf_count':ref_count,'prd_count':pred_count,'TP':TP, 'FP':FP, 'FN':FN, "F1":of1, "CR":corr, "CM":comp, 'QL':qual}
        
    def pad(self, mask):
        return np.pad(mask.cpu(), ((1, 1), (1, 1)), 'minimum')
    
    def repad(self, mask):
        return mask[1:-1, 1:-1]


####### Dynamic weight module for inverse loss weighted gradient update ##################
class DynamicWeightModule(nn.Module):
    def __init__(self):
        super(DynamicWeightModule, self).__init__()
        self.loss = []
        self.truth = []
        self.weight = []
        
    def meta_weighted_loss(self, reset=True):
        self.compute_task_weight()
        count = len(self.loss)
        weighed_loss = sum([self.weight[i]*self.loss[i] for i in range(count)])/count  # inverse weighted eman loss
        if reset:
            self.reset_state()
        return weighed_loss
    
    def update_state(self, x):
        self.truth.append(1-x)
        self.loss.append(x)

    def compute_task_weight(self):
        sums = sum(self.truth)
        for i in range(len(self.truth)):
            w = self.truth[i]/sums
            self.weight.append(w)

    def reset_state(self):
        self.loss = []
        self.truth = []
        self.weight = []

####################################################################
class DynamicWeightModuleL(nn.Module):
    def __init__(self):
        super(DynamicWeightModuleL, self).__init__()
        self.loss = []
        self.truth = []
        self.weight = []
        
    def meta_weighted_loss(self, reset=True):
        self.compute_task_weight()
        count = len(self.loss)
        weighed_loss = sum([self.weight[i]*self.loss[i] for i in range(count)])/count  # inverse weighted eman loss
        if reset:
            self.reset_state()
        return weighed_loss
    
    def update_state(self, x):
        # self.truth.append(1-x)
        self.loss.append(x)

    def compute_task_weight(self):
        sums = sum(self.loss)
        for i in range(len(self.loss)):
            w = self.loss[i]/sums
            self.weight.append(w)

    def reset_state(self):
        self.loss = []
        # self.truth = []
        self.weight = []

####################################################################

#### MIXED sample trainer ######

class MixedTrainer:
    def __init__(self, root, weight_path, batch_size=10, lr=0.001, epochs=100, clear_mem=True):
        if clear_mem:
            torch.cuda.empty_cache()
        self.root = root
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.weight_path = weight_path
        self.is_cuda = torch.cuda.is_available()
        self.device =  torch.device("cuda" if self.is_cuda else "cpu")
        self.DATA = custom_data_loader(self.root, self.batch_size)
        self.acc_fn = AccuracyMetric().to(self.device) # binary accuracy metrics
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True).to(self.device)
        self.merge = DictMerger()
        self.clear_mem = clear_mem
        
    def train_mixer(self,init_weight='imagenet'):
        if not self.clear_mem:
            torch.cuda.empty_cache()
        assert init_weight in [None, 'imagenet'], 'initialization should be either "None" or "imagenet"'
        set_seeds()

        if init_weight == None:
            end_fold = 'weight_random'
        elif init_weight == 'imagenet':
            end_fold = 'weight_imagenet'
        else:
            raise ValueError('path not known')
        save_dir = self.weight_path + '/train/mixed/'+ end_fold

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=False)

        self.TL = {'loss': []}                                         # main mixed  training loss
        self.TA = {'acc': [], 'iou': [], 'f1s':[], 'rec':[], 'pre':[]}
        self.VL = {'loss': []}                                         # main mixed validation loss
        self.VA = {'acc': [], 'iou': [], 'f1s':[], 'rec':[], 'pre':[]}

        model = DeepUnet(encoder_weights=init_weight)
        model.to(self.device)
        self.best_weight = copy.deepcopy(model.state_dict())

        if isinstance(self.lr, list) and len(self.lr) == 2:
            max_lr = self.lr[0]
            min_lr = self.lr[1]
        else:
            max_lr = self.lr
            min_lr = 0.000001

        opt = torch.optim.SGD(model.parameters(), max_lr)

        train_loader = self.DATA.load_train_loader(partition='classic')
        valid_loader = self.DATA.load_valid_loader()

        best_valid = 0
        for epoch in range(1, self.epochs+1):
            epoch_loss = 0
            epoch_acc = {'acc':0, 'iou': 0, 'f1s':0, 'rec':0, 'pre':0}
            control = 0
            for j, data in enumerate(train_loader):   # this iterates through all data loaders(datasets) per epoch
                opt.zero_grad()
                X, Y = data[0].float().to(self.device), data[1].float().to(self.device)
                logits = model(X)
                loss = self.loss_fn(logits, Y)
                acc = self.acc_fn(logits, Y)
                loss.backward()
                opt.step()

                with torch.no_grad():
                    epoch_loss+=loss.item()
                    for key in list(self.TA.keys()):
                        epoch_acc[key] += acc[key].item()
                print(f'step: {j}, step_loss: {loss.item()}, step_acc: {acc["acc"].item()}')
                control+=1

            with torch.no_grad():
                epoch_loss = epoch_loss / control
                self.TL['loss'].append(epoch_loss)
                for key in list(self.TA.keys()):
                    self.TA[key].append(epoch_acc[key]/control)
                print(f'+++ {epoch}: step: {j}, train_loss: {epoch_loss}, train_acc:{epoch_acc["acc"]/control} +++')

            # validate the model
            val_loss = 0
            val_acc = {'acc': 0, 'iou': 0, 'f1s':0, 'rec':0, 'pre':0}
            control = 0

            with torch.no_grad():
                for image, target in valid_loader:
                    image, target = image.float().to(self.device), target.float().to(self.device)
                    output = model(image)
                    vloss = self.loss_fn(output, target)
                    vacc = self.acc_fn(output, target)
                    val_loss +=vloss.item()
                    for key in list(val_acc.keys()):
                        val_acc[key]+=vacc[key].item()
                    control+=1

            val_loss = val_loss/control

            self.VL['loss'].append(val_loss)
            for key in list(self.VA.keys()):
                self.VA[key].append(val_acc[key]/control)
            print(f'+++ {epoch}: step: {j}, valid_loss: {val_loss}, valid_acc: {val_acc["acc"]/control}')

            if epoch == 1:
                best_valid = self.VL["loss"][-1]
                self.best_weight = copy.deepcopy(model.state_dict())
            else:
                if self.VL["loss"][-1]<=best_valid:
                    best_valid = self.VL["loss"][-1]
                    self.best_weight = copy.deepcopy(model.state_dict())
                else:
                    pass

        touts = self.merge.merge(self.TL, self.TA)
        vouts = self.merge.merge(self.VL, self.VA)

        name = save_dir + '/checkpoint.pth'   # self.weigth_path
        torch.save(self.best_weight, name)

        np.save(save_dir + '/train_summary.npy', touts)
        np.save(save_dir + '/valid_summary.npy', vouts)


########### MAML trainer class ################
class MamlTrainer:
    def __init__(self, root, weight_path, batch_size=10, adapt_step=1, adapt_lr=0.001, meta_lr=0.0001, epochs=100, clear_mem=True):
        if clear_mem:
            torch.cuda.empty_cache()
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
        self.acc_fn = AccuracyMetric().to(self.device)                                 # binary accuracy metrics
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True).to(self.device)      
        self.n_tasks = len(os.listdir(root))
        self.merge = DictMerger()
        self.clear_mem = clear_mem
   
    def trainMAML(self, init_weight='imagenet'):
        assert init_weight in [None, 'imagenet'], 'initialization should be either "None" or "imagenet"'
        if not self.clear_mem:
            torch.cuda.empty_cache()
        set_seeds()
        if init_weight == None:
            end_fold = 'weight_random'
        elif init_weight == 'imagenet':
            end_fold = 'weight_imagenet'
        else:
            raise ValueError('path not known')
        save_dir = self.weight_path + '/' + end_fold
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=False)
        
        self.TL = {'loss': []}                                          # meta train loss
        self.TA = {'acc': [], 'iou': [], 'f1s':[], 'rec':[], 'pre':[]}  
        self.VL = {'loss': []}                                          # meta validation loss
        self.VA = {'acc': [], 'iou': [], 'f1s':[], 'rec':[], 'pre':[]} 
        
        model = DeepUnet(encoder_weights=init_weight)
        maml = l2l.algorithms.MAML(model, lr=self.adapt_lr, first_order=True, allow_nograd=True)  # shortens the iner gradient update
        maml.to(self.device)
        # self.best_weight = copy.deepcopy(maml.state_dict())
      
        if isinstance(self.meta_lr, list) and len(self.meta_lr) == 2:
            max_lr = self.meta_lr[0]
            min_lr = self.meta_lr[1]
        else:
            max_lr = self.meta_lr
            min_lr = 0.000001
            
        opt = torch.optim.SGD(maml.parameters(), max_lr)
        
        train_loader = self.DATA.load_train_loader()
        valid_loader = self.DATA.load_valid_loader()
        
        best_valid = 0
        for epoch in range(1, self.epochs+1):
            for j, data in enumerate(zip(*train_loader)):      # this iterates through all data loaders(datasets) per epoch
                opt.zero_grad()
                meta_train_loss = 0
                meta_train_acc = {'acc': 0, 'iou': 0, 'f1s':0, 'rec':0, 'pre':0}  # [0]*6
                control = 0                                   # number of effective tasks
                for i in range(len(data)):                    #  iterate through each tasks
                    X, Y = data[i][0], data[i][1]             # images and labels respectively
                    x_support, y_support = X[:self.ql].float().to(self.device), Y[:self.ql].float().to(self.device)
                    x_query, y_query = X[self.ql:].float().to(self.device), Y[self.ql:].float().to(self.device)
                    learner = maml.clone()
                    for _ in range(self.adapt_step):               # number of adaptation steps
                        logits = learner(x_support)
                        loss = self.loss_fn(logits, y_support)
                    learner.adapt(loss)                            # inner loop update
                    query_logits = learner(x_query)
                    query_loss = self.loss_fn(query_logits, y_query)
                    querry_acc = self.acc_fn(query_logits, y_query)
                    query_loss.backward()
                    
                    meta_train_loss += query_loss.item()
                    for key in list(meta_train_acc.keys()):
                        meta_train_acc[key]+=querry_acc[key].item()
                    control += 1
                    print(f'inner_loss: {query_loss}, inner_acc: {querry_acc["acc"]}')
                
                # meta_train_loss.backward()                      # retain_graph=True
                for p in maml.parameters():
                    p.grad.data.mul_(1.0 / control)
                opt.step()
                meta_train_loss = meta_train_loss / control
                with torch.no_grad():
                    self.TL['loss'].append(meta_train_loss) # needs change
                    for key in list(meta_train_acc.keys()):
                        self.TA[key].append(meta_train_acc[key]/control)
                    print(f'+++ {epoch}: step: {j}, meta-loss: {meta_train_loss}, mat-accu:{meta_train_acc["acc"]/control} +++') 
            
            if epoch % 5 == 0:
                name = save_dir + '/checkpoint.pth'      # self.weigth_path 
                torch.save(maml.state_dict(), name)
        touts = self.merge.merge(self.TL, self.TA) 
        np.save(save_dir + '/train_summary.npy', touts)
        
        del model
        del maml
        del x_support
        del y_support
        del learner
        del logits
        del query_logits
        del query_loss
        del querry_acc
        torch.cuda.empty_cache()
        gc.collect()
            
            # validate the model
#             val_loss = 0
#             val_acc = {'acc': 0, 'iou': 0, 'f1s':0, 'rec':0, 'pre':0}   
#             control = 0
            
#             with torch.no_grad():
#                 for image, target in valid_loader:
#                     image, target = image.float().to(self.device), target.float().to(self.device)
#                     output = maml(image)
#                     vloss = self.loss_fn(output, target)
#                     vacc = self.acc_fn(output, target)
#                     val_loss+=vloss.item()

#                     for key in list(val_acc.keys()):
#                         val_acc[key]+=vacc[key]
#                     control+=1

#                 val_loss = val_loss/control       # average over all batchs

#                 self.VL['loss'].append(val_loss)
#                 for key in list(val_acc.keys()):
#                     self.VA[key].append(val_acc[key]/control)

#                 print(f'+++ {epoch}: step: {j}, valid-loss: {val_loss}, valid_acc {val_acc["acc"]/control} +++')

#             if epoch == 1:
#                 best_valid = self.VL['loss'][-1]
#                 self.best_weight = copy.deepcopy(maml.state_dict())
#             else:
#                 if self.VL['loss'][-1]<=best_valid:
#                     best_valid = self.VL['loss'][-1]
#                     self.best_weight = copy.deepcopy(maml.state_dict())
#                 else:
#                     pass
        
#         touts = self.merge.merge(self.TL, self.TA)    # merge the wtow dictionaries
#         vouts = self.merge.merge(self.VL, self.VA)    # merge the two dictionaries
        
#         name = save_dir + '/checkpoint.pth'      # self.weigth_path 
#         torch.save(self.best_weight, name)
        
#         np.save(save_dir + '/train_summary.npy', touts)
#         np.save(save_dir + '/valid_summary.npy', vouts)



############################################################
################### Linear Weighted MAML ################### 
class MamlTrainerLW:
    def __init__(self, root, weight_path, batch_size=10, adapt_step=1, adapt_lr=0.001, meta_lr=0.0001, epochs=100, clear_mem=True):
        if clear_mem:
            torch.cuda.empty_cache()
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
        self.acc_fn = AccuracyMetric().to(self.device)                                 # binary accuracy metrics
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True).to(self.device)      
        self.n_tasks = len(os.listdir(root))
        self.merge = DictMerger()
        self.clear_mem = clear_mem
   
    def trainMAML(self, init_weight='imagenet'):
        assert init_weight in [None, 'imagenet'], 'initialization should be either "None" or "imagenet"'
        if not self.clear_mem:
            torch.cuda.empty_cache()
        set_seeds()
        if init_weight == None:
            end_fold = 'weight_random'
        elif init_weight == 'imagenet':
            end_fold = 'weight_imagenet'
        else:
            raise ValueError('path not known')
        save_dir = self.weight_path + '/' + end_fold
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=False)
        
        self.TL = {'loss': []}                                          # meta train loss
        self.TA = {'acc': [], 'iou': [], 'f1s':[], 'rec':[], 'pre':[]}  
        self.VL = {'loss': []}                                          # meta validation loss
        self.VA = {'acc': [], 'iou': [], 'f1s':[], 'rec':[], 'pre':[]} 
        
        updater = DynamicWeightModuleL().to(self.device)
        model = DeepUnet(encoder_weights=init_weight)
        maml = l2l.algorithms.MAML(model, lr=self.adapt_lr, first_order=True, allow_nograd=True)  # shortens the iner gradient update 
        maml.to(self.device)
        # self.best_weight = copy.deepcopy(maml.state_dict())
      
        if isinstance(self.meta_lr, list) and len(self.meta_lr) == 2:
            max_lr = self.meta_lr[0]
            min_lr = self.meta_lr[1]
        else:
            max_lr = self.meta_lr
            min_lr = 0.000001
            
        opt = torch.optim.SGD(maml.parameters(), max_lr)
        
        train_loader = self.DATA.load_train_loader()
        valid_loader = self.DATA.load_valid_loader()
        
        best_valid = 0
        for epoch in range(1, self.epochs+1):
            for j, data in enumerate(zip(*train_loader)):      # this iterates through all data loaders(datasets) per epoch
                #meta_train_loss = 0
                meta_train_acc = {'acc': 0, 'iou': 0, 'f1s':0, 'rec':0, 'pre':0}  # [0]*6
                control = 0                                   # number of effective tasks
                opt.zero_grad()
                for i in range(len(data)):                    #  iterate through each tasks
                    X, Y = data[i][0], data[i][1]             # images and labels respectively
                    x_support, y_support = X[:self.ql].float().to(self.device), Y[:self.ql].float().to(self.device)
                    x_query, y_query = X[self.ql:].float().to(self.device), Y[self.ql:].float().to(self.device)
                    learner = maml.clone()
                    for _ in range(self.adapt_step):               # number of adaptation steps
                        logits = learner(x_support)
                        loss = self.loss_fn(logits, y_support)
                    learner.adapt(loss)                            # inner loop update
                    query_logits = learner(x_query)
                    query_loss = self.loss_fn(query_logits, y_query)
                    querry_acc = self.acc_fn(query_logits, y_query)
                    
                    # meta_train_loss += query_loss
                    updater.update_state(query_loss) # add loss to dynamic module
                    for key in list(meta_train_acc.keys()):
                        meta_train_acc[key]+=querry_acc[key].detach().cpu().item()
                    control += 1
                    print(f'inner_loss: {query_loss}, inner_acc: {querry_acc["acc"]}')

                # meta_train_loss = meta_train_loss / control
                meta_train_loss = updater.meta_weighted_loss()
                
                meta_train_loss.backward()                      # retain_graph=True
                updater.reset_state() # clear the update module 
                opt.step()
                with torch.no_grad():
                    self.TL['loss'].append(meta_train_loss.cpu().item())

                    for key in list(meta_train_acc.keys()):
                        self.TA[key].append(meta_train_acc[key]/control)
                    print(f'+++ {epoch}: step: {j}, meta-loss: {meta_train_loss}, mat-accu:{meta_train_acc["acc"]/control} +++') 
            if epoch % 5 == 0:
                name = save_dir + '/checkpoint.pth'      # self.weigth_path
                torch.save(maml.state_dict(), name)
        touts = self.merge.merge(self.TL, self.TA)
        np.save(save_dir + '/train_summary.npy', touts)
            
        del model
        del maml
        del x_support
        del y_support
        del learner
        del logits
        del query_logits
        del query_loss
        del querry_acc
        torch.cuda.empty_cache()
        gc.collect()
        
            # validate the model
#             val_loss = 0
#             val_acc = {'acc': 0, 'iou': 0, 'f1s':0, 'rec':0, 'pre':0}   
#             control = 0
#             with torch.no_grad():
#                 for image, target in valid_loader:
#                     image, target = image.float().to(self.device), target.float().to(self.device)
#                     output = maml(image)
#                     vloss = self.loss_fn(output, target)
#                     vacc = self.acc_fn(output, target)
#                     val_loss+=vloss.cpu().item()

#                     for key in list(val_acc.keys()):
#                         val_acc[key]+=vacc[key].cpu().item()
#                     control+=1

#                 val_loss = val_loss/control       # average over all batchs

#                 self.VL['loss'].append(val_loss)
#                 for key in list(val_acc.keys()):
#                     self.VA[key].append(val_acc[key]/control)

#                 print(f'+++ {epoch}: step: {j}, valid-loss: {val_loss}, valid_acc {val_acc["acc"]/control} +++')
            
#             if epoch == 1:
#                 best_valid = self.VL['loss'][-1]
#                 self.best_weight = copy.deepcopy(maml.state_dict())
#             else:
#                 if self.VL['loss'][-1]<=best_valid:
#                     best_valid = self.VL['loss'][-1]
#                     self.best_weight = copy.deepcopy(maml.state_dict())
#                 else:
#                     pass
        
#         touts = self.merge.merge(self.TL, self.TA)    # merge the wtow dictionaries
#         vouts = self.merge.merge(self.VL, self.VA)    # merge the two dictionaries
        
#         name = save_dir + '/checkpoint.pth'      # self.weigth_path 
#         torch.save(self.best_weight, name)
        
#         np.save(save_dir + '/train_summary.npy', touts)
#         np.save(save_dir + '/valid_summary.npy', vouts)
############################################################

##### The class that used to adapt trained model        
class Adapter:
    def __init__(self, root, tr_size=0.1, v_size=0.1, lr=0.1, batch_size=8, epochs=20, weight_path=None, test=True):
        self.root = root
        self.tr_size = tr_size
        self.v_size = v_size
        self.lr = lr
        self.epochs = epochs
        self.weight_path = weight_path
        self.batch_size = batch_size
        self.is_cuda = torch.cuda.is_available()
        self.device =  torch.device("cuda" if self.is_cuda else "cpu")
        self.acc_fn = AccuracyMetric().to(self.device) # binary accuracy metrics
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True).to(self.device)
        self.Data = Adaptation_data_loader(root=root,
                                           sampling='systematic',
                                           batch_size=self.batch_size,
                                           v_size=self.v_size,
                                           tr_size=self.tr_size)
        self.TL = {'loss': []}                      
        self.TA = {'acc': [], 'iou': [], 'f1s':[], 'rec':[], 'pre':[]}  
        self.VL = {'loss': []}                        
        self.VA = {'acc': [], 'iou': [], 'f1s':[], 'rec':[], 'pre':[]}
        self.merge = DictMerger()
        self.test = test

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

        assert model_type in ['maml', 'classic', 'mixed']
        
        if model_type == 'maml':
            sub_fold = model_type
            model = DeepUnet(encoder_weights=init_weight)
            model = l2l.algorithms.MAML(model, lr=self.lr, first_order=True, allow_nograd=True)
            
        elif model_type == 'classic' or model_type == 'mixed':
            sub_fold = model_type
            model = DeepUnet(encoder_weights=init_weight)
        else:
            raise ValueError('model type {model_type} is known')
        
        save_dir = self.weight_path + '/' + sub_fold + '/' + end_fold  
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
        
        model.train()
        model.to(self.device)
        self.best_weight = copy.deepcopy(model.state_dict())
        
        adapt_loader = self.Data.train_loader()   # adapt traner
        valid_loader = self.Data.valid_loader()  # adapt validtor
        test_loader = self.Data.test_loader()  # adaptation data loader
        best_valid = 0
        for epoch in range(1, self.epochs+1):
            epoch_loss = 0
            epoch_acc = {'acc':0,'iou':0,'f1s':0,'rec':0,'pre':0}
            control = 0
            
            for j, data in enumerate(adapt_loader): # this iterates through all data loaders(datasets) per epoch
                X, Y = data[0].float().to(self.device), data[1].float().to(self.device)
                optim.zero_grad()
                logits = model(X)
                loss = self.loss_fn(logits,Y)
                acc = self.acc_fn(logits,Y)
                loss.backward()
                optim.step()
                epoch_loss+=loss.item()
                
                with torch.no_grad():
                    for key in list(epoch_acc.keys()):
                        epoch_acc[key]+=acc[key].item()
                control+=1
                print(f'Step-loss: {loss.item()}, step_acc: {acc["acc"].item()}')
            
            with torch.no_grad():
                epoch_loss = epoch_loss / control
                self.TL['loss'].append(epoch_loss)
                for key in list(self.TA.keys()):
                    self.TA[key].append(epoch_acc[key]/control)
                print(f'+++ {epoch}: step: {j}, train_loss: {epoch_loss/control}, train_acc: {epoch_acc["acc"]/control} +++')
                
            val_loss = 0
            val_acc = {'acc':0, 'iou':0, 'f1s':0, 'rec':0, 'pre':0}
            control = 0

            with torch.no_grad():
                for image, target in valid_loader:
                    image, target = image.float().to(self.device), target.float().to(self.device)
                    output = model(image)
                    vloss = self.loss_fn(output, target)
                    vacc = self.acc_fn(output, target)
                    val_loss+=vloss.item()
                    for key in list(val_acc.keys()):
                        val_acc[key]+=vacc[key].item()
                    control+=1

            val_loss = val_loss/control
            self.VL['loss'].append(val_loss)
            for key in list(self.VA.keys()):
                self.VA[key].append(val_acc[key])
            print(f'+++ {epoch}: step: {j}, val_loss: {val_loss}, va_acc {val_acc["acc"]/control}')

            if epoch == 1:
                best_valid = self.VL['loss'][-1]
                self.best_weight = copy.deepcopy(model.state_dict())
            else:
                if self.VL['loss'][-1]<=best_valid:
                    best_valid = self.VL['loss'][-1]
                    self.best_weight = copy.deepcopy(model.state_dict())
                else:
                    pass
                
        touts = self.merge.merge(self.TL, self.TA)
        vouts = self.merge.merge(self.VL, self.VA)
        
        name =  save_dir + '/checkpoint.pth'  # self.weigth_path
        torch.save(self.best_weight, name)
        
        np.save(save_dir + '/train_summary.npy', touts)
        np.save(save_dir + '/valid_summary.npy', vouts)
        
        if self.test: # test after adaptation
            print('Computing test metrics')
            model.load_state_dict(self.best_weight)
            model.eval()

            self.obj_metric = ObjectAccuracy()       # check 
            pix_acc = {'acc':0, 'iou': 0, 'f1s':0, 'rec':0, 'pre':0}
            obj_acc = {'rf_count':0, 'prd_count':0,'TP':0, 'FP':0, 'FN':0, "F1":0, "CR":0, "CM":0, 'QL':0}
            control = 0

            for (x, y) in test_loader:
                logs = model(x.float().to(self.device))
                pacc = self.acc_fn(logs, y.float().to(self.device))
                oacc = self.obj_metric.compute(logs, y.float().to(self.device))
                
                for key in list(pix_acc.keys()):
                    pix_acc[key]+=pacc[key]
                for key in list(obj_acc.keys()):
                    obj_acc[key]+=oacc[key]
                control += 1
            
            for key in list(pix_acc.keys()):
                pix_acc[key] = pix_acc[key]/control
            for key in list(obj_acc.keys()):
                if key in ['F1','CM','CR','QL']:
                    obj_acc[key] = obj_acc[key]/control
                    
            test_report = self.merge.merge(pix_acc, obj_acc)
            np.save(save_dir + '/test_summary.npy', test_report)
            

################ Entropy based Pseudolabel filtering for Meta -self-supervised adaptation 
class Filter(nn.Module):
    '''https://github.com/Haochen-Wang409/U2PL/blob/main/train_semi.py'''
    def __init__(self, tot_epoch=120, alpha_not=0.2, return_metric=False):
        self.tot_epoch = tot_epoch
        self.return_metric=return_metric
        self.alpha_not = alpha_not

    def pick(self, x, epoch=None):
        alpha = self.compute_alpha(epoch)
        a = 1-alpha
        x_sigmoid = x.sigmoid()
        entropy = -x_sigmoid*torch.log(x_sigmoid+1e-10) # entropy value for binary classification
        gamma = torch.quantile(entropy.view(entropy.shape[0],-1), a, dim=1)
        val = torch.ge(entropy, gamma[:,None, None, None])  # entropy value
        clas = x_sigmoid>=0.5
        result = val.float()*clas.float()

        if not self.return_metric:
            return result
        else:
            return result, gamma, alpha

    def compute_alpha(self, current_epoch=None):
        final_alpha = self.alpha_not*(1-current_epoch/self.tot_epoch)
        return final_alpha

def compute_ULW(target, percent, logit_teach):
    '''Target: Predicted hard class from teacher model, using eval
        percent: Percent of which the reliable classes are considered 
        logit_teach: the logit predictions from the teacher model in training mode'''
    batch_size, num_class, h, w = logit_teach.shape

    with torch.no_grad():
        # drop pixels with high entropy
        a = percent/100
        prob = logit_teach.sigmoid()
        entropy = -prob*torch.log(prob + 1e-10)

        thresh = torch.quantile(entropy.view(entropy.shape[0],-1), a, dim=1) #.percentile(entropy[target != 0.0].detach().cpu().numpy().flatten(), percent)
        mask = torch.ge(entropy, thresh[:None, None, None]).float() * target

        #u_loss_weight =  batch_size * h * w / torch.sum(thresh_mask)
        #print(f'Mask treshold: {thresh}, loss weight: {u_loss_weight}')

    return mask# u_loss_weight

def dynamic_weight(state_epoch, total_epoch, drop=30):
    percent_unreliable = (100 - drop) * (1 - state_epoch /total_epoch)
    percent_reliable = (100 - percent_unreliable)/100
    return percent_reliable


#############################  Mean Teacher Approach with Unsupervised Learning ##################################
class MeanTeacher:
    def __init__(self, root, tr_size=0.1, v_size=0.1, lr=0.1, batch_size=8, epochs=20, weight_path=None, test=True):
        self.root = root
        self.tr_size = tr_size
        self.v_size = v_size
        self.lr = lr
        self.epochs = epochs
        self.weight_path = weight_path
        self.batch_size = batch_size
        self.is_cuda = torch.cuda.is_available()
        self.device =  torch.device("cuda" if self.is_cuda else "cpu")
        self.acc_fn = AccuracyMetric().to(self.device) # binary accuracy metrics
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True).to(self.device)
        self.Data = Adaptation_data_loader(root=root,
                                           sampling='systematic',
                                           batch_size=self.batch_size,
                                           v_size=self.v_size,
                                           tr_size=self.tr_size)
        self.TL = {'lossT': []}
        self.TUL = {'lossU':[]}
        self.TSL = {'lossS':[]}
        self.TA = {'acc': [], 'iou': [], 'f1s':[], 'rec':[], 'pre':[]}
        self.VL = {'lossV': []}
        self.VA = {'acc': [], 'iou': [], 'f1s':[], 'rec':[], 'pre':[]}
        self.merge = DictMerger()
        self.test = test
        self.m = 0.9 # mean teacher balncing momentum
        # self.filter = Filter(tot_epoch=epochs)
        self.drop = 20

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

        assert model_type in ['maml', 'classic', 'mixed']

        if model_type == 'maml':
            sub_fold = model_type
            model = DeepUnet(encoder_weights=init_weight)
            model = l2l.algorithms.MAML(model, lr=self.lr, first_order=True, allow_nograd=True)

        elif model_type == 'classic' or model_type == 'mixed':
            sub_fold = model_type
            model = DeepUnet(encoder_weights=init_weight)
        else:
            raise ValueError('model type {model_type} is known')

        save_dir = self.weight_path + '/' + sub_fold + '/' + end_fold
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        #if checkpoint is not None:
           # model.load_state_dict(torch.load(checkpoint))
            #print(f'Checkpoint loaded from {checkpoint}')
        #else:
            #raise ValueError('Weight path can not be "None" for adaptation phase')
        teach_model = copy.deepcopy(model)
        teach_model.load_state_dict(torch.load(checkpoint))
        for param in teach_model.parameters():
            param.requires_grad = False
        teach_model.to(self.device)

        if isinstance(self.lr, list) and len(self.lr) == 2:
            max_lr = self.lr[0]
            min_lr = self.lr[1]
        else:
            max_lr = self.lr
            min_lr = 0.000001
        optim = torch.optim.SGD(model.parameters(), max_lr)
        model.train()
        model.to(self.device)
        self.best_weight = copy.deepcopy(model.state_dict())

        adapt_loader = self.Data.train_loader()   # adapt traner
        valid_loader = self.Data.valid_loader()  # adapt validtor
        test_loader = self.Data.test_loader()  # adaptation data loader
        best_valid = 0
        for epoch in range(1, self.epochs+1):
            #if epoch < 40:
             #   for g in optim.param_groups:
              #      g['lr'] = 0.1
            #elif epoch >=40 and epoch < 80:
             #   for g in optim.param_groups:
              #      g['lr'] = 0.01
            #else:
             #   for g in optim.param_groups:
                  #  g['lr'] = 0.001
            epoch_loss = 0 # supervised
            epoch_loss_unsup = 0
            epoch_loss_tot = 0
            epoch_acc = {'acc':0,'iou':0,'f1s':0,'rec':0,'pre':0}
            control = 0

            for j,(da, db) in enumerate(zip(cycle(adapt_loader), test_loader)): # data in enumerate(adapt_loader): # this iterates through all data loaders(datasets) per epoch
                X, Y = da[0].float().to(self.device), da[1].float().to(self.device)
                Xu, _ = db[0].float().to(self.device), db[1]
                optim.zero_grad()
                teach_model.eval()
                target_teach = (teach_model(Xu).sigmoid()>=0.5).float()
                teach_model.train()
                t_logit = teach_model(Xu)
                ##percent = compute_drop_percent(epoch, self.epochs, drop=20)
                ##mask = compute_ULW(target_teach, percent, t_logit) # weight term for unsupervised weight for reliable pixels self.filter.pick(x=t_logit,epoch=epoch) 
                logits = model(X)
                loss_sup = self.loss_fn(logits,Y)
                logit_us = model(Xu)
                #weight = dynamic_weight(epoch, self.epochs)
                loss_unsup = self.loss_fn(logit_us,target_teach)
                acc = self.acc_fn(logits,Y)
                tot_loss = loss_sup + loss_unsup
                tot_loss.backward()
                optim.step()
                epoch_loss+=loss_sup.item()
                epoch_loss_unsup+=loss_unsup.item()
                epoch_loss_tot +=tot_loss.item()

                with torch.no_grad():
                    for key in list(epoch_acc.keys()):
                        epoch_acc[key]+=acc[key].item()
                control+=1
                print(f'stp supl: {loss_sup.item()}, step_usupl: {loss_unsup.item()}, step_totl: {tot_loss.item()}, step_acc: {acc["acc"].item()}')

            with torch.no_grad():
                epoch_loss = epoch_loss / control
                epoch_loss_unsup = epoch_loss_unsup/control
                epoch_loss_tot = epoch_loss_tot/control
                self.TSL['lossS'].append(epoch_loss)
                self.TUL['lossU'].append(epoch_loss_unsup)
                self.TL['lossT'].append(epoch_loss_tot)
                for key in list(self.TA.keys()):
                    self.TA[key].append(epoch_acc[key]/control)
                print(f'+++ {epoch}: step: {j}, SL: {epoch_loss}, UL: {epoch_loss_unsup}, TL: {epoch_loss_tot}, train_acc: {epoch_acc["acc"]/control} +++')

            val_loss = 0
            val_acc = {'acc':0, 'iou':0, 'f1s':0, 'rec':0, 'pre':0}
            control = 0

            with torch.no_grad():
                for image, target in valid_loader:
                    image, target = image.float().to(self.device), target.float().to(self.device)
                    output = model(image)
                    vloss = self.loss_fn(output, target)
                    vacc = self.acc_fn(output, target)
                    val_loss+=vloss.item()
                    for key in list(val_acc.keys()):
                        val_acc[key]+=vacc[key].item()
                    control+=1

            val_loss = val_loss/control
            self.VL['lossV'].append(val_loss)
            for key in list(self.VA.keys()):
                self.VA[key].append(val_acc[key])
            print(f'+++ {epoch}: step: {j}, val_loss: {val_loss}, va_acc {val_acc["acc"]/control}')

            for paramst, paramtr in zip(model.parameters(), teach_model.parameters()):
                # pmst is parameter of student model
                paramtr.data = paramtr.data * self.m + paramst.data * (1. - self.m) # m is momentum pmsm is teacher model trained with sample mixing

            if epoch == 1:
                best_valid = self.VL['lossV'][-1]
                self.best_weight = copy.deepcopy(model.state_dict())
            else:
                if self.VL['lossV'][-1]<=best_valid:
                    best_valid = self.VL['lossV'][-1]
                    self.best_weight = copy.deepcopy(model.state_dict())
                else:
                    pass

        touts = self.merge.merge([self.TL,self.TUL, self.TSL, self.TA])
        vouts = self.merge.merge([self.VL, self.VA])

        name =  save_dir + '/Fcheckpoint.pth'  # self.weigth_path
        torch.save(self.best_weight, name)

        np.save(save_dir + '/Ftrain_summary.npy', touts)
        np.save(save_dir + '/Fvalid_summary.npy', vouts)

        if self.test: # test after adaptation
            print('Computing test metrics')
            model.load_state_dict(self.best_weight)
            model.eval()

            #self.obj_metric = ObjectAccuracy()       # check
            pix_acc = {'acc':0, 'iou': 0, 'f1s':0, 'rec':0, 'pre':0}
            obj_acc = {'rf_count':0, 'prd_count':0,'TP':0, 'FP':0, 'FN':0, "F1":0, "CR":0, "CM":0, 'QL':0}
            control = 0

            for (x, y) in test_loader:
                logs = model(x.float().to(self.device))
                pacc = self.acc_fn(logs, y.float().to(self.device))
                #oacc = self.obj_metric.compute(logs, y.float().to(self.device))

                for key in list(pix_acc.keys()):
                    pix_acc[key]+=pacc[key]
                #for key in list(obj_acc.keys()):
                 #   obj_acc[key]+=oacc[key]
                control += 1

            for key in list(pix_acc.keys()):
                pix_acc[key] = pix_acc[key].cpu().item()/control
            #for key in list(obj_acc.keys()):
             #   if key in ['F1','CM','CR','QL']:
              #      obj_acc[key] = obj_acc[key]/control
            print('++++Pixel accuracy metrics+++')
            print(pix_acc)

            #test_report = self.merge.merge([pix_acc, obj_acc])
            np.save(save_dir + '/Ftest_summary.npy', pix_acc) 


class BaseTrainer:
    def __init__(self, root, weight_path, batch_size=20, lr=0.1,epochs=100, v_size=0.1, tr_size=0.1, test=True):
        self.root = root
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.v_size = v_size
        self.tr_size = tr_size
        self.weight_path = weight_path
        self.is_cuda = torch.cuda.is_available()
        self.device =  torch.device("cuda" if self.is_cuda else "cpu")
        self.acc_fn = AccuracyMetric().to(self.device) # binary accuracy metrics
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True).to(self.device)      # sigmoid focal loss with logit
        self.Data = Adaptation_data_loader(root= root,
                                           sampling='systematic',
                                           batch_size=self.batch_size,
                                           v_size=self.v_size,
                                           tr_size = self.tr_size)
        self.merge = DictMerger()
        self.test = test
        
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
        save_dir = self.weight_path + '/' + end_fold
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=False)
    
        self.TL = {'loss': []}                      
        self.TA = {'acc': [], 'iou': [], 'f1s':[], 'rec':[], 'pre':[]}  
        self.VL = {'loss': []}                        
        self.VA = {'acc': [], 'iou': [], 'f1s':[], 'rec':[], 'pre':[]} 
        
        # model = Unet(nclass=1, weights=init_weight, activate=False)
        model = DeepUnet(encoder_weights=init_weight)
        model.to(self.device)
        self.best_weight = copy.deepcopy(model.state_dict())
        
        if isinstance(self.lr, list) and len(self.lr) == 2:
            max_lr = self.lr[0]
            min_lr = self.lr[1]
        else:
            max_lr = self.lr
            min_lr = 0.000001
    
        opt = torch.optim.SGD(model.parameters(), max_lr)
        # meta_scheduler = CosineAnnealingLR(opt, T_max=max_lr, eta_min=min_lr)  # needs reference 
        
        train_data =self.Data.test_loader()    # self.Data.train_loader()  # adapt traner
        valid_data = self.Data.valid_loader()  # adapt validtor
        test_data = self.Data.train_loader()    # train data 
         
        best_valid = 0
        for epoch in range(1, self.epochs+1):
            epoch_loss = 0
            epoch_acc = {'acc':0, 'iou':0, 'f1s':0, 'rec':0, 'pre':0}   # [0]*8
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
                    for key in list(epoch_acc.keys()):
                        epoch_acc[key] += acc[key]        
                print(f'step_loss: {loss.item()}, step_acc: {acc["acc"]}')
                control+=1
                
            with torch.no_grad():
                epoch_loss = epoch_loss / control
                self.TL['loss'].append(epoch_loss)
                for key in list(self.TA.keys()):
                    self.TA[key].append(epoch_acc[key]/control)
                print(f'+++ {epoch}: step: {j}, train_loss: {epoch_loss}, train_acc:{epoch_acc["acc"]/control} +++')

            val_loss = 0
            val_acc = {'acc':0, 'iou':0, 'f1s':0, 'rec':0, 'pre':0} 
            control = 0
            with torch.no_grad():
                for image, target in valid_data:
                    image, target = image.float().to(self.device), target.float().to(self.device)
                    output = model(image)
                    vloss = self.loss_fn(output, target)
                    vacc = self.acc_fn(output, target)
                    val_loss +=vloss.item()
                    for key in list(val_acc.keys()):
                        val_acc[key]+=vacc[key].item()
                    control+=1

            val_loss = val_loss/control
           
            self.VL['loss'].append(val_loss)
            for key in list(self.VA.keys()):
                self.VA[key].append(val_acc[key]/control)
            print(f'+++ {epoch}: step: {j}, valid_loss: {val_loss}, valid_acc: {val_acc["acc"]/control}')
            
            if epoch == 1:
                best_valid = self.VL['loss'][-1]
                self.best_weight = copy.deepcopy(model.state_dict())
            else:
                if self.VL['loss'][-1] <= best_valid:
                    best_valid = self.VL['loss'][-1]
                    self.best_weight = copy.deepcopy(model.state_dict())
                else:
                    pass
                
        touts = self.merge.merge(self.TL, self.TA)
        vouts = self.merge.merge(self.VL, self.VA)
        
        name = save_dir + '/checkpoint.pth'   # self.weigth_path
        torch.save(self.best_weight, name)
            
        np.save(save_dir + '/train_summary.npy', touts)
        np.save(save_dir + '/valid_summary.npy', vouts)
        # save model weight
        
        
        if self.test:
            print('Computing test metrics')
            model.load_state_dict(self.best_weight)
            model.eval()
            self.obj_metric = ObjectAccuracy()  # check 
            pix_acc = {'acc':0, 'iou':0, 'f1s':0, 'rec':0, 'pre':0}
            obj_acc = {'rf_count':0, 'prd_count':0,'TP':0, 'FP':0, 'FN':0, "F1":0, "CR":0, "CM":0, 'QL':0}
            control = 0
            for (x, y) in test_data:
                logs = model(x.float().to(self.device))
                pacc = self.acc_fn(logs, y.float().to(self.device))
                oacc = self.obj_metric.compute(logs, y.float().to(self.device))
                for key in list(pix_acc.keys()):
                    pix_acc[key]+=pacc[key]
                for key in list(obj_acc.keys()):
                    obj_acc[key]+=oacc[key]
                control +=1
            for key in list(pix_acc.keys()):
                pix_acc[key] = pix_acc[key]/control
            for key in list(obj_acc.keys()):
                if key in ["F1", "CR", "CM", "QL"]:
                    obj_acc[key] = obj_acc[key]/control
            test_report = self.merge.merge(pix_acc, obj_acc)
            np.save(save_dir + '/test_summary.npy', test_report)
        
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
        
        self.pix_ac_fn = AccuracyMetric().to(self.device)
        self.obj_acc_fn = ObjectAccuracy()
        self.merge = DictMerger()
        
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
            model = DeepUnet(encoder_weights=init_weight)
            model = l2l.algorithms.MAML(model, lr=self.lr, first_order=True, allow_nograd=True)
        elif  model_type == 'classic':
            sub_fold = 'classic'
            model = DeepUnet(encoder_weights=init_weight)
        else:
            raise ValueError('model type {model_type} is known')
        
        model.load_state_dict(torch.load(checkpoint)) # load weight adapted weight
        
        save_dir = self.out_path + '/' + sub_fold + '/' + end_fold
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        model.eval()
        model.to(self.device)
        test_loader = self.Data.test_loader()
        
        acc = {'acc': 0, 'iou': 0, 'f1s':0, 'rec':0, 'pre':0}  
        obj = {'rf_count':0, 'prd_count':0,'TP':0, 'FP':0, 'FN':0, "F1":0, "CR":0, "CM":0, "QL":0}
        control = 0
        
        for data in test_loader:
            X, Y = data[0].float().to(self.device), data[1].float().to(self.device)
            logits = model(X)
            pix_acc = self.pix_ac_fn(logits, Y) # iou function 
            obj_acc = self.obj_acc_fn.compute(logits, Y) # f1 function
            
            for key in list(pix_acc.keys()):
                acc[key]+=pix_acc[key].item()
            for key in list(obj_acc.keys()):
                obj[key]+=obj_acc[key].item()
            control+=1

        for key in list(acc.keys()):
            acc[key] = acc[key]/control
        for key in list(obj.keys()):
            if key in ["F1", "CR", "CM", "QL"]:
                obj[key] = obj[key]/control
            else:
                pass

        final_metric = self.merge.merge(acc, obj)
        if report:
            print(final_metric)
        np.save(save_dir + '/accuracy_summary.npy', final_metric)
