from skimage.metrics import structural_similarity
from skimage.io import imread
import numpy as np
import random
import pandas as pd
import os
import argparse
from glob import glob
from tqdm import tqdm

def makeNormal(x):
    return (x-x.min())/(x.max()-x.min())

def computSSIM_array(dst1, dst2, normalize=False, out_root=None, name=None, to_csv=False):
    alls = []
    for i in tqdm(range(len(dst1))):
        for j in range(len(dst2)):
            if  i == j:
                pass
            else:
                im1 = dst1[i]
                im2 = dst2[i]
                if normalize:
                    im1 = makeNormal(im1)
                    im2 = makeNormal(im2) 
                ssim = structural_similarity(im1=im1,
                                             im2=im2,
                                             win_size=11,
                                             channel_axis=2)
                alls.append(ssim)
    alls = np.array(alls).reshape(dst1.shape[0],dst2.shape[0])
    if to_csv:
        np.savetxt(f'{out_root}/{name}.csv', alls, delimiter=',')
    else:
        np.save(f'{out_root}/{name}.npy', alls)

def computSSIM_files(dst1, dst2, normalize=False, out_root=None, name=None, to_csv=False):
    alls = []
    for i in tqdm(range(len(dst1))):
        for j in range(len(dst2)):
            if  i == j:
                pass
            else:
                im1 = imread(dst1[i])
                im2 = imread(dst1[j])
                if normalize:
                    im1 = normalize(im1)
                    im2 = normalize(im2)
                ssim = structural_similarity(im1=im1,
                                             im2=im2,
                                             win_size=11,
                                             channel_axis=2)
                alls.append(ssim)
    alls = np.array(alls).reshape(len(dst1)-1,len(dst2))
    if to_csv:
        np.savetxt(f'{out_root}/{name}.csv', alls, delimiter=',')
    else:
        np.save(f'{out_root}/{name}.npy',alls)
        
def automateAll(root, sample=False, size=10, from_file=True, normalize=False, out_root=None, to_csv=False):
    if not os.path.exists(out_root):
        os.makedirs(out_root, exist_ok=True)
    folds1 = os.listdir(root)
    folds2 = os.listdir(root)
    for fold in folds1:
        for fold_ in folds2:
            if fold == fold_:
                pass
            else:
                print(fold, fold_)
                files1 = [f'{root}/{fold}/{file}' for file in os.listdir(f'{root}/{fold}')]
                files2 = [f'{root}/{fold_}/{file}' for file in os.listdir(f'{root}/{fold_}')]
                
                if sample:
                    files1 = random.sample(files1, size)
                    files2 = random.sample(files2, size)
                if from_file:
                    print(len(files1), len(files2))
                    computSSIM_files(dst1=files1,
                                     dst2=files2,
                                     normalize=normalize,
                                     out_root=out_root,
                                     name=f'{fold}vs{fold}',
                                     to_csv=to_csv)
                else:
                    print(len(files1), len(files2))
                    images_1 = np.stack([imread(file) for file in files1])
                    images_2 = np.stack([imread(file) for file in files2])

                    computSSIM_array(dst1=images_1,
                                     dst2=images_2,
                                     normalize=normalize,
                                     out_root=out_root,
                                     name=f'{fold}vs{fold}',
                                     to_csv=to_csv)    
def parseArgs():
    parser = argparse.ArgumentParser(description='A model to adapt MAML trained model to a specific task')
    parser.add_argument('--root', help='root folder that contained all image folders', dest='root', type=str, default='/ssim_test_data')
    parser.add_argument('--im_format', help='image format either .png, .tif', dest='im_format', type=str, default='.jpg')
    parser.add_argument('--sample', help='sample the images, if needed for quick computation', dest="sample", action="store_true") # action true
    parser.add_argument('--size', help='if sample is true, the number of samples to include', dest='size', default=10, type=int, required=False)
    parser.add_argument('--normalize', help='Whether to normalize the image', dest="normalize", action="store_true") # action store 
    parser.add_argument('--save', help='save the computed metric', dest="save", action="store_true") # store true
    parser.add_argument('--out_root', help='The root to save computed metric', dest='out_root', type=str, required=False, default='/ssim_samplestest')
    parser.add_argument('--to_csv', help='Whether to save as csv', dest="to_csv", action="store_true")
    parser.add_argument('--from_file', help='Whether to save as csv', dest="from_file", action="store_true")
    args= parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseArgs()
    automateAll(root=args.root,
                sample=True,
                size=20,
                normalize=args.normalize,
                out_root=args.out_root,
                to_csv=args.to_csv)
