import argparse
import os
from main_maml import BaseTrainer

def parseArgs():
    parser = argparse.ArgumentParser(description='model agnostic meta learning implementation both for classic and MAML')
    parser.add_argument('root', help='Root folder that contained image and label data folders', type=str)
    parser.add_argument('weight_path', help='path/folder to save weight', type = str)
    parser.add_argument('--batch_size', help='batch size to load task datasets', default=20, type=int, required=False)
    parser.add_argument('--lr', help='Learning rate to update model weight', default=0.001, type=float, required=False)
    parser.add_argument('--v_size', help='Validation data size', default=0.1, type=float, required=False)
    parser.add_argument('--tst_size', help='test data size', default=0.1, type=float, required=False)
    parser.add_argument('--freq', help='number of repeated experiments', default=2, type=int, required=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseArgs()
    for fold in os.listdir(args.root):
        data_folder = args.root + '/' + fold
        weight_fold = args.weight_path + '/' + fold
        if args.freq == 1:
            trainer = BaseTrainer(root=data_foldert,
                                  weight_path=weight_fold,
                                  batch_size=args.batch_size,
                                  lr=args.lr,
                                  epochs=args.epochs,
                                  v_size=args.v_size,
                                  tr_size=args.tst_size)
            trainer.trainbase(init_weight=None)
            trainer.trainbase(init_weight='imagenet')
        else:
            for i in range(args.freq): # repeated experiment for numerical stability
                iter_fold = '0' + str(i)
                for fold in os.listdir(args.root):
                    data_folder = args.root + '/' + fold
                    weight_fold = args.weight_path + '/' + iter_fold + '/' + fold
                    trainer = BaseTrainer(root=data_folder,
                                      weight_path=weight_fold,
                                      batch_size=args.batch_size,
                                      lr=args.lr,
                                      epochs=args.epochs,
                                      v_size=args.v_size,
                                      tr_size=args.tst_size)
                    trainer.trainbase(init_weight=None)
                    trainer.trainbase(init_weight='imagenet')
                