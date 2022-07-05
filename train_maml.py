# model to train 
import argparse
from main_maml import MAML

def parseArgs():
    parser = argparse.ArgumentParser(description='model agnostic meta learning implementation both for classic and MAML')
    parser.add_argument('root', help='Root folder that contained all task data folders', type=str)
    parser.add_argument('weight_path', help='path/folder to save weight', type = str)
    parser.add_argument('--batch_size', help='batch size to load task datasets', default=20, type=int, required=False)
    parser.add_argument('--adapt_step', help='number of inner adaptation steps for a specific task', default=1, required=False)
    parser.add_argument('--adapt_lr', help='adaptation learning rate for inner model', default=0.05, type=float, required=False)
    parser.add_argument('--meta_lr', help='Learning rate to update meta model/outer model', default=0.001, type=float, required=False)
    parser.add_argument('--epochs', help='number of epochs to train', default=2, type=int, required=False)
    parser.add_argument('--freq', help='number of repeated experiments', default=2, type=int, required=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseArgs()
    if args.freq == 1:
        newModel = MAML(root=args.root,
                    weight_path=args.weight_path,
                    batch_size=args.batch_size,
                    adapt_step=args.adapt_step,
                    adapt_lr=args.adapt_lr,
                    meta_lr=args.meta_lr,
                    epochs=args.epochs)
        newModel.trainMML(init_weight=None)
        newModel.trainCLASSIC(init_weight=None)
        newModel.trainMML(init_weight='imagenet')
        newModel.trainCLASSIC(init_weight='imagenet')
    else:
        for i in range(args.freq):
            iter_fold = '0'+str(i)
            weight_fold = args.weight_path + iter_fold + '/'
            newModel = MAML(root=args.root,
                    weight_path=weight_fold,
                    batch_size=args.batch_size,
                    adapt_step=args.adapt_step,
                    adapt_lr=args.adapt_lr,
                    meta_lr=args.meta_lr,
                    epochs=args.epochs)
            newModel.trainMML(init_weight=None)
            newModel.trainCLASSIC(init_weight=None)
            newModel.trainMML(init_weight='imagenet')
            newModel.trainCLASSIC(init_weight='imagenet')

