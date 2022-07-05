import os
import argparse 
from main_maml import Adapter

def parseArgs():
    parser = argparse.ArgumentParser(description='A model to adapt MAML trained model to a specific task')
    parser.add_argument('root', help='root folder that contained all task data folders', type=str)
    parser.add_argument('weight_path', help='path to save weight', type=str)
    parser.add_argument('--train_size', help='train size from adaptation target data', default=0.1, type=float, required=False)
    parser.add_argument('--valid_size', help='validation data size from adaptation data', default=0.1, type=float, required=False)
    parser.add_argument('--lr', help='learning rate for model adaptation', default=0.001, type=float, required=False)
    parser.add_argument('--batch_size', help='batch size for daaloading', default=20, type=int, required=False)
    parser.add_argument('--epochs', help='epochs to train adaptation model', default=40, type=int, required=False)
    parser.add_argument('--input_weights', help='input weights root folder', type=str, required=True)
    args= parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseArgs()
    for folder in os.listdir(args.root):
        data_path = args.root + '/' + folder
        out_path = args.weight_path + '/' + folder  # check here 
        for iter_fold in os.listdir(args.input_weights):      # number of iteration
            data_dir = args.input_weights + '/' + iter_fold   # data within iteration
            for d_fold in os.listdir(data_dir):
                final_out_root = data_dir + '/' + d_fold
                adapter = Adapter(root=data_path,
                              tr_size=args.train_size,
                              v_size=args.valid_size,
                              lr=args.lr,
                              batch_size=args.batch_size,
                              epochs=args.epochs,
                              weight_path=final_out_root) # 'D:/meta_l2l/out2'
                
                weight_init = ['weight_random', 'imagenet']
                for init in weight_init:
                    checkpoint = args.input_weights + '/' + iter_fold + '/' + d_fold + '/' + init + '/' + 'checkpoint.pth'
                    if init == 'weight_random':
                        init_weight = None
                    else:
                        init_weight = init_weight
                    adapter.adapt(init_weight=init_weight, model_type='classic', checkpoint=checkpoint)