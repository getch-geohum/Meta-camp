import argparse
import os
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
    parser.add_argument('--samle_size', help='input weights root folder', type=int, required=False, default=10)
    parser.add_argument('--freq', help='frequency of iteratios to repeat the experiment', type=int, required=False, default=4)
    
    args= parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArgs()
    i = 0
    for folder in os.listdir(args.root):
        data_path = args.root + '/' + folder
        model_type = ['classic', 'maml']
        weight_init = ['weight_random', 'weight_imagenet']
        # out_path = args.weight_path + '/' + folder + '/' + str(args.samle_size) + '%'
        for j in range(args.freq):
            iter_fold = '0' + str(j)
            # final_out_root = out_path + '/' + iter_fold
            final_out_root = args.weight_path + '/' + iter_fold + '/adapt/' + folder + '/' + str(args.samle_size) + '%'
            print(final_out_root)
            os.makedirs(final_out_root, exist_ok=True)
        
            adapter = Adapter(root=data_path,
                              tr_size=args.train_size,
                              v_size=args.valid_size,
                              lr=args.lr,
                              batch_size=args.batch_size,
                              epochs=args.epochs,
                              weight_path=final_out_root) # 'D:/meta_l2l/out2'

            for mtype in model_type:
                for init in weight_init:
                    if mtype == 'classic':
                        checkpoint = args.input_weights + '/' + iter_fold + '/train/' + mtype + '/' + init + '/' + 'classic_checkpoint.pth'
                    elif mtype == 'maml':
                        checkpoint = args.input_weights + '/' + iter_fold + '/train/' + mtype + '/' + init + '/' + 'maml_checkpoint.pth'
                    print(i+1, ':', checkpoint)
                    if init == 'weight_random':
                        init_weight = None
                    else:
                        init_weight = 'imagenet'

                    adapter.adapt(init_weight=init_weight, model_type=mtype, checkpoint=checkpoint)