import argparse
import os
from main_maml import Tester
from glob import glob
import re 

def sortAlphnumeric(num): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(num, key = alphanum_key)

def parseArgs():
    parser = argparse.ArgumentParser(description='A model to adapt MAML trained model to a specific task')
    parser.add_argument('root', help='root folder that contained all task data folders', type=str)
    parser.add_argument('out_path', help='path to save test_results', type=str)
    parser.add_argument('--input_weights', help='root folder that contains all the input files', type=str, required=True)
    parser.add_argument('--train_size', help='train size from adaptation target data', default=0.1, type=float, required=False)
    parser.add_argument('--valid_size', help='validation data size from adaptation data', default=0.1, type=float, required=False)
    parser.add_argument('--lr', help='learning rate for model adaptation', default=0.001, type=float, required=False) # form formality
    parser.add_argument('--batch_size', help='batch size for daaloading', default=20, type=int, required=False)
    parser.add_argument('--samle_size', help='input weights root folder', type=int, required=False, default=10)
    parser.add_argument('--freq', help='frequency of test based on number of adaptations done', default=4, type=int, required=False)
    
    args= parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseArgs()
    model_type = ['classic', 'maml']
    weight_init = ['weight_random', 'weight_imagenet']
    i = 0
    for d_fold in os.listdir(args.root):
        data_dir = args.root + '/' + d_fold
        for j in range(args.freq):
            freq_fold = '0'+str(j)
            out_fold = args.out_path + '/{}/test/{}/{}%'.format(freq_fold, d_fold, args.samle_size)
            tester = Tester(root=data_dir,
                            out_path=out_fold,
                            batch_size=args.batch_size,
                            tr_size=args.train_size,
                            v_size=args.valid_size)
            
            for mtype in model_type:
                for init in weight_init:
                    checkfold  = args.input_weights + '/' + freq_fold + '/adapt/{}/10%/{}/{}'.format(d_fold, mtype, init)
                    checkpoint_lists = set(glob(checkfold + '/*.pth'))
                    checkpoint = sortAlphnumeric(checkpoint_lists)[-1]
                    if init == 'weight_random':
                        weight = None
                    elif init == 'weight_imagenet':
                        weight = 'imagenet'
                    else:
                        raise ValueError(f'initialization type{init} not known')
                    tester.test(init_weight=weight, model_type=mtype, checkpoint= checkpoint, report=True)
                    
            
            
        