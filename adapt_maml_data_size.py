import argparse
import os
from Anew_copy_main_maml_unet import Adapter

def parseArgs():
    parser = argparse.ArgumentParser(description='A model to adapt MAML trained model to a specific task')
    parser.add_argument('--root', help='root folder that contained all task data folders', type=str)
    parser.add_argument('--weight_path', help='path to save weight', type=str)
    parser.add_argument('--train_size', help='train size from adaptation target data', default=0.1, type=float, required=False)
    parser.add_argument('--valid_size', help='validation data size from adaptation data', default=0.1, type=float, required=False)
    parser.add_argument('--lr', help='learning rate for model adaptation', default=0.1, type=float, required=False)
    parser.add_argument('--batch_size', help='batch size for daaloading', default=20, type=int, required=False)
    parser.add_argument('--epochs', help='epochs to train adaptation model', default=120, type=int, required=False)
    parser.add_argument('--weight', help='input weights root folder', type=str, required=False)
    #parser.add_argument('--out_root', help='Output root folder to save the results', required=True, type=str)
    # parser.add_argument('--freq', help='frequency of iteratios to repeat the experiment', type=int, required=False, default=4)
    
    args= parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArgs()
    dsize = [0.15, 0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for folder in os.listdir(args.root):
        data_path = args.root + '/' + folder
        model_type = ['maml']
        weight_init = ['weight_imagenet']    # D:\MEAT_LEARN\MAML_explore_lr\train\maml
        weight = {'tza_nduta_21oct2016':'/home/getch/ssl/output/MAMLw/dynamic_weight/adapt/tza_nduta_21oct2016/1e-05=with=1e-06/maml/weight_imagenet/checkpoint.pth',
                'minawa_12feb2017':'/home/getch/ssl/output/MAMLw/dynamic_weight/adapt/minawa_12feb2017/1e-06=with=1e-06/maml/weight_imagenet/checkpoint.pth',
                'kutuplong_feb_2018':'/home/getch/ssl/output/MAMLw/dynamic_weight/adapt/kutuplong_feb_2018/0.001=with=0.01/maml/weight_imagenet/checkpoint.pth',
                'ken_deg_08Apr2017':'/home/getch/ssl/output/MAMLw/dynamic_weight/adapt/ken_deg_08Apr2017/1e-05=with=0.01/maml/weight_imagenet/checkpoint.pth'}
        for size in dsize:
            final_out_root = args.weight_path + '/' + '/adapt_weighte_data_size/' + folder + '/{}'.format(str(size))
            print(final_out_root)
            
            if not os.path.exists(final_out_root):
                os.makedirs(final_out_root, exist_ok=True)
            adapter = Adapter(root=data_path,
                    tr_size=size,
                    v_size=args.valid_size,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    weight_path=final_out_root) # 'D:/meta_l2l/out2'
            for mtype in model_type:
                for init in weight_init:
                    if init == 'weight_random':
                        init_weight = None
                    else:
                        init_weight = 'imagenet'
                    assert os.path.exists(weight[folder]), 'checkpoint {} not exists'.format(weight[folder])
                    print(weight[folder])
                    adapter.adapt(init_weight=init_weight, model_type=mtype, checkpoint=weight[folder])
