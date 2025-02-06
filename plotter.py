import torch 
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def give_aplot():
    print('======================== AA ==========================')
    mtype = ['MAML', 'MAMLss','MIXED']
    sub = ['maml','classic']
    root = '/adapt_test_size'
    root = '/adapt_test_size'
    main = '/adapt_test_size/{}/{}/{}/{}/weight_imagenet/test_summary.npy'
    main1 = '/adapt_test_size/{}/{}/{}/{}/weight_imagenet/Ftest_summary.npy'
    alls = {}
    for model in mtype:
        mm = {}
        for folder in os.listdir(root + '/' + model):
            print(f"============================ {model} : {folder} =============================")
            f11 = []
            iou1 = []
            for d_size in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                if model in ['MAML', 'MAMLss']:
                    fill = 'maml'
                else:
                    fill = 'classic'
                if model == 'MAMLss':
                    file = main1.format(model, folder, str(d_size), fill)
                else:
                    file = main.format(model, folder, str(d_size), fill)
                if os.path.exists(file):
                    # print('A')
                    aa = np.load(open(file, 'rb'), allow_pickle=True).item()
                    # print('B')
                    if type(aa['iou']) == np.float:
                        iou = aa['iou']
                        f1 = aa['f1s']
                    else:
                        iou = aa['iou'].item()
                        f1 = aa['f1s'].item()
                    f11.append(f1)
                    iou1.append(iou)
                    print('d_size: ', d_size, 'iou: ', iou, 'f1: ', f1)
            mm[folder] = {'iou':iou1, 'f1':f11}
        alls[model] = mm


    print('======================== BB ==========================')
    MODEL = []
    SITE = []
    IOU = []
    F1 = []
    SIZE = []
    size = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    for key in alls.keys():
        for keys in alls[key].keys():
            MODEL+=[key]*8
            SITE+=[keys]*8
            IOU+=alls[key][keys]['iou']
            F1+=alls[key][keys]['f1']
            SIZE+=size
    df = pd.DataFrame(list(zip(MODEL, SITE, SIZE, IOU, F1)), columns =['Model', 'Target', 'data size', 'MIoU', 'F-1'])
    df['data size'] = df['data size'].astype(str)

    print('======================== CC ==========================')
    legs = ['Deghale-2017', 'Kutuplong-2018', 'Minawao-2017', 'Nduta-2016'] # list(np.unique(SITE))
    fig, axx = plt.subplots(2,3, figsize = (10,8), sharex=True, sharey=True)
    for i, model in enumerate(['MIXED', 'MAML', 'MAMLss']):
        df_cp = df[df['Model'] == model]
        sns.barplot(ax=axx[0][i], x="Target", y="MIoU", hue='data size', data=df_cp, ci=None)
        sns.barplot(ax=axx[1][i], x="Target", y="F-1", hue='data size', data=df_cp, ci=None)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:8], labels[:8], loc='center right', title='Ratio of \nadaptation\ndata')
    for i in range(2):
        for j in range(3):
            axx[i][j].legend_.remove()
            axx[i][j].set_ylim(0.4, 0.8)
            if j >= 1:
                axx[i][j].set_ylabel("")
            if i == 1:
                plt.sca(axx[i][j])
                plt.xticks(range(0,4), legs, color='blue', rotation=80)
            if i == 0:
                axx[i][j].set_title(['MIXED', 'MAML', 'MAMLss'][j])
                axx[i][j].set_xlabel("")
    fig.subplots_adjust(hspace=0.03, wspace=0)
    plt.savefig(root + '/adaptation_data_sizFinal.png', format='png', dpi=350,  bbox_inches='tight')
    plt.show()
    
    print('======================== Saved ==========================')
    
if __name__ == '__main__':
    give_aplot()
