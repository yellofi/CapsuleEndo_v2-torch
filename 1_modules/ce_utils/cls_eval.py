import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from sklearn.metrics import confusion_matrix
from itertools import product
from sklearn.metrics import roc_curve, auc

def extract_score(model, test_X, batch_size=20):
#     model.evaluation(test_X, test_Y, batch_size)
    model.inference(test_X, batch_size)
    return model.score

class cls_metric:
    def __init__(self, data_y, score):
        self.true = data_y 
        self.score = score
        self.pred = np.argmax(score, axis = 1)
        self.accr = 100*np.mean(np.equal(self.true, self.pred))
        self.cm = confusion_matrix(self.true, self.pred)
    
    def confusion_mat(self, class_name, x_angle = 0, 
                      save_dir = './results/1_confusion_mat/', save_name = None):
        
        def plot_cm(cm, value_size, mode):

            plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
            thresh = cm.max()/2.
            for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
                if mode == 'percent':
                    value = np.round(cm[i, j]/(np.sum(cm, 1)[i]), 3)
                elif mode == 'num':
                    value = cm[i, j]
                plt.text(j, i, value,
                         fontsize = value_size,
                         horizontalalignment = 'center',
                         color = 'white' if cm[i, j] > thresh else 'black')
            plt.ylabel('Actual', fontsize = 20)
            plt.xlabel('Predicted', fontsize = 20)
            plt.xticks([i for i in range(len(class_name))], class_name, rotation=x_angle, fontsize = 18)
            plt.yticks([i for i in range(len(class_name))], class_name, rotation=0, fontsize = 18)
         
        if save_dir != None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
        if save_name == None:
            save_name = 'model'
        
        modes = ['num','percent']
        for i, m in zip(range(len(modes)), modes):
            fig = plt.figure(figsize = (13, 10))
            plot_cm(self.cm, value_size = 22, mode = m)   
            plt.show()
            if save_dir != None:
                fig.savefig(save_dir + '{}_CM_{}.png'.format(save_name, m), bbox_inches='tight')
            plt.close(fig)
            
    def cm2metric(self):

#         assert 0 not in np.sum(self.cm, axis = 0),'Totally misclassified class exists'
        
        if 0 in np.sum(self.cm, axis = 0):
            print('Totally misclassified class exists')

        if len(self.cm) == 2:

            acc = round(100*np.trace(self.cm) / np.sum(self.cm), 2)
            sen = round(100*self.cm[1, 1] / np.sum(self.cm[1, :]), 2)
            spe = round(100*self.cm[0, 0] / np.sum(self.cm[0, :]), 2)
    
            if not (np.sum(self.cm[:, 0]) == 0 and np.sum(self.cm[:, 1]) == 0):
                npv = round(100*self.cm[0, 0] / np.sum(self.cm[:, 0]), 2)
                ppv = round(100*self.cm[1, 1] / np.sum(self.cm[:, 1]), 2)
                
            elif np.sum(self.cm[:, 0]) == 0:
                npv = 0
                ppv = round(100*self.cm[1, 1] / np.sum(self.cm[:, 1]), 2)
            elif np.sum(self.cm[:, 1]) == 0:
                npv = round(100*self.cm[0, 0] / np.sum(self.cm[:, 0]), 2)
                ppv = 0
                
            return [acc, sen, spe, npv, ppv]

        elif len(self.cm) >2:
            acc = round(100*np.trace(self.cm) / np.sum(self.cm), 2)
            spe = round(100*self.cm[0, 0] / np.sum(self.cm[0, :]), 2)
            npv = round(100*self.cm[0, 0] / np.sum(self.cm[:, 0]), 2)

            sen = np.zeros(len(self.cm)-1)
            ppv = np.zeros(len(self.cm)-1)

            for i in range(1, len(self.cm)):
                sen[i-1] = np.sum(self.cm[i, :])*(self.cm[i, i] / np.sum(self.cm[i, :]))
                if np.sum(self.cm[:, i]) > 0:
                    ppv[i-1] = np.sum(self.cm[:, i])*(self.cm[i, i] / np.sum(self.cm[:, i]))
                elif np.sum(self.cm[:, i]) == 0:
                    ppv[i-1] = 0

            sen = round(100*np.sum(sen)/np.sum(self.cm[1:, :]), 2)
            ppv = round(100*np.sum(ppv)/np.sum(self.cm[:, 1:]), 2)
        
            return [acc, sen, spe, npv, ppv]

    def confusion_idx(self, cls = 'positive'):
        if cls == 'positive':
            fn_idx = np.where(self.true - self.pred == 1)[0]
            tp_idx = np.setxor1d(np.where(self.true == 1), fn_idx)
            return tp_idx, fn_idx
        elif cls == 'negative':
            fp_idx = np.where(self.true - self.pred == -1)[0]
            tn_idx = np.setxor1d(np.where(self.true == 0), fp_idx)
            return tn_idx, fp_idx
     
    

        
def get_accuracy_by_type(model, filename, X, Y, target_category = None):
    
    def get_accuracy(model, x, y):
        model.evaluation(x, y, batch_size = 20, print_accr = False)
        return model.accr
    
    label_dir = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed'
    label = pd.read_csv(label_dir + '/label.csv', index_col = 0)
    
    types = []
    accrs = []
    
    if target_category == None:
        target_category = label.columns[:-1]
    
    for sub_cls in target_category:
    
        x, y = [], []
        for name, img, cls in zip(filename, X, Y):
            if label.loc[('').join(name.split('__c_-_-_-'))][sub_cls] == 1:
                x.append(img)
                y.append(cls)
        
        if len(x) != 0:
            types.append(sub_cls)
            x = np.concatenate([x])
            y = np.asarray(y)

            accrs.append(get_accuracy(model, x, y)) 
    return types, accrs

def plot_roc_curve(model_list, score_list, label, save_dir = './results/', optimal_threshold = None):
    
    fig, ax = plt.subplots(1, 1, figsize= (10, 8))
    
    if len(model_list) == 1:
        fpr, tpr, thresholds = roc_curve(label, score_list[0][:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw = 2, color = 'r', label='{} (AUC = {:.4f})'.format(model_list[0], roc_auc))
        
    elif len(model_list) > 1:
        for model, score in zip(model_list, score_list):
            fpr, tpr, thresholds = roc_curve(label, score[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw = 2, label='{} (AUC = {:.4f})'.format(model, roc_auc))

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(labelsize = 15)
    ax.set_xlabel('False Positive Rate', fontsize = 20)
    ax.set_ylabel('True Positive Rate', fontsize = 20)

    ax.set_xlim([0, 1.0])
    ax.set_ylim([0.0, 1])

    if optimal_threshold != None:
        gmean_score = np.sqrt(tpr * (1-fpr))
        idx = np.argmax(gmean_score)

        ax_xlim, ax_ylim = ax.get_xlim(), ax.get_ylim()

        width = ax_xlim[1] - ax_xlim[0]
        height = ax_ylim[1] - ax_ylim[0]

        ax.scatter(fpr[idx], tpr[idx], color = 'k')
        ax.axhline(tpr[idx], color = 'k', linestyle='dashed', linewidth=1, xmax=(fpr[idx]- ax_xlim[0])/width)
        ax.axvline(fpr[idx], color = 'k', linestyle='dashed', linewidth=1, ymax=(tpr[idx]- ax_ylim[0])/height)
        ax.text(fpr[idx] + 0.01, tpr[idx] - 0.02, '({:.2f}, {:.2f})'.format(100*tpr[idx], 100*(1-fpr[idx])), fontsize = 15)

    ax.legend(fontsize = 17, borderpad = 0.4, edgecolor = 'k', loc="lower right")

    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.grid()

    plt.show()

    if save_dir != None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_dir + 'roc_curve.png', bbox_inches='tight')