import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.pyplot as plt

''' Class to test the models '''
class Test_Model:
    def __init__(self, predicts, targets):
        self.confusion = None
        self.predicts = predicts
        self.targets = targets
        confusion_vector = predicts / targets

        self.TP = torch.sum(confusion_vector == 1, dim=0).double()
        self.FP = torch.sum(confusion_vector == float('inf'), dim=0).double()
        self.TN = torch.sum(torch.isnan(confusion_vector), dim=0).double()
        self.FN = torch.sum(confusion_vector == 0, dim=0).double()

        self.acc = torch.sum((self.predicts == self.targets).double(), dim=0) / self.predicts.shape[0]
        self.pre = self.TP / (self.TP + self.FP)
        self.rec = self.TP / (self.TP + self.FN)
        self.bcr = (self.pre + self. rec) / 2
        self.f1 = (2 * self.pre * self.rec) / (self.pre + self.rec)

        self.acc[torch.isnan(self.acc)] = 0
        self.pre[torch.isnan(self.pre)] = 0
        self.rec[torch.isnan(self.rec)] = 0
        self.bcr[torch.isnan(self.bcr)] = 0
        

    def accuracy(self):
        return self.acc

    def precision(self):
        return self.pre

    def recall(self):
        return self.rec

    def BCR(self):
        return self.bcr

    def F1(self):
        return self.f1
        
    def avg_accuracy(self):
        return torch.mean(self.acc)

    def avg_precision(self):
        return torch.mean(self.pre)

    def avg_recall(self):
        return torch.mean(self.rec)

    def avg_BCR(self):
        return torch.mean(self.bcr)
    
    def avg_F1(self):
        return torch.mean(self.f1)
    
    def confusion_matrix(self):
        d = self.targets.shape[1]
        conf = np.zeros(shape=(d+1, d+1))

        for (y_true, y_pred) in zip(self.targets, self.predicts):
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().numpy()

            indices_tar = set(y_true.nonzero()[0])
            indices_pred = set(y_pred.nonzero()[0])
            intersection = indices_tar & indices_pred

            if len(indices_tar) == 0 and len(indices_pred) == 0:
                conf[d][d] += 1
            elif len(indices_tar) == 0 and len(indices_pred) > 0:
                for ind in indices_pred:
                    conf[ind][d] += 1
            elif len(indices_tar) and len(indices_pred) == 0:
                for ind in indices_pred:
                    conf[d][ind] += 1
            else:
                if len(intersection) == 0:
                    for i in indices_pred:
                        for j in indices_tar:
                            conf[i][j] += 1
                else:
                    for k in intersection:
                        conf[k][k] += 1
                    tar2 = indices_tar - intersection
                    pred2 = indices_pred - intersection
                    if len(tar2) == 0 and len(pred2) == 0:
                        continue
                    elif len(tar2) == 0 and len(pred2) > 0:
                        for ind in pred2:
                            conf[ind][d] += 1
                    elif len(tar2) > 0 and len(pred2) == 0:
                        for ind in tar2:
                            conf[d][ind] += 1
                    else:
                        for i in pred2:
                            for j in tar2:
                                conf[i][j] += 1

        self.confusion = conf / sum(conf)
        return conf
            

    if __name__=="__main__":
        prediction = torch.tensor([
            [1, 0, 0., 1],
            [1, 1., 0., 1],
            [0, 1., 0., 1]
        ])

        target = torch.tensor([
            [1, 1, 0., 1],
            [1, 0, 1., 1],
            [1, 1., 0., 1]
        ])

        test_model = Test_Model(prediction, target)