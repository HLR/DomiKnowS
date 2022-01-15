from regr.program.metric import MetricTracker, CMWithLogitsMetric
from regr.utils import wrap_batch
import torch

class PRF1SepTracker(MetricTracker):
    def __init__(self, metric=CMWithLogitsMetric(),confusion_matrix=True):
        super().__init__(metric)
        self.confusion_matrix=confusion_matrix

    def forward(self, values):
        if not "class_names" in values[0]:

            CM = wrap_batch(values)

            if isinstance(CM['TP'], list):
                tp = sum(CM['TP'])
            else:
                tp = CM['TP'].sum().float()

            if isinstance(CM['FP'], list):
                fp = sum(CM['FP'])
            else:
                fp = CM['FP'].sum().float()

            if isinstance(CM['FN'], list):
                fn = sum(CM['FN'])
            else:
                fn = CM['FN'].sum().float()

            if isinstance(CM['TN'], list):
                tn = sum(CM['TN'])
            else:
                tn = CM['TN'].sum().float()

            if tp:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 * p * r / (p + r)
            else:
                p = torch.zeros_like(torch.tensor(tp))
                r = torch.zeros_like(torch.tensor(tp))
                f1 = torch.zeros_like(torch.tensor(tp))
            if (tp + fp + fn + tn):
                accuracy=(tp + tn) / (tp + fp + fn + tn)
            return {'tp': tp, 'fp': fp, 'fn': fn, 'tn':tn, 'P': p, 'R': r, 'F1': f1,"accuracy":accuracy}
        else:
            output={}
            names=values[0]["class_names"][:]
            n=len(names)

            matrix=[[0 for i in range(n)] for j in range(n)]
            for batch in values:
                for label,pred in zip(batch["labels"],batch["preds"]):
                    matrix[label][pred]+=1
            if self.confusion_matrix:
                output[str(names)]=matrix
            for name in names:
                TP,TN,FP,FN=frp_from_matrix(names.index(name),matrix)
                if (TP+FP):
                    output[name+" Precision"]=TP/(TP+FP)
                else:
                    output[name + " Precision"] = 0
                if (TP+FN):
                    output[name + " Recall"] =TP/(TP+FN)
                else:
                    output[name + " Recall"]=0
                if (output[name+" Precision"]+output[name + " Recall"]):
                    output[name + " F1"] =2*(output[name+" Precision"]*output[name + " Recall"])/(output[name+" Precision"]+output[name + " Recall"])
                else:
                    output[name + " F1"]=0
                if (TP+TN+FP+FN):
                    output[name + " Accuracy"] =(TP+TN)/(TP+TN+FP+FN)
                else:
                    output[name + " Accuracy"]=0
            output["Total Accuracy of All Classes"]=sum([matrix[i][i] for i in range(n)])/sum([sum(matrix[i]) for i in range(n)])
            return output
