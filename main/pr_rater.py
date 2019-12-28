import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class PrRater:
    def __init__(self, filename, score, cls, prob):
        self.filename = filename
        self.score = score
        self.cls = cls
        self.prob = prob


def predict(prn1_output="C:/Users/29140/Desktop/PRN1.xlsx", prn2_output="C:/Users/29140/Desktop/PRN2.xlsx", omega=0.1):
    df1 = pd.read_excel(prn1_output)
    df2 = pd.read_excel(prn2_output)

    dict1 = {}
    dict2 = {}

    filenames1 = df1['filename']
    gts1 = df1['gt']
    preds1 = df1['pred']
    probs1 = df1['prob']

    for i in range(len(filenames1)):
        dict1[filenames1[i]] = {
            'gt_cls': gts1[i],
            'pred_cls': preds1[i],
            'prob': probs1[i],
        }

    filenames2 = df2['filename']
    gts2 = df2['gt']
    preds2 = df2['pred']

    for j in range(len(filenames2)):
        dict2[filenames2[j]] = {
            'gt': gts2[j],
            'pred_score': preds2[j]
        }

    prediction = []
    groundtruth = []

    for k, v in dict1.items():
        if v['pred_cls'] == 0:
            ci = -1
        elif v['pred_cls'] == 1:
            ci = 1

        yi = (1 + ci * omega * (1 - v['prob'])) * dict2[k]['pred_score']

        prediction.append(yi)
        groundtruth.append(dict2[k]['gt'])

    mae = round(mean_absolute_error(np.array(groundtruth).ravel(), np.array(prediction).ravel()), 4)
    rmse = round(np.math.sqrt(mean_squared_error(np.array(groundtruth).ravel(), np.array(prediction).ravel())), 4)
    pc = round(np.corrcoef(np.array(groundtruth).ravel(), np.array(prediction).ravel())[0, 1], 4)

    print('===============The Mean Absolute Error of {0} is {1}===================='.format('ComboLoss', mae))
    print('===============The Root Mean Square Error of {0} is {1}===================='.format('ComboLoss', rmse))
    print('===============The Pearson Correlation of {0} is {1}===================='.format('ComboLoss', pc))


if __name__ == '__main__':
    predict()
