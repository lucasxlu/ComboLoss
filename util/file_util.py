import os
import numpy as np
import pandas as pd


def mkdirs_if_not_exist(dir_name):
    """
    create new folder if not exist
    :param dir_name:
    :return:
    """
    if not os.path.isdir(dir_name) or not os.path.exists(dir_name):
        os.makedirs(dir_name)


def score_stat(data_name):
    mp = {}
    datalist = []
    if data_name == 'SCUT-FBP5500':
        train_txt = "/Users/lucasx/Documents/Dataset/SCUT-FBP5500/train_test_files/split_of_60%training and 40%testing/train.txt"
        with open(train_txt, mode='rt', encoding='utf-8') as f:
            for line in f.readlines():
                score = float(line.split(' ')[1])
                datalist.append(score)
                cls = round(score) - 1
                if cls in mp.keys():
                    mp[cls] += 1
                else:
                    mp[cls] = 1
        datalist = np.array(datalist)
        q1 = np.percentile(datalist, 25)
        median = np.percentile(datalist, 50)
        q3 = np.percentile(datalist, 75)
        print(q1, median, q3)
    elif data_name == 'HotOrNot':
        anno_csv = '/Users/lucasx/Documents/Dataset/eccv2010_beauty_data/eccv2010_split5.csv'
        df = pd.read_csv(anno_csv, header=None)
        for score in df[1].tolist():
            score = float(score)
            if score < -1:
                cls = 0
            elif -1 <= score < 1:
                cls = 1
            elif score >= 1:
                cls = 2
            if cls in mp.keys():
                mp[cls] += 1
            else:
                mp[cls] = 1
    elif data_name == 'SCUT-FBP-CV':
        anno_txt = '/Users/lucasx/Documents/Dataset/SCUT-FBP5500/train_test_files/5_folders_cross_validations_files' \
                   '/cross_validation_5/train_5.txt'
        with open(anno_txt, mode='rt', encoding='utf-8') as f:
            for line in f.readlines():
                score = float(line.split(' ')[1])
                datalist.append(score)
                cls = round(score) - 1
                if cls in mp.keys():
                    mp[cls] += 1
                else:
                    mp[cls] = 1
        datalist = np.array(datalist)
        q1 = np.percentile(datalist, 25)
        median = np.percentile(datalist, 50)
        q3 = np.percentile(datalist, 75)
        print(q1, median, q3)
    elif data_name == 'SCUT-FBP':
        pass

    print(mp)
    print([round(max(mp.values()) / mp[i], 2) for i in range(3)])


if __name__ == '__main__':
    score_stat("HotOrNot")
