'''
分割序列数据集，将数据集分成80%训练， 18%测试  2%验证
'''
import glob
import os
import numpy as np
from torch.utils.data.dataset import random_split

#读取数据文件，每一行为一个序列，一个序列由10张图片组成
def readDataFile(path):
    sequences_file = path
    sequences_line = open(sequences_file, mode="r").readlines()
    sequences = []
    for line in sequences_line:
        sequence = line.split("\t")
        sequence = sequence[:-1]

        if len(sequence) != 10:
            print("%s is error" % line)
            continue

        sequences.append(line)

    return sequences


def split_sequence(target_file):

    # 划分数据集
    sequences = readDataFile(target_file)
    assert len(sequences) > 10

    # 80:18:2
    train_len = int(len(sequences) * 0.8)
    valid_len = int(len(sequences) * 0.18)
    test_len = len(sequences) - train_len - valid_len
    train_sequences, valid_sequences,test_sequences = random_split(sequences, (train_len, valid_len,test_len))

    # write to file
    work_path,target_file_name = os.path.split(target_file)
    filename_suffix, _ = os.path.splitext(target_file_name)
    trainset_file = open(os.path.join(work_path,filename_suffix + "_train.txt"), mode='w+')
    valset_file = open(os.path.join(work_path,filename_suffix + "_val.txt"), mode='w+')
    testset_file = open(os.path.join(work_path,filename_suffix + "_test.txt"), mode='w+')

    written = 0
    for line in train_sequences:
        trainset_file.write(line)
        written = written + 1
    assert (written == train_len)
    print("Train set is {0}".format(written))

    written = 0
    for line in valid_sequences:
        valset_file.write(line)
        written = written + 1
    assert (written == valid_len)
    print("Valid set is {0}".format(written))

    written = 0
    for line in test_sequences:
        testset_file.write(line)
        written = written + 1
    assert (written == test_len)
    print("Test set is {0}".format(written))

#todo 调用一次
#split_sequence("sf/train_2014.txt")

file_list = glob.glob(os.path.join("sf/", '*_predict.txt'))
for file in file_list:
    split_sequence(file)
