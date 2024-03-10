
import os
import torch
import numpy as np

rec_file = "results/output/20220905/predict.txt"
label_file = "E:/kjzhongxin/stage2/ConvLSTM2/data/2014-label.txt"

error = 0
correct = 0

#读取GT
gt_lines = open(label_file, mode="r").readlines()
gt = {}
for gt_line in gt_lines:
    gt_target, gt_result = gt_line.rstrip().split(" ")
    gt_target_name, _ = os.path.splitext(gt_target)
    gt[gt_target_name] = gt_result

#读取使用STIP输出的图片的预测结果
rec_lines = open(rec_file, mode="r").readlines()
rec_results = {}
for rec_line in rec_lines:
    rec_target,rec_result = rec_line.rstrip().split(" ")

    _,rec_target_name = os.path.split(rec_target)
    rec_target_name,_ = os.path.splitext(rec_target_name)

    date = rec_target_name.split("_")[0] + "00"
    no = int(rec_target_name.split("_")[2])
    #if no == 0:
    #    continue

    rec_results[date] = rec_result
    if date not in gt:
        print("Not found {} : {} ".format(date,rec_result))
        continue

    if gt[date] == rec_result:
        correct += 1
    else:
        error += 1
        print("{} : {}/{}".format(rec_target_name, rec_result, gt[date]))

print("Error {}: Correct {}, per: {:.2f}%".format(error,correct,float(correct)/float(error + correct) * 100))






