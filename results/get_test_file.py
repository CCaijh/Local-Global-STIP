import os
import torch
import numpy as np

database_dir = "E:\\kjzhongxin\\stage2\\ConvLSTM2\\data\\2014-full"
dest_dir = "20220905_hand"
rec_file = "E:/kjzhongxin/stage2/LDL/results/LDL/20220905_hand.txt"

if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

rec_lines = open(rec_file, mode="r").readlines()
rec_results = {}
for rec_line in rec_lines:
    rec_target,rec_result = rec_line.rstrip().split(" ")

    _,rec_target_name = os.path.split(rec_target)
    rec_target_name,_ = os.path.splitext(rec_target_name)

    date = rec_target_name.split("_")[0]

    target_file_name = os.path.join(database_dir,date+'.png')
    os.system ("copy %s %s" % (target_file_name, dest_dir))

