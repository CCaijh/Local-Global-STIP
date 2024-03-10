import os
import argparse
import numpy as np
from core.data_provider import datasets_factory
from core.models.model_factoryv2 import Model
import core.trainer as trainer
# import pynvml
from torch.utils.data.dataset import random_split
from core.utils.logger import setup_logger
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# pynvml.nvmlInit()
configs = argparse.ArgumentParser(description='STIP')
configs.add_argument('--dataset', type=str, default='sf')
configs = configs.parse_args()
configs.tied = True
args = configs
if configs.dataset == 'sf':
    from configs.sf_configs import parser
else:
    exit(0)

parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()
args.tied = True

#处理输出目录
str_now = datetime.now().strftime('%Y%m%d%H%M%S') #时间戳
args.save_dir = os.path.join(args.save_dir,str_now,"weights")
args.gen_frm_dir = os.path.join(args.gen_frm_dir,str_now)
args.perforamnce_dir = os.path.join(args.perforamnce_dir,str_now)

def schedule_sampling(eta, itr, batch_size):
    zeros = np.zeros((batch_size,
                      args.total_length - args.input_length - 1,
                      # args.img_time,
                      args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    print('eta: ', eta)
    random_flip = np.random.random_sample(
        (batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_height // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_height // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag

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

        sequences.append(sequence)

    return sequences

def train_wrapper(model):
    logger = setup_logger("stip", args.gen_frm_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(args.gen_frm_dir))
    logger.info(args)

    begin = 0
    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # meminfo_begin = pynvml.nvmlDeviceGetMemoryInfo(handle)

    if args.pretrained_model_pm and args.pretrained_model_d:
        model.load(args.pretrained_model_pm, args.pretrained_model_d)
        begin = int(args.pretrained_model_pm.split('-')[-1])
    # todo
    # 划分数据集
    sequences1 = readDataFile(args.data_train_path)
    sequences2 = readDataFile(args.data_val_path)
    # assert len(sequences) > 10
    #
    # # 8:2
    # train_len = int(len(sequences) * 0.8)
    # valid_len = len(sequences) - train_len
    # train_sequences, valid_sequences = random_split(sequences, (train_len, valid_len))

    # load data
    train_input_handle = datasets_factory.data_provider(configs=args,
                                                        sequences=sequences1,
                                                        dataset=args.dataset,
                                                        batch_size=args.batch_size,
                                                        is_training=True,
                                                        is_shuffle=True)

    val_input_handle = datasets_factory.data_provider(configs=args,
                                                      sequences=sequences2,
                                                      dataset=args.dataset,
                                                      batch_size=args.batch_size,
                                                      is_training=False,
                                                      is_shuffle=False)
    eta = args.sampling_start_value
    eta -= (begin * args.sampling_changing_rate)
    itr = begin
    for epoch in range(0, args.max_epoches):
        if itr > args.max_iterations:
            break
        for ims in train_input_handle:
            if itr > args.max_iterations:
                break
            batch_size = ims.shape[0]
            eta, real_input_flag = schedule_sampling(eta, itr, batch_size)
            logger.info("Eta:{:.8f}".format(eta))
            if itr % args.test_interval == 0 and itr > 0:
                print('Validate:')
                trainer.test(model, val_input_handle, args, itr)
            trainer.train(model, ims, real_input_flag, args, itr)
            if itr % args.snapshot_interval == 0 and itr > begin:
                model.save(itr)
            itr += 1
            # meminfo_end = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print("GPU memory:%dM" % ((meminfo_end.used - meminfo_begin.used) / (1024 ** 2)))


def test_wrapper(model):
    logger = setup_logger("stip", args.gen_frm_dir, if_train=False)
    logger.info("Saving model in the path :{}".format(args.gen_frm_dir))
    logger.info(args)

    model.load(args.pretrained_model_pm, args.pretrained_model_d)

    # 处理数据集
    sequences = readDataFile(args.data_train_path)
    assert len(sequences) > 0

    test_input_handle = datasets_factory.data_provider(configs=args,
                                                       sequences=sequences,
                                                       dataset=args.dataset,
                                                       batch_size=args.batch_size,
                                                       is_training=False,
                                                       is_shuffle=False)

    itr = 1
    for i in range(itr):
        trainer.test(model, test_input_handle, args, itr)


if __name__ == '__main__':
    print('Initializing models')
    model = Model(args)
    if args.is_training:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        train_wrapper(model)
    else:
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        test_wrapper(model)
