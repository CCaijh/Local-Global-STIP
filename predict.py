import os
import argparse
import torch
import cv2
import numpy as np
from core.models.model_factoryv2 import Model
from core.utils.logger import setup_logger
from datetime import datetime
from configs.sf_configs import parser
from PIL import Image
from core.utils import preprocess
from torchvision import transforms
import codecs
import lpips
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data.dataset import random_split

#读取数据文件，每一行为一个序列，一个序列由10张图片组成
def ReadDataFile(path):
    sequences_file = path
    sequences_line = open(sequences_file, mode="r").readlines()
    sequences = []
    for line in sequences_line:
        sequence = line.rstrip().split("\t")
        #sequence = sequence[:-1]

        if len(sequence) != 10:
            print("%s is error" % line)
            continue

        sequences.append(sequence)

    return sequences

#读取一个序列，共10张 : 包含两张结果图片
# 此处需要修改，实际使用的时候，输入图片只会有8张
def ReadSequence(sequence, img_height, img_width, img_channel,image_path, patch_size, transform=None):
    assert len(sequence) == 10

    data_slice = np.ndarray(shape=(10, img_height, img_width, img_channel), dtype=np.uint8)
    for i in range(10):
        image = Image.open(os.path.join(image_path, sequence[i][0:4], sequence[i]))
        image = image.resize((img_width, img_width))
        try:
            data_slice[i, :] = np.array(image)
        except:
            print(data_slice.shape)
            print('%d,%s' % (i, os.path.join(image_path, sequence[i][0:4], sequence[i])))

    video_x = preprocess.reshape_patch(data_slice, patch_size)
    sample = video_x

    if transform:
        sample = transform(sample)

    return sample

#transform组件
class Norm(object):
    def __init__(self, max=255):
        self.max = max

    def __call__(self, sample):
        video_x = sample
        new_video_x = video_x / self.max
        return new_video_x

#transform组件
class ToTensor(object):

    def __call__(self, sample):
        video_x = sample
        video_x = video_x.transpose((0, 3, 1, 2))
        video_x = np.array(video_x)
        return torch.from_numpy(video_x).float()


# to_analysis = False表示不输出性能分析文件
# sequence是为了获取预测的文件名
def test(model, output_path , sequence, test_ims, configs, itr, to_analysis = False, to_visual = False):

    # 处理path
    visual_path = os.path.join(output_path,"v")
    predict_path = os.path.join(output_path,"p")
    if not os.path.exists(visual_path):
        os.mkdir(visual_path)
    if not os.path.exists(predict_path):
        os.mkdir(predict_path)

    #预测图片
    batch_size = test_ims.shape[0]
    real_input_flag = np.zeros(
        (batch_size,
         configs.total_length - configs.input_length - 1,
         configs.img_height // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    img_gen, features = model.test(test_ims, real_input_flag)

    #img_gen1 = np.zeros(img_gen.shape)
    #img_gen1[:, :, :,128:224, :] = img_gen[:, :, :,128:224, :]
    #img_gen1 = img_gen1.transpose(0, 1, 3, 4, 2)
    #img_gen1 = preprocess.reshape_patch_back(img_gen1, configs.patch_size)

    img_gen = img_gen.transpose(0, 1, 3, 4, 2)
    img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
    test_ims = test_ims.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)
    test_ims = preprocess.reshape_patch_back(test_ims, configs.patch_size)

    output_length = configs.total_length - configs.input_length
    output_length = min(output_length, configs.total_length - 1)
    img_out = img_gen[:, -output_length:, :]
    #img_out2 = img_gen1[:, -output_length:, :]
    #img_out1 = img_gen[:, -output_length:, 256:448,0:512]



    # 输出单张
    for i in range(output_length):
        output_img = img_out[0, -output_length + i, :]
        output_img = np.maximum(output_img, 0)
        output_img = np.minimum(output_img, 1)
        output_img_name,_ = os.path.splitext(sequence[configs.input_length  + i])
        #cv2.imwrite(os.path.join(predict_path, "%s_%d_%d.png" % (output_img_name, itr, i)), (output_img * 255).astype(np.uint8))

        img_obj = Image.fromarray((output_img * 255).astype(np.uint8))
        img_obj.save(os.path.join(predict_path, "%s_%d_%d.png" % (output_img_name, itr, i)))

        #output_img1 = img_out2[0, -output_length + i, :]
        #output_img1 = np.maximum(output_img1, 0)
        #output_img1 = np.minimum(output_img1, 1)
        #cv2.imwrite(os.path.join(output_path, "%s_%d_%d_ld.png" % (output_img_name, itr, i)),
        #            (output_img1 * 255).astype(np.uint8))


    #可视化
    if to_visual:
        res_width = configs.img_width
        res_height = configs.img_height
        img = np.ones((2 * res_height, configs.total_length * res_width, configs.img_channel))
        name = str(itr) + '.png'
        file_name = os.path.join(visual_path, name)
        for i in range(configs.total_length):
            img[:res_height, i * res_width:(i + 1) * res_width, :] = test_ims[0, i, :]

        for i in range(output_length):
            img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width,
            :] = img_out[0, -output_length + i, :]

        img = np.maximum(img, 0)
        img = np.minimum(img, 1)
        # error
        img_obj = Image.fromarray((img * 255).astype(np.uint8))
        img_obj.save(file_name)
        #cv2.imwrite(file_name, (img * 255).astype(np.uint8))

    #分析性能
    if to_analysis:
        loss_fn = lpips.LPIPS(net='alex', spatial=True).to(configs.device)
        f = codecs.open(output_path +'/performance.txt', 'w+')
        f.truncate()
        avg_mse = 0
        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0
        batch_id = 0
        img_mse, img_psnr, ssim, img_lpips, mse_list, psnr_list, ssim_list, lpips_list = [], [], [], [], [], [], [], []
        for i in range(configs.total_length - configs.input_length):
            img_mse.append(0)
            img_psnr.append(0)
            ssim.append(0)
            img_lpips.append(0)

            mse_list.append(0)
            psnr_list.append(0)
            ssim_list.append(0)
            lpips_list.append(0)

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :]
            gx = img_out[:, i, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            psnr = 0
            t1 = torch.from_numpy((x - 0.5) / 0.5).to(configs.device)
            t1 = t1.permute((0, 3, 1, 2))
            t2 = torch.from_numpy((gx - 0.5) / 0.5).to(configs.device)
            t2 = t2.permute((0, 3, 1, 2))
            shape = t1.shape
            if not shape[1] == 3:
                new_shape = (shape[0], 3, *shape[2:])
                t1.expand(new_shape)
                t2.expand(new_shape)
            d = loss_fn.forward(t1, t2)
            lpips_score = d.mean()
            lpips_score = lpips_score.detach().cpu().numpy() * 100
            for sample_id in range(batch_size):
                mse_tmp = np.square(
                    x[sample_id, :] - gx[sample_id, :]).mean()
                psnr += 10 * np.log10(1 / mse_tmp)
            psnr /= (batch_size)
            img_mse[i] += mse
            img_psnr[i] += psnr
            img_lpips[i] += lpips_score
            mse_list[i] = mse
            psnr_list[i] = psnr
            lpips_list[i] = lpips_score
            avg_mse += mse
            avg_psnr += psnr
            avg_lpips += lpips_score
            score = 0
            for b in range(batch_size):
                score += compare_ssim(x[b, :], gx[b, :], multichannel=True)
            score /= batch_size
            ssim[i] += score
            ssim_list = score
            avg_ssim += score

        f.writelines(str(batch_id) + ',' + str(psnr_list) + ',' + str(mse_list) + ',' + str(lpips_list) + ',' + str( ssim_list) + '\n')

        batch_id = batch_id + 1
        f.close()

        with codecs.open(output_path + '/data.txt', 'w+') as data_write:
            data_write.truncate()
            avg_mse = avg_mse / (batch_id * output_length)
            print('mse per frame: {:.8f}'.format(avg_mse))
            for i in range(configs.total_length - configs.input_length):
                print(str(img_mse[i] / batch_id))
                img_mse[i] = img_mse[i] / batch_id
            data_write.writelines(str(avg_mse)+'\n')
            data_write.writelines(str(img_mse)+'\n')
            avg_psnr = avg_psnr / (batch_id * output_length)
            print('psnr per frame: {:.8f}'.format(avg_psnr))
            for i in range(configs.total_length - configs.input_length):
                print(str(img_psnr[i] / batch_id))
                img_psnr[i] = img_psnr[i] / batch_id
            data_write.writelines(str(avg_psnr)+'\n')
            data_write.writelines(str(img_psnr)+'\n')

            avg_ssim = avg_ssim / (batch_id * output_length)
            print('ssim per frame: {:.8f}'.format(avg_ssim))
            for i in range(configs.total_length - configs.input_length):
                print(str(ssim[i] / batch_id))
                ssim[i] = ssim[i] / batch_id
            data_write.writelines(str(avg_ssim)+'\n')
            data_write.writelines(str(ssim)+'\n')

            avg_lpips = avg_lpips / (batch_id * output_length)
            print('lpips per frame: {:.8f}'.format(avg_lpips))
            for i in range(configs.total_length - configs.input_length):
                print(str(img_lpips[i] / batch_id))
                img_lpips[i] = img_lpips[i] / batch_id
            data_write.writelines(str(avg_lpips)+'\n')
            data_write.writelines(str(img_lpips)+'\n')

if __name__ == '__main__':

    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    args.tied = True
    args.is_training = False

    args.pretrained_model_pm = "results/sf/20220802091932/weights/model_pm.ckpt-150000"
    args.pretrained_model_d =  "results/sf/20220802091932/weights/model_d.ckpt-150000"
    args.gen_frm_dir = "results/output/20220802091932/"
    args.data_train_path = "data/sf/seq_02_15_1003_test.txt"#"data/sf/2014_predict_test.txt"
    args.image_path = "Z:\kjzhongxin\stage1_hainan\output-Img"#"Z:\\kjzhongxin\\一期海南站数据整理\\output-Img\\2014"#"E:/kjzhongxin/stage2/ConvLSTM2/data/2014"
    args.batch_size = 1
    to_analysis = False
    to_visual = True
    spilt_sequence = False

    #初始化模型
    model = Model(args)
    model.load(args.pretrained_model_pm, args.pretrained_model_d)

    if not os.path.exists(args.gen_frm_dir):
        os.makedirs(args.gen_frm_dir)

    # 读取输出文件列表
    sequences = ReadDataFile(args.data_train_path)
    if spilt_sequence:
        train_len = int(len(sequences) * 0.95)
        valid_len = len(sequences) - train_len
        _, valid_sequences = random_split(sequences, (train_len, valid_len))
    else:
        valid_sequences = sequences

    itr = 1
    output_path = args.gen_frm_dir
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for sequence in valid_sequences:
        print("predict: {}/{}".format(itr, len(valid_sequences)))
        input_imgs = ReadSequence(sequence,
                                  args.img_height,args.img_width,args.img_channel,args.image_path,args.patch_size,
                                  transform=transforms.Compose([Norm(), ToTensor()]))

        input_imgs =  torch.unsqueeze(input_imgs, 0) #扩展一维为batch_size
        #print(input_imgs.shape)

        #预测，输出结果
        if to_analysis:
            output_path = args.gen_frm_dir + '/' + str(itr)
            if not os.path.exists(output_path):
                os.mkdir(output_path)

        test(model, output_path,sequence, input_imgs,args, itr,to_analysis = to_analysis, to_visual = to_visual)

        itr = itr + 1



