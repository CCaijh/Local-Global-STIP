import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import STIP
from core.models.Discriminator import Discriminator
import torch.optim.lr_scheduler as lr_scheduler
import logging
from core.utils import preprocess

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.patch_height = configs.img_height // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_channel = configs.img_channel * (configs.patch_size ** 2)
        self.num_layers = configs.num_layers
        networks_map = {
            'stip': STIP.RNN
        }
        num_hidden = []
        for i in range(configs.num_layers):
            num_hidden.append(configs.num_hidden)
        self.num_hidden = num_hidden
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        # print("Network state:")
        # for param_tensor in self.network.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        #     print(param_tensor, '\t', self.network.state_dict()[param_tensor].size())
        if configs.dataset == 'sjtu4k':
            from core.models.Discriminator_4k import Discriminator
        else:
            from core.models.DiscriminatorV2 import DiscriminatorContext

        self.Discriminator = DiscriminatorContext(self.patch_height, self.patch_width, self.patch_channel,self.configs.D_num_hidden).to(self.configs.device)
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.optimizer_D = Adam(self.Discriminator.parameters(), lr=configs.lr_d)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=configs.lr_decay)
        self.scheduler_D = lr_scheduler.ExponentialLR(self.optimizer_D, gamma=configs.lr_decay)
        self.MSE_criterion = nn.MSELoss()
        self.D_criterion = nn.BCELoss()
        self.L1_loss = nn.L1Loss()

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model_pm.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("save predictive model to %s" % checkpoint_path)

        stats = {}
        stats['net_param'] = self.Discriminator.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model_d.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("save discriminator model to %s" % checkpoint_path)

    def load(self, pm_checkpoint_path, d_checkpoint_path):
        print('load predictive model:', pm_checkpoint_path)
        stats = torch.load(pm_checkpoint_path, map_location=torch.device(self.configs.device))
        self.network.load_state_dict(stats['net_param'])

        print('load discriminator model:', d_checkpoint_path)
        stats = torch.load(d_checkpoint_path, map_location=torch.device(self.configs.device))
        self.Discriminator.load_state_dict(stats['net_param'])

    # frames shape : 1,9,12,256,256 : batch, sequence number, channel number , height , width
    def clip_local_area(self, frames):
        temp = frames.transpose(0, 1, 3, 4, 2)
        img_gen = preprocess.reshape_patch_back(temp, self.configs.patch_size)

        #截取一部分
        img_gen = img_gen[:, :, 256:448, 0:512]
        return torch.Tensor(img_gen).cuda()

    #代码与STRPM相同
    def train(self, frames, mask, itr):
        logger = logging.getLogger("stip.train")

        # print(frames.shape)
        self.network.train()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)

        next_frames, _ = self.network(frames_tensor, mask_tensor)  #1,9,12,256,256
        ground_truth = frames_tensor[:, 1:]  #1,9,12,256,256

        batch_size = next_frames.shape[0]
        zeros_label = torch.zeros(batch_size).cuda()
        ones_label = torch.ones(batch_size).cuda()

        # train D
        self.Discriminator.zero_grad()
        d_gen, _, _ = self.Discriminator(next_frames.detach())
        d_gt, _, _ = self.Discriminator(ground_truth)
        D_loss = self.D_criterion(d_gen, zeros_label) + self.D_criterion(d_gt, ones_label)
        D_loss.backward(retain_graph=True)
        self.optimizer_D.step()

        self.optimizer.zero_grad()
        d_gen_pre, features_gd_gen,features_ld_gen = self.Discriminator(next_frames) # features_gen shape[1,9,64]
        _, features_gd_gt,features_ld_gt = self.Discriminator(ground_truth)
        loss_l1 = self.L1_loss(next_frames, ground_truth)

        #截取其中一段
        ld_next_frames = self.clip_local_area(next_frames.detach().cpu().numpy())
        ld_ground_truth = self.clip_local_area(ground_truth.detach().cpu().numpy())
        loss_l2_gd = self.MSE_criterion(next_frames, ground_truth)
        loss_l2_local = self.MSE_criterion(ld_next_frames, ld_ground_truth)

        gen_D_loss = self.D_criterion(d_gen_pre, ones_label)
        loss_gd_features = self.MSE_criterion(features_gd_gen, features_gd_gt)
        loss_ld_features = self.MSE_criterion(features_ld_gen, features_ld_gt)
        loss_gen = self.configs.gd_loss_percent * loss_l2_gd \
                   + self.configs.ld_loss_percent * loss_l2_local \
                   + self.configs.gd_feature_loss_percent * loss_gd_features \
                   + self.configs.ld_feature_loss_percent * loss_ld_features \
                   + 0.001 * gen_D_loss  #第一次为0.2:0.8,0.001：0.009
        loss_gen.backward()
        self.optimizer.step()
        if itr >= self.configs.sampling_stop_iter and itr % self.configs.delay_interval == 0:
            self.scheduler.step()
            self.scheduler_D.step()
            logger.info('Lr decay to:{:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        return next_frames, loss_l1.detach().cpu().numpy(), loss_l2_local.detach().cpu().numpy(), loss_l2_gd.detach().cpu().numpy(), D_loss.detach().cpu().numpy(), \
               gen_D_loss.detach().cpu().numpy(), loss_ld_features.detach().cpu().numpy(), loss_gd_features.detach().cpu().numpy(), d_gt.mean().detach().cpu().numpy(), d_gen.mean().detach().cpu().numpy(), d_gen_pre.mean().detach().cpu().numpy(),\

    def test(self, frames, mask):
        self.network.eval()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames, features = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy(), features.detach().cpu().numpy()
