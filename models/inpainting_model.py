from .base_model import BaseModel
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.loss import GANLoss, MultiscaleL1Loss, MaskedL1Loss, center_loss
import models.networks as networks
from collections import OrderedDict


class InpaintingModel(BaseModel):
    def __init__(self, opt):
        super(InpaintingModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained model
        self.netG = networks.define_G(opt).to(self.device)
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            self.netG.train()
            self.netD.train()

        self.load()  # load G and D

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == 'ml1':
                    self.cri_pix = MultiscaleL1Loss().to(self.device)
                else:
                    raise NotImplementedError('Unsupported loss type: {}'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)

                else:
                    raise NotImplementedError('Unsupported loss type: {}'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
                self.guided_cri_fea = MaskedL1Loss().to(self.device)
            else:
                self.cri_fea = None
            if self.cri_fea:  # load VGG model
                # self.vgg = Vgg19()
                # self.vgg.load_state_dict(torch.load(vgg_model))
                # for param in self.vgg.parameters():
                #     param.requires_grad = False
                self.vgg = networks.define_F(opt)
                self.vgg.to(self.device)
                self.vgg_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
                self.vgg_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
                self.vgg_fns = [self.cri_fea] * len(self.vgg_layers)

            ## discriminator features
            if train_opt['dis_feature_weight'] > 0:
                l_dis_fea_type = train_opt['dis_feature_criterion']
                if l_dis_fea_type == 'l1':
                    self.cri_dis_fea = nn.L1Loss().to(self.device)
                elif l_dis_fea_type == 'l2':
                    self.cri_dis_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Unsupported loss type: {}'.format(l_dis_fea_type))
                self.l_dis_fea_w = train_opt['dis_feature_weight']
            else:
                self.cri_dis_fea = None
            if self.cri_dis_fea:
                self.dis_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
                self.dis_fns = [self.cri_dis_fea] * len(self.dis_weights)

            ## center loss weight
            if train_opt['center_weight'] > 0:
                self.l_center_w = train_opt['center_weight']
            else:
                self.l_center_w = 0

            # G & D gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']

            # optimizers
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_policy'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer,
                                                                    train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('Unsupported learning scheme: {}'.format(train_opt['lr_policy']))

            self.log_dict = OrderedDict()
            # print network
            self.print_network()

    def feed_data(self, data):
        self.var_input = data['input'].to(self.device)
        self.var_mask = data['mask'].to(self.device)
        self.var_bbox = data['bbox'].to(self.device)
        self.var_target = data['target'].to(self.device)

    def optimize_parameters(self):

        # update G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.output = self.var_mask.detach() * self.netG(torch.cat([self.var_input, self.var_mask], dim=1)) + (
                1 - self.var_mask.detach()) * self.var_input.detach()
        l_g_total = 0
        if self.cri_pix:  # pixel loss
            l_g_pix = self.l_pix_w * self.cri_pix(self.output, self.var_target)
            l_g_total += l_g_pix
        if self.cri_fea:  # vgg feature matching loss
            target_feas = [fea.detach() for fea in self.vgg(self.var_target)]
            output_feas = self.vgg(self.output)

            # self-guided regression loss
            error_map = torch.mean(torch.pow(torch.abs(self.output.detach() - self.var_target), 2.0), dim=1, keepdim=True)
            error_map_max_w, _ = torch.max(error_map, dim=3, keepdim=True)
            error_map_max, _ = torch.max(error_map_max_w, dim=2, keepdim=True)
            error_map_normalized = error_map / error_map_max
            vgg_losses = []
            vgg_losses.append(self.vgg_weights[0] * self.guided_cri_fea(output_feas[0], target_feas[0], 1.0 + error_map_normalized))
            error_map_normalized_downsample = F.avg_pool2d(error_map_normalized, kernel_size=2, stride=2)
            vgg_losses.append(self.vgg_weights[1] * self.guided_cri_fea(output_feas[1], target_feas[1], 1.0 + error_map_normalized_downsample))

            for i, fea in enumerate(output_feas[2:], start=2):
                vgg_losses.append(self.vgg_weights[i] * self.vgg_fns[i](fea, target_feas[i]))

            l_g_fea = self.l_fea_w * sum(vgg_losses)
            l_g_total += l_g_fea

            if self.l_center_w > 0:
                l_g_center = self.l_center_w * center_loss(output_feas[3], target_feas[3])
                l_g_total += l_g_center

        pred_g_fake, dis_feas_fake = self.netD(self.crop_patch(self.output, self.var_bbox), self.output)
        pred_g_real, dis_feas_real = self.netD(self.crop_patch(self.var_target, self.var_bbox), self.var_target)
        pred_g_real.detach_()

        l_g_gan = self.l_gan_w * (self.cri_gan(pred_g_real - torch.mean(pred_g_fake), False) +
                                  self.cri_gan(pred_g_fake - torch.mean(pred_g_real), True)) / 2
        l_g_total += l_g_gan

        if self.cri_dis_fea:  # discriminator feature matching loss
            target_dis_feas = [dis_fea.detach() for dis_fea in dis_feas_real]
            dis_feas_losses = []
            for i, dis_fea in enumerate(dis_feas_fake):
                dis_feas_losses.append(self.dis_weights[i] * self.dis_fns[i](dis_fea, target_dis_feas[i]))
            l_g_dis_fea = self.l_dis_fea_w * sum(dis_feas_losses)
            l_g_total += l_g_dis_fea

        l_g_total.backward()
        self.optimizer_G.step()

        # update D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real, _ = self.netD(self.crop_patch(self.var_target, self.var_bbox), self.var_target)
        pred_d_fake, _ = self.netD(self.crop_patch(self.output, self.var_bbox).detach(), self.output.detach())
        l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
        l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
        l_d_gan = (l_d_fake + l_d_real) / 2
        l_d_total += l_d_gan

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        # G
        if self.cri_pix:
            self.log_dict['l_g_pix'] = l_g_pix.item()
        if self.cri_fea:
            self.log_dict['l_g_fea'] = l_g_fea.item()
        self.log_dict['l_g_gan'] = l_g_gan.item()
        if self.cri_dis_fea:
            self.log_dict['l_g_dis_fea'] = l_g_dis_fea.item()

        # D
        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()

        # D outputs
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def crop_patch(self, tensor, bbox):
        bbox_np = bbox.cpu().numpy().astype(int)
        b, c, h, w = tensor.shape

        out = torch.empty((b, c, h // 2, w // 2), device=self.device)
        for i in range(bbox.shape[0]):
            out[i] = tensor[i, :, bbox_np[i][0]:bbox_np[i][0] + bbox_np[i][2],
                     bbox_np[i][1]:bbox_np[i][1] + bbox_np[i][3]]
        return out

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.output = self.var_mask.detach() * self.netG(torch.cat([self.var_input, self.var_mask], dim=1)) + (
                    1 - self.var_mask.detach()) * self.var_input.detach()
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['input'] = self.var_input.detach()[0].float().cpu()
        out_dict['output'] = self.output.detach()[0].float().cpu()
        out_dict['target'] = self.var_target.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        print('Network G with parameters: {:,d}'.format(n))
        print(s)

        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            print('Discriminator D with parameters: {:,d}'.format(n))
            print(s)

        if self.cri_fea:
            s, n = self.get_network_description(self.vgg)
            print('Vgg19 with parameters: {:,d}'.format(n))
            print(s)

    def load(self):
        load_path_G = self.opt['pretrained_model_G']
        if load_path_G is not '':
            print('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        load_path_D = self.opt["pretrained_model_D"]
        if load_path_D is not '':
            print('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
