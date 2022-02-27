# -*- coding: utf-8 -*-
import math
import datetime
import losses
import os
from tqdm import tqdm
from utils.dice_loss import *
import torch.nn.functional as F
import numpy as np
import imageio
from skimage import io

from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch


writer = SummaryWriter()
running_loss_final = 0
running_loss_sub = 0
runnng_final = 0
def soft_dice(pred, target):
       disc_dice = 0
       cup_dice = 0
       
       for threshold in [0.1,0.3,0.5,0.7,0.9]:
           final_mask_temp = (target >= threshold).float()
           pred_t = (pred >= threshold).float()
           pred_t_n = pred_t.cpu()
           disc_pred = pred_t_n[:,0,:,:].numpy()
           cup_pred = pred_t_n[:,1,:,:].numpy()

           disc_pred = disc_pred.astype('int32')
           cup_pred = cup_pred.astype('int32')
           
           '''dice for torch'''
           disc_dice += dice_coeff(pred_t[:,0,:,:], final_mask_temp[:,0,:,:]).item()
           cup_dice += dice_coeff(pred_t[:,1,:,:], final_mask_temp[:,1,:,:]).item()

       return disc_dice, cup_dice


class Trainer(object):

    def __init__(self, cuda, model_rgb, model_six,model_att,optimizer_rgb,optimizer_six,optimizer_att,
                 train_sub,val_sub,train_loader, val_loader,test_loader,test_sub,max_iter,
                 snapshot, outpath, sshow, step_size, gamma,log_name,val_out_path,size_average=False):
        self.cuda = cuda
        self.model_rgb = model_rgb
        self.model_six = model_six
        self.model_att = model_att
        self.optim_rgb = optimizer_rgb
        self.optim_six = optimizer_six
        self.optim_att = optimizer_att
        self.train_sub = train_sub
        self.val_sub = val_sub
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_sub = test_sub
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.snapshot = snapshot
        self.outpath = outpath
        self.sshow = sshow
        self.step_size = step_size
        self.gamma = gamma
        self.log_name = log_name
        self.val_out_path = val_out_path
        self.size_average = size_average


    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_sub)))
        if self.step_size > 0:
            scheduler=lr_scheduler.StepLR(optimizer=self.optim_rgb, step_size=self.step_size, gamma=self.gamma)

        for epoch in range(max_epoch):
            self.epoch = epoch
            with tqdm(total=len(self.train_sub), desc=f'Epoch {epoch+1}/{max_epoch}',unit='img',leave=True) as pbar:
                self.train_epoch(pbar)
            if self.step_size > 0:
                scheduler.step()
            if self.iteration >= self.max_iter:
                writer.close()
                break



    def train_epoch(self,pbar):
        self.model_rgb.train()
        self.model_six.train()
        self.model_att.train()

        for batch_idx, data in enumerate(self.train_loader):
            imgs = data['image']
            imgs_o = data['image_ori']
            target = data['mask']

            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration
            if self.iteration >= self.max_iter:
                break
            if self.cuda:
                imgs_o, imgs= imgs_o.cuda(), imgs.cuda()
                target = [x.cuda() for x in target]
            imgs_o = Variable(imgs_o)                       # [Batch_size, C, H, W]  or  [B, 3, 256, 256]
            imgs = Variable(imgs).to(dtype=torch.float32)   # [Batch_size, C, H, W]  or  [B, 3, 256, 256]
            target = [Variable(x) for x in target]          # [a1,a2,...,a6], a1=[Batch_size,C,H,W] or [b, 2, 256, 256]

            n, c, h, w = imgs.size()

            global running_loss_final
            global running_loss_sub
            global running_final

            criterion = losses.init_loss('BCE_wise').cuda()


            """"""""""" ~~~Your Framework~~~ """""""""
            gaussian_random = np.random.normal(loc=0.0, scale=0.1, size=(1, 1, h, w))
            gaussian_random = np.repeat(gaussian_random, c, axis=1)
            gaussian_random = np.repeat(gaussian_random, n, axis=0)
            gaussian_random = torch.tensor(gaussian_random).cuda()
            imgs_g = imgs + gaussian_random
            imgs_g = imgs_g.type(torch.float32)
            # imgs_1 = np.array(imgs[0][1].cpu())
            # imgs_2 = np.array(imgs_g[0][1].cpu())
            # imageio.imwrite('/remote-home/zhaozh/Unet_ResNet/imgs/img1.jpg', imgs_1)
            # imageio.imwrite('/remote-home/zhaozh/Unet_ResNet/imgs/img2.jpg', imgs_2)

            rater_num = 1
            cond_single = torch.zeros(n, 6).to(dtype=torch.float32)
            for i in range(n):
                cond_single[i, rater_num-1] = 1.0
            if self.cuda:
                cond_single = cond_single.cuda()
            cond_ave = torch.tensor([[1/6]]).expand(n,6).to(dtype=torch.float32)
            
            '''final mask'''
            # original six rater masks
            f_mask_list = []
            # condition masks
            final_mask_list_single = []
            final_mask_list_ave = []
            for i, rater_i_mask in enumerate(target):
                rater_i_mask = rater_i_mask.to(dtype=torch.float32)
                final_mask_list_single.append(torch.mul(rater_i_mask, cond_single[:, i].unsqueeze(-1).
                                                 unsqueeze(-1).unsqueeze(-1).expand(-1, 2, 256, 256)))
                final_mask_list_ave.append(torch.mul(rater_i_mask, 1/6))
                f_mask_list.append(rater_i_mask)
            final_mask_single = sum(final_mask_list_single)
            final_mask_ave = sum(final_mask_list_ave)

            # outputs_single, f_single = self.model_rgb(imgs, cond_single)
            outputs_ave, f_ave = self.model_rgb(imgs, cond_ave)
            outputs_g, f_g = self.model_rgb(imgs_g, cond_ave)
            fusion_rate = 0.7
            outputs_fusion = (torch.sign(outputs_ave) * torch.pow(torch.abs(outputs_ave), fusion_rate)) * \
                (torch.sign(outputs_g) * torch.pow(torch.abs(outputs_g), (1 - fusion_rate)))

            # loss_single = criterion(outputs_single, final_mask_single)
            loss_ave = criterion(outputs_ave, final_mask_ave)

            dices = []
            for i in range(len(outputs_g)):
                disc_dice = dice_coeff(outputs_fusion[i,0,:,:], final_mask_ave[i,0,:,:])
                cup_dice = dice_coeff(outputs_fusion[i,1,:,:], final_mask_ave[i,1,:,:])
                dices.append((disc_dice + cup_dice).reshape((1,)))
            dices = torch.cat(dices, dim=0)
            dices = torch.softmax(dices, dim=0)

            loss_all = torch.sum(loss_ave * dices)
            running_loss_final += loss_all.item()

            self.optim_rgb.zero_grad()
            loss_all.backward()
            self.optim_rgb.step()
            """"""""""""""""""""""""""""""""""""""""""

            writer.add_scalar('Loss/train_main', loss_all.item(), iteration)
            pbar.set_postfix(**{'loss (batch)': loss_all.item()})

            """"""""""" ~~~record and report~~~ """""""""
            # record
            if iteration % self.sshow == (self.sshow - 1):
                curr_time = str(datetime.datetime.now())[:19]
                print('\n [%s,%3d,%6d, The training loss of sub Net:%.3f]'%(
                curr_time, self.epoch + 1, iteration + 1,
                running_loss_final / (n * self.sshow)))
                running_loss_final = 0.0
                running_loss_sub = 0.0
                running_final = 0.0


            # report
            pbar.update(n)

            if iteration % (len(self.train_sub)+len(self.val_sub)) // (2 * n) == 0:
                for tag, value in self.model_rgb.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' +tag, value.data.cpu().numpy(), iteration)
                    writer.add_histogram('grads/' + tag, value.data.cpu().numpy(), iteration)
                    writer.add_scalar('learning_rate', self.optim_rgb.param_groups[0]['lr'], iteration)

                out = torch.sigmoid(outputs_ave)
                writer.add_images('images_ori', imgs_o, iteration)
                writer.add_images('images', imgs, iteration)
                writer.add_images('masks/true_cup', final_mask_single[:,1,:,:].unsqueeze(1), iteration)
                writer.add_images('masks/pred_cup', out[:,1,:,:].unsqueeze(1), iteration)
                writer.add_images('masks/true_disc', final_mask_single[:,0,:,:].unsqueeze(1), iteration)
                writer.add_images('masks/pred_disc', out[:,0,:,:].unsqueeze(1), iteration)

                mask_cup_list = []
                mask_disc_list = []
                
                for i in range(len(f_mask_list)):
                    mask_disc_list.append(f_mask_list[i][:, 0, :, :].unsqueeze(1))
                    mask_cup_list.append(f_mask_list[i][:, 1, :, :].unsqueeze(1))
                mask_cup = torch.cat(mask_cup_list, dim=1)
                mask_disc = torch.cat(mask_disc_list, dim=1)
                u_cup = torch.std(mask_cup, dim=1).unsqueeze(1)
                u_disc = torch.std(mask_disc, dim=1).unsqueeze(1)
                writer.add_images('attention/mask_u_cup', u_cup, iteration)
                writer.add_images('attention/mask_u_disc', u_disc, iteration)

        self.val_epoch(self.epoch,val_flag=True)

    def val_epoch(self, epoch, val_flag=True):
        print("Preparing for validation.")
        self.model_rgb.eval()
        self.model_six.eval()
        self.model_att.eval()

        mask_type = torch.float32
        if val_flag:
            n_val = len(self.val_sub)  # the number of batch
            data_loader = self.val_loader

            # save checkpoint
            savename1 = ('%s/snapshot_iter_%d.pth' % (self.outpath, epoch + 1))
            torch.save(self.model_rgb.state_dict(), savename1)
        else:
            n_val = len(self.test_sub)
            data_loader = self.test_loader



        with tqdm(total=n_val, desc=f'Model test:', leave=True) as pbar:
            iou_d = 0
            iou_c = 0
            tot = 0
            bn = 0
            disc_dice = 0
            cup_dice = 0
            disc_hard_dice = 0
            cup_hard_dice = 0
            n_all = n_val * 5  ## 5 is threshold

            for batch_idx, data in enumerate(data_loader):
                imgs, target, Name = data['image'],data['mask'],data['name']

                if self.cuda:
                    imgs = imgs.cuda()

                    target = [x.cuda() for x in target]
                imgs = Variable(imgs).to(dtype=mask_type)
                target = [Variable(x) for x in target]
                b_size = imgs.size(0)
                    
                with torch.no_grad():
                    condition = torch.tensor([[1/6]]).expand(b_size,6).to(dtype=torch.float32) # Default Majority
                    # condition = torch.randint(1, 11, (b_size, 6), dtype=torch.float32)

                    # condition = torch.zeros((b_size, 6)).to(dtype=torch.float32)
                    # rater_num = 6
                    # for i in range(b_size):
                    #     condition[i, rater_num-1] = 1.0

                    if self.cuda:
                        condition = condition.cuda()
                    for i in range(0, b_size):
                        condition[i, :] = (condition[i, :] / torch.sum(condition[i, :])).to(dtype=torch.float32)

                    '''final mask'''
                    final_mask_list = []
                    for i, rater_i_mask in enumerate(target):
                        rater_i_mask = rater_i_mask.to(dtype=torch.float32)
                        final_mask_list.append(torch.mul(rater_i_mask, condition[:, i].unsqueeze(-1).
                                                         unsqueeze(-1).unsqueeze(-1).expand(-1, 2, 256, 256)))
                    final_mask = sum(final_mask_list)


                    '''inference'''
                    coarse_pred,fea = self.model_rgb(imgs,condition)
                    pred = torch.sigmoid(coarse_pred)
                    tot += F.binary_cross_entropy_with_logits(coarse_pred, final_mask).item()


                    # hard disc/cup dice
                    a_mask = (final_mask > 0.5).float()
                    a_pred = (pred > 0.5).float()
                    disc_hard_dice += dice_coeff(a_pred[:, 0, :, :], a_mask[:, 0, :, :]).item()
                    cup_hard_dice += dice_coeff(a_pred[:, 1, :, :], a_mask[:, 1, :, :]).item()

                    # soft dice : for threshold in [0.5]:
                    for threshold in [0.1,0.3,0.5,0.7,0.9]:

                        final_mask_temp = (final_mask >= threshold).float()
                        pred_t = (pred >= threshold).float()
                        pred_t_n = pred_t.cpu()
                        disc_pred = pred_t_n[:,0,:,:].numpy()
                        cup_pred = pred_t_n[:,1,:,:].numpy()


                        disc_pred = disc_pred.astype('int32')
                        cup_pred = cup_pred.astype('int32')
                        disc_mask = final_mask_temp[:,0,:,:].squeeze().cpu().numpy().astype('int32')
                        cup_mask = final_mask_temp[:, 1, :, :].squeeze().cpu().numpy().astype('int32')


                        '''iou for numpy'''
                        iou_d += iou(disc_pred,disc_mask)
                        iou_c += iou(cup_pred,cup_mask)

                        '''dice for torch'''
                        disc_dice += dice_coeff(pred_t[:,0,:,:], final_mask_temp[:,0,:,:]).item()
                        cup_dice += dice_coeff(pred_t[:,1,:,:], final_mask_temp[:,1,:,:]).item()


                Name_a = Name[0]
                '''Save Figure'''
                num = pred.shape[0]
                for i in range(0, num):
                    cup_image_np = cup_pred[i, :, :]
                    disc_image_np = disc_pred[i, :, :]

                    disc_path = './Out/results/{0}/'.format(Name_a.split('_')[0])
                    if not os.path.exists(disc_path):
                        os.makedirs(disc_path)

                    io.imsave(disc_path + 'task01.png', np.uint8(disc_image_np * 255))

                    cup_path = './Out/results/{0}/'.format(Name_a.split('_')[0])
                    io.imsave(cup_path + 'task02.png', np.uint8(cup_image_np * 255))

                bn +=1
                pbar.update(1)

            val_score, disc_iou, cup_iou, d_dice, c_dice =  tot / n_val, iou_d / n_all, \
                                                            iou_c / n_all, disc_dice / n_all, cup_dice / n_all
            cup_hard_dice, disc_hard_dice = cup_hard_dice / n_val, disc_hard_dice / n_val



            print('Epoch:', epoch)
            if self.model_rgb.num_classes > 1:
                print('Validation average cross entropy: {}'.format(val_score))
                print('Validation average disc iou: {}'.format(disc_iou))
                print('Validation average cup iou: {}'.format(cup_iou))
                print('Validation average disc dice: {}'.format(d_dice))
                print('Validation average cup dice: {}'.format(c_dice))
                print('Validation hard_0.5 disc dice: {}'.format(disc_hard_dice))
                print('Validation hard_0.5 cup dice: {}'.format(cup_hard_dice))
                if val_flag:
                    writer.add_scalar('Loss_ave/test', val_score, epoch)
                    writer.add_scalar('Val/disc_soft_dice',d_dice, epoch)
                    writer.add_scalar('Val/cup_soft_dice', c_dice, epoch)
                    writer.add_scalar('Val/disc_hard_dice', disc_hard_dice, epoch)
                    writer.add_scalar('Val/cup_hard_dice', cup_hard_dice, epoch)
            print("Finshed on validation.")
