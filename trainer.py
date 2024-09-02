import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn as nn

import math
import os
import sys
import time
import numpy as np

from utils import angular_error, AverageMeter
from logger import create_logger

from models import create_model
from losses import create_loss

from datasets.combined_data import create_combined_loader, combine_zip_data
from datasets.one_data import create_one_loader
from optim import create_optimizer
from scheduler import create_scheduler

import shutil

class Trainer(object):
    def __init__(self, config, output_dir):

        self.config = config
        self.train_test = config.train_test
        self.stop_thresh = config.stop_thresh
        self.output_dir = output_dir
        self.save_step = config.save_step
        self.gpu_id = config.gpu_id
        self.logger = create_logger(name='GAZE',
                            output_dir=self.output_dir,
                            filename='log.txt')

        # configure tensorboard logging
        log_dir = os.path.join(output_dir, 'log')
        self.writer = SummaryWriter(log_dir=log_dir)

        self.test_eth = config.test_eth
        self.test_tag = config.test_tag
        self.terr_path = os.path.join(config.base_dir, 'errors.txt')

        self.logger.info("%s:%s" % (config.base_dir, self.test_tag))
        
        self.err_path = os.path.join(output_dir, 'error.txt')

        data_dir = config.gaze_data
        input_size = config.input_size
        batch_size = config.batch_size
        data_type = config.data_type
        test_ids = config.test_ids
        self.data_type = config.data_type

        self.validation = config.validation

        self.companion = config.companion
        self.valid_loader = None
        self.valid_err = 100
        if config.companion:
            self.dataloader = create_combined_loader(data_dir, input_size, batch_size, data_type, test_ids)
            self.logger.info("Companion sets Training")
            self.valid_loader = create_one_loader(data_dir, input_size, batch_size, "eth", [], False)
            self.logger.info("Validation dataset size: %d" % (len(self.valid_loader.dataset)))
            self.valid_err = 100
        else:
            self.dataloader = create_one_loader(data_dir, input_size, batch_size, data_type, test_ids, True)
            self.logger.info("Single set Training")

        self.num_batches = len(self.dataloader)
        self.num_samples = len(self.dataloader.dataset)

        self.batch_size = config.batch_size
        self.total_iter = self.num_batches * config.epochs

        batch_size = config.test_batch_size
        self.test_loader = create_one_loader(data_dir, input_size, batch_size, data_type, test_ids, False, eth_test=self.test_eth)
        self.num_test = len(self.test_loader.dataset)
        self.test_model = config.test_model
        self.test_label = config.test_label
        self.test_err = 100
        

        # training params
        self.start_epoch = 0
        self.epochs = config.epochs  # the total epoch to train
        self.iter_size = config.iter_size
        self.train_iter = self.start_epoch * self.num_batches

        # configure tensorboard logging
        self.model = create_model(config)
        self.with_la = config.with_la
        if config.with_la:
            self.logger.info("Create model with LA")
        else:
            self.logger.info("Create model without LA")

        if len(config.finetune) > 0:
            resume_path = config.finetune
            info_str = 'Finetune from: %s' % resume_path
            self.logger.info(info_str)
            checkpoint = torch.load(resume_path, map_location='cpu')['model_state']
            self.model.load_state_dict(checkpoint, strict=False)

        self.loss_function = create_loss(config)

        if self.gpu_id >= 0:
            self.model.cuda(self.gpu_id)
            self.loss_function.cuda(self.gpu_id)

        self.lr = config.base_lr
        self.optimizer = create_optimizer(config, self.model)
        self.scheduler = create_scheduler(config, self.optimizer)


    def train(self):
        info_str = "Train on {} samples".format(self.num_samples)
        self.logger.info(info_str)
        # train for each epoch

        err, valid_err = self.test(True, 0)
        self.save_checkpoint(0)

        for epoch in range(self.start_epoch, self.epochs):
            info_str = 'Epoch: {}/{} - base LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.lr)
            self.logger.info(info_str)

            for param_group in self.optimizer.param_groups:
                info_str = 'Learning rate: %.6f' % param_group['lr']
                self.logger.info(info_str)

            # train for 1 epoch
            if self.companion:
                _, one_avg = self.train_one_epoch(epoch)
            else:
                one_avg = self.train_one_epoch_single(epoch)

            # save the model for each epoch
            if (epoch+1) % self.save_step == 0 or (epoch+1)==self.epochs:
                self.save_checkpoint(epoch+1)
            if self.train_test:
                err, valid_err = self.test(True, epoch+1)

                if valid_err < self.valid_err:
                    self.valid_err = valid_err
                    self.test_err = err
                    self.save_checkpoint(0)
                        
            self.scheduler.step()  # update learning rate
        
        self.writer.close()
        if not self.validation:
            self.save_checkpoint(0)
        self.test()

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.
        """
        self.model.train()
        batch_time = AverageMeter()
        eth_errors = AverageMeter()
        one_errors = AverageMeter()
        losses_gaze = AverageMeter()
        losses = AverageMeter()

        tic_t = time.time()
        tic = time.time()

        iter_count = 0
        self.optimizer.zero_grad()
        for i, data in enumerate(self.dataloader):
           
            data_cmb = combine_zip_data(data)

            if self.gpu_id >= 0:
                data_cmb = {k: v.cuda(self.gpu_id) for k, v in data_cmb.items()}

            # train gaze net
            pred_gaze = self.model(data_cmb)

            target = data_cmb['gaze']
            target_var = target.float()
                
            loss_gaze = self.loss_function(pred_gaze, target_var)
            loss = loss_gaze

            loss = loss / self.iter_size
            loss.backward()
            iter_count += 1

            if iter_count == self.iter_size or (i+1) == len(self.dataloader):
                self.optimizer.step()
                self.optimizer.zero_grad()
                iter_count = 0

            errors = angular_error(pred_gaze.detach().cpu().data.numpy(), target_var.detach().cpu().data.numpy())
            gaze_error = np.mean(errors[:self.batch_size//2])
            one_errors.update(gaze_error.item(), self.batch_size//2)

            gaze_error = np.mean(errors[self.batch_size//2:])
            eth_errors.update(gaze_error.item(), self.batch_size//2)

            losses_gaze.update(loss_gaze.item(), target_var.size()[0])
            losses.update(loss.item() * self.iter_size, target_var.size()[0])

            self.train_iter = self.train_iter + 1
            
            if (i+1) % 1000 == 0:
                self.writer.add_scalar('Loss/gaze', losses_gaze.avg, self.train_iter)
                self.writer.add_scalar('Loss/total', losses.avg, self.train_iter)
                self.writer.add_scalar('Error/train', one_errors.avg, self.train_iter)

            info_step = len(self.dataloader) // 4
            # info_step = 100
            # report information
            if (i+1) % info_step == 0:
                # INFO
                est_time = np.round((self.total_iter - self.train_iter) * batch_time.avg / 60.0)
                info_str = 'Epoch %02d/%02d iteration %d/%d LR %.6f TLeft %.1f mins' % (epoch+1, self.epochs, 
                    self.train_iter, self.total_iter, self.optimizer.param_groups[0]['lr'], est_time)
                self.logger.info(info_str)

                # Error Loss
                msg = "train {} error: {:.3f} - eth error: {:.3f}, gaze loss: {:.5f}"
                info_str = msg.format(self.data_type, one_errors.avg, eth_errors.avg, losses_gaze.avg)  
                self.logger.info(info_str)
                

            # batch time avg
            toc = time.time()
            batch_time.update(toc - tic)
            tic = time.time()

        toc_t = time.time()

        info_str = 'running time is %.1f mins' % ((toc_t - tic_t)/ 60.0)
        self.logger.info(info_str)

        return one_errors.avg, eth_errors.avg

    def train_one_epoch_single(self, epoch):
        """
        Train the model for 1 epoch of the training set.
        """
        self.model.train()
        batch_time = AverageMeter()
        eth_errors = AverageMeter()
        one_errors = AverageMeter()
        losses_gaze = AverageMeter()
        losses = AverageMeter()

        tic_t = time.time()
        tic = time.time()

        iter_count = 0
        self.optimizer.zero_grad()
        for i, data in enumerate(self.dataloader):
            
            # data_cmb = combine_zip_data(data)
            data_cmb = data

            if self.gpu_id >= 0:
                data_cmb = {k: v.cuda(self.gpu_id) for k, v in data_cmb.items()}

            # train gaze net
            pred_gaze = self.model(data_cmb)

            target = data_cmb['gaze']
            target_var = target.float()
                
            loss_gaze = self.loss_function(pred_gaze, target_var)
            loss = loss_gaze

            loss = loss / self.iter_size
            loss.backward()
            iter_count += 1

            if iter_count == self.iter_size or (i+1) == len(self.dataloader):
                self.optimizer.step()
                self.optimizer.zero_grad()
                iter_count = 0

            data_size = target_var.size()[0]
            errors = angular_error(pred_gaze.detach().cpu().data.numpy(), target_var.detach().cpu().data.numpy())
            gaze_error = np.mean(errors)
            one_errors.update(gaze_error.item(), data_size)

            losses_gaze.update(loss_gaze.item(), data_size)
            losses.update(loss.item() * self.iter_size, data_size)

            self.train_iter = self.train_iter + 1
            
            if (i+1) % 1000 == 0:
                self.writer.add_scalar('Loss/gaze', losses_gaze.avg, self.train_iter)
                self.writer.add_scalar('Loss/total', losses.avg, self.train_iter)
                self.writer.add_scalar('Error/train', one_errors.avg, self.train_iter)

            info_step = len(self.dataloader) // 4
            # info_step = 100
            # report information
            if (i+1) % info_step == 0:
                # INFO
                est_time = np.round((self.total_iter - self.train_iter) * batch_time.avg / 60.0)
                info_str = 'Epoch %02d/%02d iteration %d/%d LR %.6f TLeft %.1f mins' % (epoch+1, self.epochs, 
                    self.train_iter, self.total_iter, self.optimizer.param_groups[0]['lr'], est_time)
                self.logger.info(info_str)

                # Error Loss
                msg = "train {} error: {:.3f}, gaze loss: {:.5f}"
                info_str = msg.format(self.data_type, one_errors.avg, losses_gaze.avg)  
                self.logger.info(info_str)

            # batch time avg
            toc = time.time()
            batch_time.update(toc - tic)
            tic = time.time()

        toc_t = time.time()

        info_str = 'running time is %.1f mins' % ((toc_t - tic_t)/ 60.0)
        self.logger.info(info_str)

        return one_errors.avg

    
    def test_data(self, data_loader):
        self.model.eval()
        num_valid = len(data_loader.dataset)
        pred_gaze_all = np.zeros((num_valid, 2))
        gt_gaze_all = np.zeros((num_valid, 2))
        save_index = 0
        for i, data in enumerate(data_loader):
            if self.gpu_id >= 0:
                data = {k: v.cuda() for k, v in data.items()}
            input_img = data['image']
            batch_size = input_img.size(0)
            pred_gaze = self.model(data)
            
            pred_gaze_all[save_index:save_index+batch_size, :] = pred_gaze.cpu().data.numpy()

            if (self.test_label or self.train_test) and (not self.test_eth):
                gaze = data['gaze'].cpu()
                gt_gaze_all[save_index:save_index+batch_size, :] = gaze.numpy()
                
            save_index += batch_size

        pred_gaze_all = pred_gaze_all[:save_index]
        gt_gaze_all = gt_gaze_all[:save_index]
        return pred_gaze_all, gt_gaze_all

    @torch.no_grad()
    def test(self, train_test=False, epoch=0):
        """
        Test the pre-treained model on the whole test set. Note there is no label released to public, you can
        only save the predicted results. You then need to submit the test resutls to our evaluation website to
        get the final gaze estimation error.
        """
        self.model.eval()

        if not train_test:
            if self.test_model == '':
                model_path = os.path.join(self.output_dir, 'best_ckpt.pth.tar')
            else:
                model_path = self.test_model
                self.output_dir = os.path.split(model_path)[0]
            self.load_checkpoint(is_strict=False, input_file_path=model_path)

        # if self.with_la:
        #     # print(self.model.convertor.A.detach().cpu().numpy().reshape((-1)))
        #     print(self.model.convertor.D.detach().cpu().numpy().reshape((-1)))

        num_test = len(self.test_loader.dataset)
        self.logger.info('Testing on %s %d samples' % (self.data_type, num_test))
        pred_gaze_all, gt_gaze_all = self.test_data(self.test_loader)
        
        res_path = os.path.join(self.output_dir, 'within_eva_results.txt')
        np.savetxt(res_path, pred_gaze_all, delimiter=',')
        self.logger.info('Result saved in %s' % res_path)


        if self.test_eth:
            return None
            
        err = angular_error(pred_gaze_all, gt_gaze_all).mean()

        # validation set
        valid_err = err
        if self.valid_loader is not None and train_test:
            num_valid = len(self.valid_loader.dataset)
            self.logger.info('Validate on XGaze %d samples' % num_valid)
            pred_gaze_all, gt_gaze_all = self.test_data(self.valid_loader)
            valid_err = angular_error(pred_gaze_all, gt_gaze_all).mean()

        if train_test:
            with open(self.err_path, 'a+') as f:
                if self.valid_loader is not None:
                    err_line = '%02d,%.4f,%.4f' % (epoch, err, valid_err)
                else:
                    err_line = '%02d,%.4f' % (epoch, err)
                f.write(err_line+'\n')
                self.logger.info('Gaze error: %s saved in %s' %(err_line, self.err_path))
        else:
            with open(self.terr_path, 'a+') as f:
                err_line = '%s,%.4f' % (self.test_tag, err)
                f.write(err_line+'\n')
                self.logger.info('Gaze error: %s saved in %s' %(err_line, self.terr_path))

        return err, valid_err


    def save_checkpoint(self, epoch):
        """
        Save a copy of the model
        """
        if epoch > 0:
            file_name = 'epoch_' + str(epoch)+ '_ckpt.pth.tar'
        else:
            file_name = 'best_ckpt.pth.tar'
        state = {'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    # 'optim_state': self.optimizer.state_dict(),
                    # 'scheule_state': self.scheduler.state_dict()
                }
          
        ckpt_path = os.path.join(self.output_dir, file_name)
        torch.save(state, ckpt_path)

        info_str = 'Save file to: %s' % ckpt_path
        self.logger.info(info_str)

    def load_checkpoint(self, input_file_path='./ckpt/ckpt.pth.tar', is_strict=True):
        """
        Load the copy of a model.
        """
        self.logger.info('load the pre-trained model: %s' % input_file_path)
        ckpt = torch.load(input_file_path)

        # load variables from checkpoint
        self.model.load_state_dict(ckpt['model_state'], strict=is_strict)
        self.start_epoch = ckpt['epoch'] - 1

        