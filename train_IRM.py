import os
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from dataset.dataset import NIH_e1 as data_e1
from dataset.dataset import NIH_e2 as data_e2

import torch
import torch.nn as nn
from torch import autograd
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.backends.cudnn as cudnn
from torchvision import models

from models import *
from utils.eval_utils import compute_metrics, binary_accuracy
from utils.logger_utils import Logger
from config import options

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train():
    global_step = 0
    best_loss = 100
    best_acc = 0

    def penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = criterion(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    for epoch in range(options.epochs):
        log_string('**' * 30)
        log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
        net.train()

        train_loss_e1, train_loss_e2, train_loss = 0, 0, 0
        targets_e1, outputs_e1, targets_e2, outputs_e2 = [], [], [], []
        batch_id = -1

        for (img_e1, target_e1, path_name_e1), (img_e2, target_e2, path_name_e2) in zip(train_loader_e1, train_loader_e2):
            global_step += 1
            batch_id += 1

            img_e1, img_e2 = img_e1.cuda(), img_e2.cuda()
            target_e1, target_e2 = target_e1.cuda(), target_e2.cuda()
            target_e1 = target_e1.view(target_e1.size()[0], -1).float()
            target_e2 = target_e2.view(target_e2.size()[0], -1).float()

            # Forward pass e1
            output_e1 = net(img_e1)
            batch_loss_e1 = criterion(output_e1, target_e1)
            targets_e1 += [target_e1]
            outputs_e1 += [output_e1]
            #train_loss_e1 += batch_loss_e1.item()

            # Forward pass e2
            output_e2 = net(img_e2)
            batch_loss_e2 = criterion(output_e2, target_e2)
            targets_e2 += [target_e2]
            outputs_e2 += [output_e2]
            #train_loss_e2 += batch_loss_e2.item()

            # IRM
            penalty_e1 = penalty(output_e1, target_e1)
            penalty_e2 = penalty(output_e2, target_e2)

            train_nll = torch.stack([batch_loss_e1, batch_loss_e2]).mean()
            train_penalty = torch.stack([penalty_e1, penalty_e2]).mean()
            batch_loss = train_nll.clone()

            penalty_weight = (options.penalty_weight
                              if global_step >= options.penalty_anneal_iters else 1.0)
            batch_loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                batch_loss /= penalty_weight

            train_loss += batch_loss.item()


            # cIRM
            """irm_penalty = 0
            for y in range(2):
                out_e1 = output_e1[target_e1 == y]
                out_e2 = output_e2[target_e2 == y]

                penalty_e1 = penalty(out_e1, target_e1[target_e1 == y])
                penalty_e2 = penalty(out_e2, target_e2[target_e2 == y])

                irm_penalty += (0.5)*torch.stack([penalty_e1, penalty_e2]).mean()

            train_nll = torch.stack([batch_loss_e1, batch_loss_e2]).mean()
            batch_loss = train_nll.clone()

            penalty_weight = (options.penalty_weight
                              if global_step >= options.penalty_anneal_iters else 1.0)
            batch_loss += penalty_weight * irm_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                batch_loss /= penalty_weight

            train_loss += batch_loss.item()"""


            # Backward and optimize
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if (batch_id + 1) % options.disp_freq == 0:
                train_loss /= options.disp_freq
                train_auc_e1 = compute_metrics(torch.cat(outputs_e1), torch.cat(targets_e1))
                train_auc_e2 = compute_metrics(torch.cat(outputs_e2), torch.cat(targets_e2))

                train_acc_e1 = binary_accuracy(torch.cat(outputs_e1), torch.cat(targets_e1))
                train_acc_e2 = binary_accuracy(torch.cat(outputs_e2), torch.cat(targets_e2))

                train_auc = 0.5*(train_auc_e1+train_auc_e2)
                train_acc = 0.5 * (train_acc_e1 + train_acc_e2)
                log_string("epoch: {0}, step: {1}, global step: {2}, train_loss: {3:.4f}, train_acc_e1: {4: .4f}, train_acc_e2: {5: .4f}, train_acc: {6: .4f}, train_auc: {7: .4f},"
                           .format(epoch+1, batch_id+1, global_step, train_loss, train_acc_e1, train_acc_e2, train_acc, train_auc))
                info = {'loss': train_loss,
                        'acc_e1': train_acc_e1,
                        'acc_e2': train_acc_e2,
                        'auc': train_auc}
                for tag, value in info.items():
                    train_logger.scalar_summary(tag, value, global_step)
                train_loss = 0
                targets, outputs = [], []

            if (batch_id + 1) % options.val_freq == 0:
                log_string('--' * 30)
                log_string('Evaluating at step #{}'.format(global_step))
                best_loss, best_acc = evaluate(best_loss=best_loss,
                                               best_acc=best_acc,
                                               global_step=global_step)
                net.train()


def evaluate(**kwargs):
    best_loss = kwargs['best_loss']
    best_acc = kwargs['best_acc']
    global_step = kwargs['global_step']

    net.eval()
    test_loss = 0
    targets, outputs = [], []
    with torch.no_grad():
        for batch_id, (data, target, path_name) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            target = target.view(target.size()[0], -1).float()

            output = net(data)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            test_loss += batch_loss.item()

        test_loss /= (batch_id + 1)
        test_auc = 0 #compute_metrics(torch.cat(outputs), torch.cat(targets))
        test_acc = binary_accuracy(torch.cat(outputs), torch.cat(targets))

        # check for improvement
        loss_str, auc_str = '', ''
        if test_loss <= best_loss:
            loss_str, best_loss = '(improved)', test_loss
        if test_acc >= best_acc:
            auc_str, best_acc = '(improved)', test_acc

            # save checkpoint model
            state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            save_path = os.path.join(model_dir, 'best_model.ckpt')  # .format(global_step))
            torch.save({
                'global_step': global_step,
                'loss': test_loss,
                'acc': test_acc,
                'save_dir': model_dir,
                'state_dict': state_dict},
                save_path)
            log_string('Model saved at: {}'.format(save_path))
        else:
            # save checkpoint model
            state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            save_path = os.path.join(model_dir, '{}'.format(global_step))
            torch.save({
                'global_step': global_step,
                'loss': test_loss,
                'acc': test_acc,
                'save_dir': model_dir,
                'state_dict': state_dict},
                save_path)
            log_string('Model saved at: {}'.format(save_path))

        # display
        log_string("validation_loss: {0:.4f} {1}, validation_acc: {2:.02%}{3}"
                   .format(test_loss, loss_str, test_acc, auc_str))

        # write to TensorBoard
        info = {'loss': test_loss,
                'acc': test_acc}
        for tag, value in info.items():
            test_logger.scalar_summary(tag, value, global_step)


        log_string('--' * 30)
        return best_loss, best_acc


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    model_dir = os.path.join(save_dir, 'models')
    logs_dir = os.path.join(save_dir, 'tf_logs')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # bkp of model def
    os.system('cp {}/models/resnet.py {}'.format(BASE_DIR, save_dir))
    # bkp of train procedure
    os.system('cp {}/train.py {}'.format(BASE_DIR, save_dir))
    os.system('cp {}/config.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    if options.model == 'resnet':
        net = resnet.resnet50(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, options.num_classes)
    elif options.model == 'vgg':
        net = vgg19_bn(pretrained=True, num_classes=options.num_classes)
    elif options.model == 'inception':
        net = inception_v3(pretrained=True)
        net.aux_logits = False
        net.fc = nn.Linear(2048, options.num_classes)
    elif options.model == 'densenet':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        # add final layer with # outputs in same dimension of labels with sigmoid
        # activation
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 14), nn.Sigmoid())

        net = densenet.densenet121(pretrained=True)
        net.classifier = nn.Linear(net.classifier.in_features, out_features=options.num_classes)

    log_string('{} model Generated.'.format(options.model))
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in net.parameters())))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.cuda()
    net = nn.DataParallel(net)
    ##################################
    # Loss and Optimizer
    ##################################
    criterion = nn.BCEWithLogitsLoss() #CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=options.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # Load dataset
    ##################################
    os.system('cp {}/dataset/dataset.py {}'.format(BASE_DIR, save_dir))

    train_dataset_e1 = data_e1(mode='train', data_len=options.data_len)
    train_loader_e1 = DataLoader(train_dataset_e1, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)
    train_dataset_e2 = data_e2(mode='train', data_len=options.data_len)
    train_loader_e2 = DataLoader(train_dataset_e2, batch_size=options.batch_size,
                                 shuffle=True, num_workers=options.workers, drop_last=False)
    test_dataset = data_e1(mode='test', data_len=options.data_len)
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
               format(options.epochs, options.batch_size, len(train_dataset_e1), len(test_dataset)))
    train_logger = Logger(os.path.join(logs_dir, 'train'))
    test_logger = Logger(os.path.join(logs_dir, 'test'))

    train()
