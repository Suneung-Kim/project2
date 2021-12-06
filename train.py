from __future__ import print_function
import os
from PIL import Image
import sys
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from utils import *
from datasets import VehicleDataset
import time

def print_overwrite(step, total_step, loss, acc, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write(f"Train Steps: {step}/{total_step} | Loss: {loss:.4f} | Acc: {acc*100.:.2f} %")   
    else:
        sys.stdout.write(f"Valid Steps: {step}/{total_step} | Loss: {loss:.4f} | Acc: {acc*100.:.2f} %")
    sys.stdout.flush()
    


def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # trainset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform_train)
    trainset = VehicleDataset('./data/train', transform=transform_train, mode='train')

    len_valid_set = int(0.2*len(trainset))
    len_train_set = len(trainset) - len_valid_set
    print("The length of Train set is {}".format(len_train_set))
    print("The length of Valid set is {}".format(len_valid_set))

    train_dataset , valid_dataset,  = torch.utils.data.random_split(trainset , [len_train_set, len_valid_set])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print(len(trainloader), len(validloader))

    # Model
    if resume:
        net = torch.load(model_path)
    else:
        net = load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True)
    netp = torch.nn.DataParallel(net)

    # GPU
    device = torch.device("cuda")
    net.to(device)
    # cudnn.benchmark = True

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},
        {'params': net.features.parameters(), 'lr': 0.0002}

    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    start_time = time.time()
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # update learning rate
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            # Step 1
            optimizer.zero_grad()
            inputs1 = jigsaw_generator(inputs, 8)
            output_1, _, _, _ = netp(inputs1)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()

            # Step 2
            optimizer.zero_grad()
            inputs2 = jigsaw_generator(inputs, 4)
            print(inputs2.shape)
            _, output_2, _, _ = netp(inputs2)
            loss2 = CELoss(output_2, targets) * 1
            loss2.backward()
            optimizer.step()

            # Step 3
            optimizer.zero_grad()
            inputs3 = jigsaw_generator(inputs, 2)
            print(inputs2.shape)
            _, _, output_3, _ = netp(inputs3)
            loss3 = CELoss(output_3, targets) * 1
            loss3.backward()
            optimizer.step()

            # Step 4
            optimizer.zero_grad()
            _, _, _, output_concat = netp(inputs)
            concat_loss = CELoss(output_concat, targets) * 2
            concat_loss.backward()
            optimizer.step()

            #  training log
            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()

            if batch_idx % 50 == 0:
                # print_overwrite(batch_idx, len(trainloader), train_loss, 100. * float(correct) / total, 'train')
                print( 'Training |epoch: %d | Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    epoch+1 ,batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                    train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))
                # print(
                #     'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                #     batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                #     train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                #     100. * float(correct) / total, correct, total))

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                epoch+1, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                train_loss4 / (idx + 1)))

        correct = 0
        total = 0
        idx = 0

        net.eval()
        print('Validataion Result')
        for batch_idx, (inputs, targets) in enumerate(validloader):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            output_1, output_2, output_3, output_concat= net(inputs)

            #  training log
            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % 2== 0:
                print( 'validation, epoch: %d ,Step: %d   Acc: %.3f%% (%d/%d)' % (
                    epoch + 1, batch_idx,
                    100. * float(correct) / total, correct, total))
                # print_overwrite(batch_idx, len(trainloader), train_loss, 100. * float(correct) / total, 'train')
                # print(
                #     'Step: %d   Acc: %.3f%% (%d/%d)' % (
                #     batch_idx,
                #     100. * float(correct) / total, correct, total))

        # train_acc = 100. * float(correct) / total
        # # train_loss = train_loss / (idx + 1)
        # with open(exp_dir + '/results_train.txt', 'a') as file:
        #     file.write(
        #         'Step: %d   Acc: %.3f%% (%d/%d)' % (
        #             batch_idx,
        #             100. * float(correct) / total, correct, total))
        if epoch % 1 == 0:
            # val_acc, val_acc_com, val_loss = test(net, CELoss, 3)
            # if val_acc_com > max_val_acc:
            #     max_val_acc = val_acc_com
                net.cpu()
                torch.save(net, './' + store_name + '/model_{}.pth'.format(epoch))
                net.to(device)
            # with open(exp_dir + '/results_test.txt', 'a') as file:
            #     file.write('Iteration %d, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (
            #     epoch, val_acc, val_acc_com, val_loss))
        # if epoch < 5 or epoch >= 80:
        #     # val_acc, val_acc_com, val_loss = test(net, CELoss, 3)
        #     # if val_acc_com > max_val_acc:
        #     #     max_val_acc = val_acc_com
        #         net.cpu()
        #         torch.save(net, './' + store_name + '/model_{}.pth'.format(epoch))
        #         net.to(device)
        #     # with open(exp_dir + '/results_test.txt', 'a') as file:
        #     #     file.write('Iteration %d, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (
        #     #     epoch, val_acc, val_acc_com, val_loss))
        # else:
        #     net.cpu()
        #     torch.save(net, './' + store_name + '/model_{}.pth'.format(epoch))
        #     net.to(device)



train(nb_epoch=20,             # number of epoch
         batch_size=4,         # batch size
         store_name='bird',     # folder for output
         resume=False,          # resume training from checkpoint
         start_epoch=0,         # the start epoch number when you resume the training
         model_path='')         # the saved model where you want to resume the training
