from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import shutil
import time
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt



from models import *
from utils import progress_bar


#######################################################################
path_current = os.path.dirname(os.path.realpath(__file__))
path_subdir = 'data'
data_filename = 'resnext.txt'

path_file = os.path.join(path_current,path_subdir,data_filename)
f=open(path_file,'w')

#########################################################
global args

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run MNISTdefault = 100')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)

#net = ResNeXt29_2x64d()
net = ResNeXt29_2x64d_rb()



net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)



##################################################################3
train_losses =np.zeros((args.epochs))
train_prec1s =np.zeros((args.epochs))
eval_losses =np.zeros((args.epochs))
eval_prec1s =np.zeros((args.epochs))
x_epoch = np.zeros((args.epochs))
global best_prec1

#####################################################################




################################################################################
def init_weight(net):
    import math
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            c_out, _, kh, kw = m.weight.size()
            n = kh * kw * c_out
            m.weight.data.normal_(0, math.sqrt(2 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

init_weight(net)



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


###########################################################################


# Training
#def train(epoch):
def train(train_loader, model, criterion, optimizer, epoch,f):

########################################
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
#######################################

    print('\nEpoch: %d' % epoch)
    net.train()


######################
    end = time.time()
#######################





    train_loss = 0
    correct = 0
    total = 0
##################################################3
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
##########################################################################3



        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

###########################################################
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()



        if i % args.print_freq == 0:
            f.write('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.2f}({top1.avg:.2f})\t'
                  'Prec@5 {top5.val:.2f}({top5.avg:.2f})\r\n'.format(
                   epoch, i, len(train_loader),
                   loss=losses, top1=top1, top5=top5))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                   epoch, i, len(train_loader),
                   loss=losses, top1=top1, top5=top5))
        
    return losses.avg, top1.avg
#######################################################################


#        optimizer.step()

        # train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))





#def test(epoch):
def test(val_loader, net, criterion,f):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0



    with torch.no_grad():
        end = time.time()

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)



#####################################################
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                f.write('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\r\n'.format(
                       i, len(val_loader), loss=losses,
                       top1=top1, top5=top5))
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), loss=losses,
                       top1=top1, top5=top5))

        f.write(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\r\n'
              .format(top1=top1, top5=top5))
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg
##############################################################################3


            # test_loss += loss.item()
            # _, predicted = outputs.max(1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.t7')
    #     best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
#    train(epoch)
#    test(epoch)
    adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train_loss, train_prec1 = train(trainloader, net, criterion, optimizer, epoch,f)

    # evaluate on validation set
    eval_loss, eval_prec1 = validate(valloader, net, criterion,f)



    train_losses[epoch] = train_loss 
    train_prec1s[epoch] = train_prec1
    eval_losses[epoch] = eval_loss
    eval_prec1s[epoch] = eval_prec1
    x_epoch[epoch] = epoch
#        train_losses =np.append( train_losses + train_loss
#        train_prec1s = train_prec1s + train_prec1
#        eval_losses = eval_losses + eval_loss
#        eval_prec1s = eval_prec1s + eval_prec1
##
#        train_loss.append(train_losses)
#        train_prec1.append(train_prec1s)
#        eval_loss.append(eval_losses)
#        eval_prec1.append(eval_prec1s)









    # remember best prec@1 and save checkpoint
    is_best = eval_prec1 > best_prec1
    best_prec1 = max(eval_prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': 'ResNext',
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
    }, is_best)










matplotlib.use('Agg')

plt.clf()
plt.close()
fig_loss = plt.figure()
ax_loss = fig_loss.add_subplot(1,1,1)
ax_loss.plot(x_epoch,train_losses,label='Train Loss')
ax_loss.plot(x_epoch,eval_losses,label='Test Loss')
ax_loss.legend(loc=1)
ax_loss.set_xlabel('epoch')
ax_loss.set_ylabel('loss')
ax_loss.set_title('Loss of Train and Test')
plot_loss_filename = 'VGGloss.png'
path_loss_file = os.path.join(path_current,path_subdir,plot_loss_filename)
fig_loss.savefig(path_loss_file)

plt.clf()
plt.close()
fig_prec = plt.figure()
ax_prec = fig_prec.add_subplot(1,1,1)
ax_prec.plot(x_epoch,train_prec1s,label='Train Best1')
ax_prec.plot(x_epoch,eval_prec1s,label='Test Best1')
ax_prec.legend(loc=1)
ax_prec.set_xlabel('epoch')
ax_prec.set_ylabel('Best1 Precision')
ax_prec.set_title('Best1 of Train and Test')
plot_prec_filename = 'VGGprec.png'
path_prec_file = os.path.join(path_current,path_subdir,plot_prec_filename)
fig_prec.savefig(path_prec_file)


f.close()
