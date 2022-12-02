# from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import resnet as models
import matplotlib.pyplot as plt
import numpy as np
import tSNE
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Training settings
batch_size = 32
iteration = 20000
lr = 0.01
momentum = 0.9
cuda = True
seed = 8   #8
log_interval = 10
l2_decay = 5e-4

# num_classes = 12    ########换数据集时记得更改####################
# root_path = r"/data/Lemon/Datasets/Image-CLEF"
# source1_name = r"/I"
# source2_name = r"/P"
# target_name  = r"/C"

num_classes = 31    ########换数据集时记得更改####################
root_path = r"/data/Lemon/Datasets/Office-31"
source1_name = r"/A"
source2_name = r"/D"
target_name  = r"/W"

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs , None)
source2_loader = data_loader.load_training(root_path, source2_name, batch_size, kwargs , None)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs , None)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)
source_total_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs , source2_name)

source1_tsne_loader = data_loader.load_testing(root_path, source1_name, batch_size, kwargs)
source2_tsne_loader = data_loader.load_testing(root_path, source2_name, batch_size, kwargs)
# 创建图纸
fig = plt.figure(num=1, figsize=(50, 25), dpi= 100)  # 创建图形实例
plt.xticks()  # 指定坐标的刻度
plt.yticks()
plt.title('t-SNE Embedding of digits', fontsize=14)
def train(model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    target_iter = iter(target_train_loader)
    source_total_iter = iter(source_total_loader)

    correct = 0
    #清空文本
    file = open(r'./training.txt', 'a')
    file.writelines(
        ['================================分割线==========================\n'])
    file.close()

    for i in range(1, iteration + 1):
        model.train()   #BN和Dropout会取平均，train=True
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate：", LEARNING_RATE)
        optimizer_total = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son3.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet3.parameters(), 'lr': LEARNING_RATE},
            {'params': model.mydense.parameters(), 'lr': LEARNING_RATE / 10},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:    #万能异常处理
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)   #张量只能在CPU上计算，变量可在GPU上计算
        target_data = Variable(target_data)
        optimizer_total.zero_grad()   #梯度初始化为0

        cls_loss, mmd_loss = model(source_data, target_data, source_label, mark=1)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
        loss = cls_loss + gamma * mmd_loss
        loss.backward()     #计算梯度
        optimizer_total.step()    #参数更新

        if i % log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\t'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item()))    #.item()得到一个元素张量里面的元素值

        try:
            source_data, source_label = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data, source_label = source2_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)   #variable可使用GPU
        target_data = Variable(target_data)
        optimizer_total.zero_grad()

        cls_loss, mmd_loss = model(source_data, target_data, source_label, mark=2)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * mmd_loss
        loss.backward()
        optimizer_total.step()

        if i % log_interval == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\t'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item()))

        try:
            source_data, source_label = source_total_iter.next()
        except Exception as err:
            source_total_iter = iter(source_total_loader)
            source_data, source_label = source_total_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)  # variable可使用GPU
        target_data = Variable(target_data)
        optimizer_total.zero_grad()

        cls_loss, mmd_loss = model(source_data, target_data, source_label, mark=3)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * mmd_loss
        loss.backward()
        optimizer_total.step()

        if i % log_interval == 0:
            print(
                'Train source total iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\t'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item()))

        if i % (log_interval * 20) == 0:
            # if i % 2000 == 0:
                # fig = plt.gcf()
                # pic = fig.subplots(2, 4, sharey=True)
                # 创建图纸
                # print('开始进行t-SNE可视化')
                # draw(source1_tsne_loader, source2_tsne_loader, 0, pic)
                # draw(source2_tsne_loader, target_test_loader, 1, pic)
                # for m in range(4):
                #     # for n in range(4):
                #     extent = pic[0, m].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                #     fig.savefig(r'.\img\img{}{}'.format(i, m), bbox_inches = extent)
                # plt.clf()
            t_correct, correct1, correct2, correct3 = test(model)
            if t_correct > correct:
                correct = t_correct

            # 向txt文件中写入训练过程 , 'a'为每次写入不会覆盖
            file = open(r'./training.txt', 'a')
            file.writelines(
                ['source 1: ', str(correct1.item()), '\t', 'source 2: ',str(correct2.item()), '\t',
                 'source 3: ',str(correct3.item()), '\t', 'source final:', str(t_correct.item()), '\t' ,
                 'max: ', str(correct.item()), '\n'])
            file.close()
            print(source1_name, source2_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")

def test(model):
    model.eval()    #BN和Dropout会取训练好的值
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2, pred3, pred = model(data, mark = 0)

            pred = torch.nn.functional.softmax(pred, dim=1)
            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)
            pred3 = torch.nn.functional.softmax(pred3, dim=1)

            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]  #variable可计算梯度，.data可访问原始tensor
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred3.data.max(1)[1]
            correct3 += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(target_test_loader.dataset)

        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}, source2 accnum {}, source3 accnum {}'.format(correct1, correct2, correct3))
    return correct, correct1, correct2, correct3

def draw(data_loader, test_loader, source, pic):
    model.eval()
    data1 = torch.tensor(0).cpu()
    data2 = torch.tensor(0).cpu()
    data3 = torch.tensor(0).cpu()
    label = torch.tensor(0).cpu()
    with torch.no_grad():
        for data_source1, label_source in data_loader:
            data_source1, label_source = data_source1.cuda(), label_source.cuda()
            data_source1, label_source = Variable(data_source1), Variable(label_source)
            data_src1, data_src2, data_src3, data_src = model(data_source1)
            if data1.sum() == 0:
                data1 = data_src1
                data2 = data_src2
                data3 = data_src3
                data = data_src
                label = label_source
            data1 = torch.cat((data1.cpu(), data_src1.cpu()), 0)
            data2 = torch.cat((data2.cpu(), data_src2.cpu()), 0)
            data3 = torch.cat((data3.cpu(), data_src3.cpu()), 0)
            data = torch.cat((data.cpu(), data_src.cpu()), 0)
            label = torch.cat((label.cpu(), label_source.cpu()), 0)
            length = data.shape[0]
    with torch.no_grad():
        for data_source1, label_source in test_loader:
            data_source1, label_source = data_source1.cuda(), label_source.cuda()
            data_source1, label_source = Variable(data_source1), Variable(label_source)
            data_src1, data_src2, data_src3, data_src = model(data_source1)
            if data1.sum() == 0:
                data1 = data_src1
                data2 = data_src2
                data3 = data_src3
                data = data_src
                label = label_source
            data1 = torch.cat((data1.cpu(), data_src1.cpu()), 0)
            data2 = torch.cat((data2.cpu(), data_src2.cpu()), 0)
            data3 = torch.cat((data3.cpu(), data_src3.cpu()), 0)
            data = torch.cat((data.cpu(), data_src.cpu()), 0)
            label = torch.cat((label.cpu(), label_source.cpu()), 0)
       
        if source == 0:
            tSNE.tsne(np.array(data1), np.array(label), pic[0, 0], flag=length)
            tSNE.tsne(np.array(data2), np.array(label), pic[0, 1], flag=length)
            tSNE.tsne(np.array(data3), np.array(label), pic[0, 2], flag=length)
            tSNE.tsne(np.array(data), np.array(label), pic[0, 3], flag=length)
        elif source == 1:
            tSNE.tsne(np.array(data1), np.array(label), pic[1, 0], flag=length)
            tSNE.tsne(np.array(data2), np.array(label), pic[1, 1], flag=length)
            tSNE.tsne(np.array(data3), np.array(label), pic[1, 2], flag=length)
            tSNE.tsne(np.array(data), np.array(label), pic[1, 3], flag=length)

if __name__ == '__main__':
    model = models.MFSAN(num_classes=num_classes)
    print(model)
    if cuda:
        model.cuda()
    train(model)
