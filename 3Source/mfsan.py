# from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import resnet as models
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 32
iteration = 20000
lr = 0.01
momentum = 0.9
cuda = True
seed = 8   #8
log_interval = 10
l2_decay = 5e-4

num_classes = 65    ########换数据集时记得更改####################
root_path = r"./Office_Home_Dataset/"
source1_name = r"Art"
source2_name = r"Clipart"
source3_name = r"Real World"
target_name = r"Product"

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs , None, None)
source2_loader = data_loader.load_training(root_path, source2_name, batch_size, kwargs , None, None)
source3_loader = data_loader.load_training(root_path, source3_name, batch_size, kwargs , None, None)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs , None, None)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)
source_total_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs , source2_name, source3_name)

def train(model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    source3_iter = iter(source3_loader)
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
            {'params': model.cls_fc_son4.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet4.parameters(), 'lr': LEARNING_RATE},
            {'params': model.mydense.parameters(), 'lr': LEARNING_RATE / 10}
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
            source_data, source_label = source3_iter.next()
        except Exception as err:
            source3_iter = iter(source3_loader)
            source_data, source_label = source3_iter.next()
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

        cls_loss, mmd_loss = model(source_data, target_data, source_label, mark=3)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * mmd_loss
        loss.backward()
        optimizer_total.step()

        if i % log_interval == 0:
            print(
                'Train source3 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\t'.format(
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

        cls_loss, mmd_loss = model(source_data, target_data, source_label, mark=4)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * mmd_loss
        loss.backward()
        optimizer_total.step()

        if i % log_interval == 0:
            print(
                'Train source total iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\t'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item()))

        if i % (log_interval * 20) == 0:
            t_correct, correct1, correct2, correct3, correct4 = test(model)
            if t_correct > correct:
                correct = t_correct

            # 向txt文件中写入训练过程 , 'a'为每次写入不会覆盖
            file = open(r'./training.txt', 'a')
            file.writelines(
                ['source 1: ', str(correct1.item()), '\t', 'source 2: ',str(correct2.item()), '\t',
                 'source 3: ',str(correct3.item()), '\t', 'source 4: ',str(correct4.item()), '\t'
                 'max: ', str(correct.item()), '\n'])
            file.close()
            print(source1_name, source2_name, source3_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")

def test(model):
    model.eval()    #BN和Dropout会取训练好的值
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2, pred3, pred4, pred = model(data, mark = 0)

            pred = torch.nn.functional.softmax(pred, dim=1)
            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)
            pred3 = torch.nn.functional.softmax(pred3, dim=1)
            pred4 = torch.nn.functional.softmax(pred4, dim=1)

            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]  #variable可计算梯度，.data可访问原始tensor
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred3.data.max(1)[1]
            correct3 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred4.data.max(1)[1]
            correct4 += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(target_test_loader.dataset)

        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}, source2 accnum {}, source3 accnum {}, source4 accnum {}'.format(correct1, correct2, correct3, correct4))
    return correct, correct1, correct2, correct3, correct4


if __name__ == '__main__':
    model = models.MFSAN(num_classes=num_classes)
    print(model)
    if cuda:
        model.cuda()
    train(model)
