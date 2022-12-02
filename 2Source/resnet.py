import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import mfsan
import numpy as np

__all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # return SublinearSequential(*layers)
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


class My_dense(nn.Module):
    def __init__(self, num_class=mfsan.num_classes):
        super(My_dense, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1, num_class, requires_grad=True) * 0.6)
        self.weight2 = nn.Parameter(torch.ones(1, num_class, requires_grad=True) * 0.2)

    def forward(self, source1, source2, source3):
        pred = (torch.mul(source1, self.weight2) + torch.mul(source2, (1 - self.weight1 - self.weight2)) + torch.mul(source3, self.weight1))
        return pred

class MFSAN(nn.Module):

    def __init__(self, num_classes):
        super(MFSAN, self).__init__()
        self.sharedNet = resnet50(True)
        self.sonnet1 = ADDneck(2048, 256)  # 2048-256
        self.sonnet2 = ADDneck(2048, 256)
        self.sonnet3 = ADDneck(2048, 256)
        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.cls_fc_son3 = nn.Linear(256, num_classes)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.mydense = My_dense()

    def forward(self, data_src, data_tgt=0, label_src=0, mark=1):
        mmd_loss = 0
        if self.training == True:
            data_src = self.sharedNet(data_src)
            data_tgt = self.sharedNet(data_tgt)

            data_tgt_son1 = self.sonnet1(data_tgt)
            data_tgt_son1 = self.avgpool(data_tgt_son1)
            data_tgt_son1_mmd = data_tgt_son1.view(data_tgt_son1.size(0), -1)
            pred_tgt1 = self.cls_fc_son1(data_tgt_son1_mmd)

            data_tgt_son2 = self.sonnet2(data_tgt)
            data_tgt_son2 = self.avgpool(data_tgt_son2)
            data_tgt_son2_mmd = data_tgt_son2.view(data_tgt_son2.size(0), -1)
            pred_tgt2 = self.cls_fc_son2(data_tgt_son2_mmd)

            data_tgt_son3 = self.sonnet3(data_tgt)
            data_tgt_son3 = self.avgpool(data_tgt_son3)
            data_tgt_son3_mmd = data_tgt_son3.view(data_tgt_son3.size(0), -1)
            pred_tgt3 = self.cls_fc_son3(data_tgt_son3_mmd)

            data_src1 = self.sonnet1(data_src)
            data_src1 = self.avgpool(data_src1)
            data_src1_mmd = data_src1.view(data_src1.size(0), -1)
            pred1 = self.cls_fc_son1(data_src1_mmd)

            data_src2 = self.sonnet2(data_src)
            data_src2 = self.avgpool(data_src2)
            data_src2_mmd = data_src2.view(data_src2.size(0), -1)
            pred2 = self.cls_fc_son2(data_src2_mmd)

            data_src3 = self.sonnet3(data_src)
            data_src3 = self.avgpool(data_src3)
            data_src3_mmd = data_src3.view(data_src3.size(0), -1)
            pred3 = self.cls_fc_son3(data_src3_mmd)

            if mark == 1:
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt1, dim=1)
                                               - torch.nn.functional.softmax(pred_tgt2, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt1, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt3, dim=1)))
                l1_loss = l1_loss / 2

                cls_loss = F.nll_loss(F.log_softmax(pred1, dim=1), label_src)  # 交叉熵
                conditional_loss = torch.mul(-1 * F.softmax(pred_tgt1, dim=1), F.log_softmax(pred_tgt1, dim=1)).mean()

                # mmd加权
                weight_XX, weight_YY, weight_XY = Weight(pred1, pred_tgt1, label_src, F.softmax(pred_tgt1, dim=1))
                XX, YY, XY = mmd.mmd(data_src1_mmd, data_tgt_son1_mmd)
                mmd_loss += ((torch.mul(weight_XX, XX) + torch.mul(weight_YY, YY) - torch.mul(weight_XY, XY)).sum())

                mmd_loss += Loss(data_src1_mmd, data_tgt_son1_mmd, label_src, F.softmax(pred_tgt1, dim=1)) * 0.02

                mmd_loss = mmd_loss + conditional_loss * 0.1 + l1_loss

                return cls_loss, mmd_loss

            if mark == 2:
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt2, dim=1)
                                               - torch.nn.functional.softmax(pred_tgt1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt2, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt3, dim=1)))
                l1_loss = l1_loss / 2

                cls_loss = F.nll_loss(F.log_softmax(pred2, dim=1), label_src)
                conditional_loss = torch.mul(-1 * F.softmax(pred_tgt2, dim=1), F.log_softmax(pred_tgt2, dim=1)).mean()

                # mmd加权
                weight_XX, weight_YY, weight_XY = Weight(pred2, pred_tgt2, label_src, F.softmax(pred_tgt2, dim=1))
                XX, YY, XY = mmd.mmd(data_src2_mmd, data_tgt_son2_mmd)
                mmd_loss += ((torch.mul(weight_XX, XX) + torch.mul(weight_YY, YY) - torch.mul(weight_XY, XY)).sum())

                mmd_loss += Loss(data_src2_mmd, data_tgt_son2_mmd, label_src, F.softmax(pred_tgt2, dim=1)) * 0.02

                mmd_loss = mmd_loss + conditional_loss * 0.1 + l1_loss

                return cls_loss, mmd_loss

            if mark == 3:  # 总体混在一起进行训练
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt3, dim=1)
                                               - torch.nn.functional.softmax(pred_tgt1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt3, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt2, dim=1)))
                l1_loss = l1_loss / 2

                pred_src = self.mydense(pred1, pred2, pred3)
                pred_tgt = self.mydense(pred_tgt1, pred_tgt2, pred_tgt3)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                conditional_loss = torch.mul(-1 * F.softmax(pred_tgt, dim=1), F.log_softmax(pred_tgt, dim=1)).mean()

                # mmd加权
                weight_XX, weight_YY, weight_XY = Weight(pred_src, pred_tgt, label_src, F.softmax(pred_tgt, dim=1))
                XX, YY, XY = mmd.mmd(data_src3_mmd, data_tgt_son3_mmd)
                mmd_loss += ((torch.mul(weight_XX, XX) + torch.mul(weight_YY, YY) - torch.mul(weight_XY, XY)).sum())

                mmd_loss += Loss(data_src3_mmd, data_tgt_son3_mmd, label_src, F.softmax(pred_tgt, dim=1)) * 0.02

                mmd_loss = mmd_loss + conditional_loss * 0.1 + l1_loss

                return cls_loss, mmd_loss

        else:
            data_src = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data_src)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.cls_fc_son1(fea_son1)

            fea_son2 = self.sonnet2(data_src)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            pred2 = self.cls_fc_son2(fea_son2)

            fea_son3 = self.sonnet3(data_src)
            fea_son3 = self.avgpool(fea_son3)
            fea_son3 = fea_son3.view(fea_son3.size(0), -1)
            pred3 = self.cls_fc_son3(fea_son3)

            pred = self.mydense(pred1, pred2, pred3)

            return pred1, pred2, pred3, pred


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def Weight(out1, out2, label, pred):
    pred_label = pred.data.max(1)[1]
    weight_source = torch.zeros_like(label).float()
    weight_target = torch.zeros_like(pred_label).float()
    for i in range(mfsan.batch_size):
        equal = torch.eq(pred_label, label[i]).int()
        num = equal.sum()
        equal = equal.unsqueeze(dim=1)
        buff = torch.mul(out2, equal)
        data = out1[i].unsqueeze(dim=0).expand(buff.size())
        if num != 0:
            buff = (torch.nn.functional.one_hot(label[i], mfsan.num_classes) * out1[i].max(0)[0]).cuda()
            weight_source[i] = torch.square(data - buff).mean()
        else:
            buff = (torch.nn.functional.one_hot(label[i], mfsan.num_classes) * out1[i].max(0)[0]).cuda()
            weight_source[i] = torch.square(data - buff).mean()

    for i in range(mfsan.batch_size):
        equal = torch.eq(label, pred_label[i]).int()
        num = equal.sum()
        equal = equal.unsqueeze(dim=1)
        buff = torch.mul(out1, equal)
        data = out2[i].unsqueeze(dim=0).expand(buff.size())
        if num != 0:
            buff = (torch.nn.functional.one_hot(pred_label[i], mfsan.num_classes) * out2[i].max(0)[0]).cuda()
            weight_target[i] = torch.square(data - buff).mean()
        else:
            buff = (torch.nn.functional.one_hot(pred_label[i], mfsan.num_classes) * out2[i].max(0)[0]).cuda()
            weight_target[i] = torch.square(data - buff).mean()

    if weight_source.sum() != 0:
        weight_source = weight_source / weight_source.sum()
    else:
        weight_source = torch.zeros(weight_source.size()).cuda()

    if weight_target.sum() != 0:
        weight_target = weight_target / weight_target.sum()
    else:
        weight_target = torch.zeros(weight_target.size()).cuda()

    weight_source = weight_source.unsqueeze(dim=1)
    weight_target = weight_target.unsqueeze(dim=1)
    weight_XX = torch.mul(weight_source, weight_source.t())
    weight_YY = torch.mul(weight_target, weight_target.t())
    weight_XY = torch.mul(weight_source, weight_target.t())

    return weight_XX, weight_YY, weight_XY

def Loss(out1, out2, label, pred):
    pred_label = pred.data.max(1)[1]
    label_set = []
    loss = 0
    intra_loss = 0  #类内
    inter_loss = 0  #类间
    num_intra = 0
    num_inter = 0
    for i in range(0, mfsan.batch_size):
        if (label[i] not in label_set):
            label_set.append(label[i])
        if (pred_label[i] not in label_set):
            label_set.append(pred_label[i])
    for i in range(0, len(label_set)):
        equal_src = torch.eq(label, label_set[i]).int()
        equal_tgt = torch.eq(pred_label, label_set[i]).int()
        if equal_src.sum() != 0 and equal_tgt.sum() != 0:
            num_intra += 1
            index = equal_src.nonzero(as_tuple=False)
            index = index.squeeze(dim=1)
            data_src = out1[index]
            index = equal_tgt.nonzero(as_tuple=False)
            index = index.squeeze(dim=1)
            data_tgt = out2[index]
            intra_loss += mmd.mmd_loss(data_src, data_tgt)

    for i in range(0, len(label_set)):
        for j in range(0, len(label_set)):
            if i != j:
                equal_src = torch.eq(label, label_set[i]).int()
                equal_tgt = torch.eq(pred_label, label_set[j]).int()
                if equal_src.sum() != 0 and equal_tgt.sum() != 0:
                    num_inter += 1
                    index = equal_src.nonzero(as_tuple=False)
                    index = index.squeeze(dim=1)
                    data_src = out1[index]
                    index = equal_tgt.nonzero(as_tuple=False)
                    index = index.squeeze(dim=1)
                    data_tgt = out2[index]
                    inter_loss += mmd.mmd_loss(data_src, data_tgt)

    if num_inter != 0 and num_intra != 0:
        loss += intra_loss / num_intra - (inter_loss / num_inter)
    return loss