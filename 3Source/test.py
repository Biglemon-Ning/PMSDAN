import torch
import torch.nn.functional as F
import numpy as np
# import mfsan

def Weight(out1 , out2 , label , pred):

    pred_label = pred.data.max(1)[1]
    weight_source = torch.zeros_like(label).float()
    weight_target = torch.zeros_like(pred_label).float()
    for i in range(mfsan.batch_size):
        equal = torch.eq(pred_label , label[i]).int()
        num = equal.sum()
        equal = equal.unsqueeze(dim= 1)
        buff = torch.mul(out2, equal)
        data = out1[i].unsqueeze(dim= 0).expand(buff.size())
        data = torch.mul(data, equal)
        if num != 0:
            weight_source[i] = torch.sqrt(torch.square(data - buff)).sum() / num.float()
        else:
            data = out1[i]
            buff = torch.nn.functional.one_hot(label[i] , mfsan.num_classes)
            weight_source[i] = 0

    for i in range(mfsan.batch_size):
        equal = torch.eq(pred_label[i] , label).int()
        num = equal.sum()
        equal = equal.unsqueeze(dim= 1)
        buff = torch.mul(out1, equal)
        data = out2[i].unsqueeze(dim= 0).expand(buff.size())
        data = torch.mul(data, equal)
        if num != 0:
            weight_target[i] = torch.sqrt(torch.square(data - buff)).sum() / num.float()
        else:
            weight_target[i] = 0

    weight_target = torch.mul(weight_target , pred.data.max(1)[0])
    weight_source = torch.nn.functional.softmax(weight_source , dim=0)
    weight_target = torch.nn.functional.softmax(weight_target , dim=0)
    weight_source = weight_source.unsqueeze(dim= 1)
    weight_target = weight_target.unsqueeze(dim= 1)
    weight_XX = torch.mul(weight_source , weight_source.t())
    weight_YY = torch.mul(weight_target, weight_target.t())
    weight_XY = torch.mul(weight_source, weight_target.t())

    return weight_XX , weight_YY , weight_XY


if __name__ == '__main__':
    # out1 = torch.rand((mfsan.batch_size , 31))  #source
    # out2 = torch.rand((mfsan.batch_size , 31))  #target
    # label = torch.randint(low= 0 , high= 31 , size=[mfsan.batch_size])    #左闭右开
    # pred = torch.rand((mfsan.batch_size , 31))
    # # pred = pred.data.max(1)[1]
    # # y = torch.eq(label1 , pred).int()
    # Weight(out1 , out2 , label , pred)
    test = torch.ones(1, 12, requires_grad=True)
    print(test.size())
