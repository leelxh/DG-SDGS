# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import cv2
from PIL import Image
import random
import torchvision.utils as vutils
import argparse
affine_par = True
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BoundaryMapping2(nn.Module):
    def __init__(self, num_input, num_output, kernel_sz=None, stride=None, padding=None):
        super(BoundaryMapping2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_output, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x_1):
        so_output = self.conv1(x_1)
        so_output = self.upsample2(so_output)
        so_output = self.conv2(so_output)
        so_output = self.upsample2(so_output)
        so_output = self.conv3(so_output)

        return so_output


class BoundaryMapping3(nn.Module):
    def __init__(self, num_input, num_output, kernel_sz=None, stride=None, padding=None):
        super(BoundaryMapping3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_output, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x_2):
        so_output = self.conv1(x_2)
        so_output = self.upsample2(so_output)
        so_output = self.conv2(so_output)
        so_output = self.upsample2(so_output)
        so_output = self.conv3(so_output)
        so_output = self.upsample2(so_output)

        return so_output


class Transformer(nn.Module):
    def __init__(self, dim, head_dim, grid_size, ds_ratio=4, expansion=1,
                 drop=0., drop_path=0., kernel_size=3, act_layer=nn.SiLU):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = Attention(dim, head_dim, grid_size=grid_size, ds_ratio=ds_ratio, drop=drop)
        self.conv = InvertedResidual(dim, hidden_dim=dim * expansion, out_dim=dim,
            kernel_size=kernel_size, drop=drop, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.conv(x))
        return x



def count_parameters(model):
    number_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Parameters: {:d} or {:.2f}M'.format(number_params, number_params / (1024 * 1024)))
    return number_params


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        # self.IN = None
        # if IN:
        #     self.IN = nn.InstanceNorm2d(planes*4, affine=affine_par)

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


class class_in_block(nn.Module):

    def __init__(self, inplanes, classin_classes=None):
        super(class_in_block, self).__init__()

        self.IN = nn.InstanceNorm2d(inplanes, affine=affine_par)
        self.classin_classes = classin_classes
        self.branches = nn.ModuleList()
        for i in classin_classes:
            self.branches.append(
                nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3, bias=False))

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, masks):
        outs=[]
        idx = 0
        masks = F.softmax(masks,dim=1)
        for i in self.classin_classes:
            mask = torch.unsqueeze(masks[:,i,:,:],1)
            mid = x * mask
            avg_out = torch.mean(mid, dim=1, keepdim=True)
            max_out, _ = torch.max(mid, dim=1, keepdim=True)
            atten = torch.cat([avg_out, max_out, mask], dim=1)
            atten = self.sigmoid(self.branches[idx](atten))
            out = mid*atten
            out = self.IN(out)
            outs.append(out)
        out_ = sum(outs)
        out_ = self.relu(out_)

        return out_


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class ResNetMulti(nn.Module):
    def __init__(self, args, block, layers, num_classes):
        self.inplanes = 64
        super(ResNetMulti, self).__init__()

        # self.classifier_1 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
        # self.classifier_2 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, bias=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.in1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.maxpool = nn.MaxPool2d(2)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.PSE_stage_1 = Transformer(dim=256, head_dim=64, grid_size=8)
        self.PSE_stage_3 = Transformer(dim=1024, head_dim=64, grid_size=8)

        self.boundary_1 = BoundaryMapping2(256, 1)
        self.boundary_3 = BoundaryMapping3(1024, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def print_weights(self):
        weights_keys = self.layer1_pred.state_dict().keys()
        for key in weights_keys:
            if "num_batches_tracked" in key:
                continue
            weights_t = self.layer1_pred.state_dict()[key].numpy()
        return weights_t

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):


        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        b, c, h, w = x.shape

        # passs
        if b > 1:
            chance = random.randint(0, 2)
            h_left = h // 4
            h_right = w // 4
            temp = torch.zeros(b, c, h, w).cuda()
            temp[:, :, :, :] = x[:, :, :, :]
            temp1 = x
            temp2 = x
            temp3 = x
            temp4 = x
            if chance == 1:
                mask_zero = torch.zeros(b, c, h_left, h_right).cuda()

                # left = h_left * random.randint(0, 3)
                # right = h_right * random.randint(0, 3)
                left = random.randint(0, 119)
                right = random.randint(0, 119)
                mask_zero[:, :, :, :] = x[:, :, left: left + h_left, right: right + h_right]

                x = adaptive_instance_normalization(x, mask_zero)

            if chance == 2:
                mask_zero_1 = torch.zeros(b, c, h_left, h_right).cuda()
                mask_zero_2 = torch.zeros(b, c, h_left, h_right).cuda()
                mask_zero_3 = torch.zeros(b, c, h_left, h_right).cuda()
                mask_zero_4 = torch.zeros(b, c, h_left, h_right).cuda()
                x_top_left_new = torch.zeros(b, c, h_left*2, h_right*2).cuda()
                x_top_right_new = torch.zeros(b, c, h_left * 2, h_right * 2).cuda()
                x_bottom_left_new = torch.zeros(b, c, h_left * 2, h_right * 2).cuda()
                x_bottom_right_new = torch.zeros(b, c, h_left * 2, h_right * 2).cuda()

                # top-left
                # left_1 = h_left * random.randint(0, 1)
                # right_1 = h_right * random.randint(0, 1)
                left_1 = random.randint(0, 39)
                right_1 = random.randint(0, 39)
                mask_zero_1[:, :, :, :] = temp1[:, :, left_1: left_1 + h_left, right_1: right_1 + h_right]
                temp1 = adaptive_instance_normalization(temp1, mask_zero_1)
                x_top_left_new[:, :, :, :] = temp1[:, :, : h_left*2, :h_right*2]

                # top-right
                # left_2 = h_left * random.randint(2, 3)
                # right_2 = h_right * random.randint(0, 1)
                left_2 = random.randint(80, 119)
                right_2 = random.randint(0, 39)
                mask_zero_2[:, :, :, :] = temp2[:, :, left_2: left_2 + h_left, right_2: right_2 + h_right]
                temp2 = adaptive_instance_normalization(temp2, mask_zero_2)
                x_top_right_new[:, :, :, :] = temp2[:, :, h_left*2: h_left*4, :h_right*2]

                # bottom-left
                # left_3 = h_left * random.randint(0, 1)
                # right_3 = h_right * random.randint(2, 3)
                left_3 = random.randint(0, 39)
                right_3 = random.randint(80, 119)
                mask_zero_3[:, :, :, :] = temp3[:, :, left_3: left_3 + h_left, right_3: right_3 + h_right]
                temp3 = adaptive_instance_normalization(temp3, mask_zero_3)
                x_bottom_left_new[:, :, :, :] = temp3[:, :, : h_left * 2, h_right * 2:h_right * 4]

                # bottom-right
                # left_4 = h_left * random.randint(2, 3)
                # right_4 = h_right * random.randint(2, 3)
                left_4 = random.randint(80, 119)
                right_4 = random.randint(80, 119)
                mask_zero_4[:, :, :, :] = temp4[:, :, left_4: left_4 + h_left, right_4: right_4 + h_right]
                temp4 = adaptive_instance_normalization(temp4, mask_zero_4)
                x_bottom_right_new[:, :, :, :] = temp4[:, :, h_left * 2: h_left * 4, h_right * 2:h_right * 4]

                temp[:, :, : h_left*2, :h_right*2] = x_top_left_new
                temp[:, :, h_left*2: h_left*4, :h_right*2] = x_top_right_new
                temp[:, :, : h_left * 2, h_right * 2:h_right * 4] = x_bottom_left_new
                temp[:, :, h_left*2: h_left*4, h_right*2:h_right*4] = x_bottom_right_new

                x = temp
                
        x = self.layer1(x)
        x = self.PSE_stage_1(x)
        boundary_1 = self.boundary_1(x)

        x = self.layer2(x)

        x_3 = self.layer3(x)
        x_3 = self.PSE_stage_3(x_3)
        boundary_3 = self.boundary_3(x_3)

        x3 = self.layer5(x_3)
        x3 = F.interpolate(x3, size=input_size, mode='bilinear', align_corners=True)

        x_4 = self.layer4(x_3)

        x4 = self.layer6(x_4)
        x4 = F.interpolate(x4, size=input_size, mode='bilinear', align_corners=True)


        return x4, x3, torch.sigmoid(boundary_1), torch.sigmoid(boundary_3)

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.conv1)
        b.append(self.in1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.PSE_stage_1.parameters())
        b.append(self.PSE_stage_3.parameters())
        b.append(self.boundary_1.parameters())
        b.append(self.boundary_3.parameters())
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.lr}]


def Res50_Class(args, num_classes=21, pretrained=True):
    model = ResNetMulti(args, Bottleneck, [3, 4, 23, 3], num_classes)

    if pretrained:

        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),
                              strict=False)


    return model


if __name__ == '__main__':
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], 19)
    restore_from = './pretrained_model/DeepLab_resnet_pretrained_init-f81d91e8.pth'
    saved_state_dict = torch.load(restore_from)


    new_params = model.state_dict().copy()

    for i in saved_state_dict:
        print("i:",i)
        i_parts = i.split('.')


    for i in new_params:
        print("i_new:",i)
        i_parts = i.split('.')

    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]

    model.load_state_dict(new_params)

