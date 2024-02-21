import satlaspretrain_models
import torch
import pytorch_warmup as warmup


weights_manager = satlaspretrain_models.Weights()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class SupHead(torch.nn.Module):
    def __init__(self, backbone_channels, num_categories=2):
        super(SupHead, self).__init__()
        use_channels = backbone_channels
        num_layers = 2
        self.num_outputs = 2

        layers = []
        for _ in range(2):
            layer = torch.nn.Sequential(
               nn.Upsample(mode='bilinear', scale_factor=2),
               torch.nn.Conv2d(use_channels, use_channels, 3, padding=1),
                torch.nn.ReLU(inplace=True))
            layers.append(layer)
        layers.append(torch.nn.Conv2d(use_channels, 2, 1))


        self.layers = torch.nn.Sequential(*layers)
        self.loss_func = lambda logits, targets: torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        
    def forward(self, image_list, raw_features, targets=None):
        raw_outputs = self.layers(raw_features[0])
        loss = None

        outputs = torch.nn.functional.softmax(raw_outputs, dim=1)

        if targets is not None:
            task_targets = torch.stack([target for target in targets], dim=0).long()
            print(task_targets.shape)
            print(raw_outputs.shape)
            loss = self.loss_func(raw_outputs, task_targets)
            #loss = self.loss_func(raw_outputs[:, :, 22:22+14, 22:22+14], task_targets)
            loss = loss.mean()
            #print(loss)


        return outputs, loss

model = weights_manager.get_pretrained_model(model_identifier="Sentinel2_SwinB_MI_MS",
                                             fpn=True,
                                             head = SupHead(128),
                                             #head=satlaspretrain_models.utils.Head.SEGMENT,
                                             num_categories = 2)#Head(128), num_categories=2)

from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
class Dataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # Create a mapping from class label to a unique integer.        
        self.datapoints = os.listdir(self.dataset_path + 'input/')
        self.datapoints = [x for x in self.datapoints if x[-4:] == '.npy']

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, 'input/')
        target_path = os.path.join(self.dataset_path, 'target/')
        input = np.load(img_path + self.datapoints[idx]).astype(np.float32) / 65535
        b2 = np.copy(input[..., 2])
        input[..., 2] = input[..., 0]
        input[..., 0] = b2
        input = np.moveaxis(input, -1, 1)
        input = np.reshape(input, (4*9, 58, 58))
        output = np.load(target_path + self.datapoints[idx]).astype(np.float32)
        output = np.clip(output, 0, 1)
        output = np.pad(output, ((21, 21), (21, 21)), mode = 'constant')
        return input, output

    def __len__(self):
        return len(self.datapoints)

TTCData = Dataset('/Volumes/Macintosh HD/Users/work/Documents/ttc-training-data/')
train_dataloader = DataLoader(
    TTCData,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

model = model.to('cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_steps = len(train_dataloader) * 20
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
#warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)


for epoch in range(1, 35):
    print("Starting Epoch...", epoch)
    batch_count = 0
    for data, target in train_dataloader:
        print(target.shape)
        data = data.to('cpu')
        target = target.to('cpu')

        output, loss = model(data, target)
        if batch_count % 20 == 0:
            print(f"{batch_count}, Train Loss = {loss}")

        #loss.backward()
        #optimizer.step()
        #optimizer.zero_grad()
        #with warmup_scheduler.dampening():
        #    lr_scheduler.step()
        batch_count += 1