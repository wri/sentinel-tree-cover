#!/usr/bin/env python
# coding: utf-8
#!TODO: GRU -> BiGRU, Weight standardization, zoneout, boundary loss, SAM (?)
#!DONE: GRU mods, label smoothing, Conv-GN-Swish-SSE-DropBlock, equibatch, LRSched, WarmUp,
import torch
import pytorch_warmup as warmup
import adabound
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
from torchvision.ops import drop_block2d

from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import hickle as hkl
from tqdm import tqdm
from torch.autograd import Variable

## Building blocks
class SSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.spatial_se = nn.Conv2d(channel, 1, kernel_size=1,
                                    stride=1, padding=0)

    def forward(self, x):
        spa_se = torch.sigmoid(self.spatial_se(x))
        spa_se = torch.mul(x, spa_se)
        return spa_se


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    3,
                                    stride = 1,
                                    padding=1,
                                    bias = False)
        self.relu = torch.nn.SiLU()
        self.gn = torch.nn.GroupNorm(out_channels // 8, out_channels)
        self.sse = SSEBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.gn(x)
        x = self.sse(x)
        return x


class UNetUpBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(mode='nearest', scale_factor=2)
        self.convblock = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.convblock(x)
        return x

class UNetDownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, f, norm = True, pad = 0):
        super(UNetDownBlock, self).__init__()
        self.norm = norm
        self.conv = nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size = 3,
                            stride = 1,
                            padding = pad,
                            bias = False)
        self.relu = torch.nn.SiLU()
        if self.norm:
            self.gn = torch.nn.GroupNorm(out_channels // 8, out_channels)
        self.sse = SSEBlock(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.norm:
            x = self.gn(x)
        x = self.sse(x)
        return x


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvGRU cell. Modified to add group normalization and SSE.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvGRUCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.in_conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=2 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=False,
        )
        self.out_conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=False,
        )
        self.gn1 = torch.nn.GroupNorm(hidden_dim // 8, hidden_dim)
        self.gn2 = torch.nn.GroupNorm(hidden_dim // 8, hidden_dim)
        self.gn3 = torch.nn.GroupNorm(hidden_dim // 8, hidden_dim)
        self.sse = SSEBlock(hidden_dim)

    def forward(self, input_tensor, cur_state):
        combined = torch.cat([input_tensor, cur_state], dim=1)
        z, r = self.in_conv(combined).chunk(2, dim=1)
        z = torch.sigmoid(self.gn1(z))
        r = torch.sigmoid(self.gn2(r))
        h = self.out_conv(torch.cat([input_tensor, r * cur_state], dim=1))
        h = self.gn3(self.sse(h))
        new_state = (1 - z) * cur_state + z * torch.tanh(h)
        return new_state

    def init_hidden(self, batch_size, device):
        return Variable(
            torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
        ).to(device)


class ConvGRU(nn.Module):
    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvGRUCell(
                    input_size=(self.height, self.width),
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self, input_tensor, hidden_state=None, pad_mask=None, batch_positions=None
    ):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        pad_maks (b , t)
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(
                batch_size=input_tensor.size(0), device=input_tensor.device
            )

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=h
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            if pad_mask is not None:
                last_positions = (~pad_mask).sum(dim=1) - 1
                layer_output = layer_output[:, last_positions, :, :, :]

            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



class UNetEncoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gru = ConvGRU(
            input_dim=17,
            input_size=(512, 512),
            hidden_dim=channels,
            kernel_size=(3, 3),
            return_all_layers=False,
        )
        self.downblock10to20 = UNetDownBlock(17, channels, 2, norm = True, pad = 1) # 46 to 44
        self.concatconv = UNetDownBlock(channels * 2, channels * 2, 2, norm = True, pad = 1) # 46 to 44
        self.maxPool1 = torch.nn.MaxPool2d(2, 2) # 44 to 22
        self.downblock20to40 = UNetDownBlock(channels * 2, channels * 4, 2) # 22 to 20
        self.maxPool2 = torch.nn.MaxPool2d(2, 2) # 20 to 10
        #self.downblock40to80 = UNetDownBlock(64, 128, 2)
        #self.maxPool3 = torch.nn.MaxPool2d(2)
        self.forwardblock = UNetDownBlock(channels * 4, channels * 8, 1) # 10 to 8

    def forward(self, x, med, train, drop_prob):
        _, temporal = self.gru(x)
        med = self.downblock10to20(med)
        #print(temporal.shape, med.shape)
        tenm = torch.concat((med, temporal), axis = 1)
        tenm = self.concatconv(tenm)
        tenm = drop_block2d(tenm, p = drop_prob, block_size = 5, training = train)
        #print(tenm.shape)
        twentym = self.maxPool1(tenm)
        twentym = self.downblock20to40(twentym)
        twentym = drop_block2d(twentym, p = drop_prob, block_size = 3, training = train)
         
        fourtym = self.maxPool2(twentym)
        fourtym = self.forwardblock(fourtym)
        fourtym = drop_block2d(fourtym, p = drop_prob, block_size = 3, training = train)

        #eightym = self.maxPool3(fourtym)
        #eightym = self.forwardblock(eightym)
        return [tenm, twentym, fourtym]

class TTCModel(torch.nn.Module):
    def __init__(self, num_channels=32,num_categories = 1):
        super(TTCModel, self).__init__()
        self.encoder = UNetEncoder(num_channels) # 44, 20, 8

        # Decoder is 2 x (up conv -> concat -> conv)
        self.upblock1 = UNetUpBlock(num_channels * 8, num_channels * 4)
        self.coconv1 = ConvBlock(num_channels * 8, num_channels * 4)
        self.upblock2 = UNetUpBlock(num_channels * 4, num_channels * 2)
        self.coconv2 = ConvBlock(num_channels * 4, num_channels * 2)
        self.out_conv1 = UNetDownBlock(num_channels * 2, num_channels * 2, 2)
        self.out_conv = torch.nn.Conv2d(num_channels * 2, 1, 1)
        
        self.loss_func = lambda logits, targets: torch.nn.functional.binary_cross_entropy_with_logits(logits,
                                                                                                      torch.clip(targets, 0.02, 0.95),
                                                                                                      reduction='none')
        
    def forward(self, imgs, targets=None, train = True, drop_prob = 0.):
        # Define forward pass
        encoded = self.encoder(imgs[:, :-1], imgs[:, -1], train, drop_prob)
        unet_encode0 = encoded[0] # 10m, 44
        unet_encode1 = encoded[-2] # 20m, 20
        unet_encode2 = encoded[-1] # 40m, 8

        up1 = self.upblock1(unet_encode2) # 8 -> 16 -> 14
        up1 = torch.concat((up1, unet_encode1[:, :, 2:-2, 2:-2]), 1) # (14, 20) -> 14
        up1 = self.coconv1(up1) # 14 -> 12
        up1 = drop_block2d(up1, training = train, p = drop_prob, block_size = 3)
        up2 = self.upblock2(up1) # 12 -> 24 -> 22
        up2 = torch.concat((up2, unet_encode0[:, :, 6:-6, 6:-6]), 1) # (22, 44)
        up2 = self.coconv2(up2) #(22 -> 20)
        up2 = drop_block2d(up2, training = train, p = drop_prob, block_size = 3)
        up2 = self.out_conv1(up2) #(20 -> 18)
        raw_outputs = self.out_conv(up2)
        loss = None

        outputs = torch.nn.functional.sigmoid(raw_outputs)#, dim=1)

        if targets is not None:
            task_targets = torch.stack([target for target in targets], dim=0)#.long()
            #loss = self.loss_func(raw_outputs, task_targets)
            bord = (14 - 14) // 2
            loss = self.loss_func(raw_outputs[:, :, bord:bord+14, bord:bord+14].squeeze(), task_targets)
            loss = loss.mean()
            
        return outputs, loss



def make_y_array(y_files):
    percs = np.zeros((len(y_files)))
    for i in range(len(y_files)):
        mean = np.load('/Volumes/Macintosh HD/Users/work/Documents/ttc-training-data/target/' + y_files[i][:-4] + '.npy')
        #if np.max(mean) == 1:
        #    mean = mean * 255
        #mean = mean / 2.55
        mean = np.mean(mean)
        percs[i] = mean
    percs = percs * 100
    ids0 = np.argwhere(percs == 0).flatten()
    ids30 = np.argwhere(np.logical_and(percs > 0, percs <= 4)).flatten()
    ids40 = np.argwhere(np.logical_and(percs > 4, percs <= 10)).flatten()
    ids50 = np.argwhere(np.logical_and(percs > 10, percs <= 18)).flatten()
    ids60 = np.argwhere(np.logical_and(percs > 18, percs <= 30)).flatten()
    ids70 = np.argwhere(np.logical_and(percs > 30, percs <= 55)).flatten()
    ids80 = np.argwhere(np.logical_and(percs > 55, percs <= 80)).flatten()
    ids90 = np.argwhere(np.logical_and(percs > 80, percs <= 100)).flatten()

    new_batches = []
    maxes = [len(ids0), len(ids30), len(ids40), len(ids50), len(ids60), len(ids70),
             len(ids80), len(ids90)]
    
    cur_ids = [0] * len(maxes)
    iter_len = len(percs)//(len(maxes))
    for i in range(0, iter_len):
        for i, val in enumerate(cur_ids):
            if val > maxes[i] - 1:
                cur_ids[i] = 0
        if cur_ids[0] >= (maxes[0] - 3):
            cur_ids[0] = 0
        #if cur_ids[8] >= (maxes[8] - 2):
        #    cur_ids[8] = 0
        to_append = [ids0[cur_ids[0]], ids0[cur_ids[0] + 1], ids0[cur_ids[0] + 2],
                    ids30[cur_ids[1]], ids40[cur_ids[2]],
                    ids50[cur_ids[3]], ids60[cur_ids[4]], 
                    ids70[cur_ids[5]], ids80[cur_ids[6]],
                    ids90[cur_ids[7]]]
        cur_ids = [x + 1 for x in cur_ids]
        np.random.shuffle(to_append)
        for i in to_append:
            new_batches.append(i)

    return new_batches


def _normalize(subtile):
        min_all = [0.006576638437476157, 0.0162050812542916, 0.010040436408026246, 
               0.013351644159609368, 0.01965362020294499, 0.014229037918669413, 
               0.015289539940489814, 0.011993591210803388, 0.008239871824216068,
               0.006546120393682765, 0.0, 0.0, 0.0, -0.1409399364817101,
               -0.4973397113668104, -0.09731556326714398, -0.7193834232943873]

        max_all = [0.2691233691920348, 0.3740291447318227, 0.5171435111009385, 
                   0.6027466239414053, 0.5650263218127718, 0.5747005416952773,
                   0.5933928435187305, 0.6034943160143434, 0.7472037842374304,
                   0.7000076295109483, 
                   0.4,
                   0.948334642387533, 
                   0.6729257769285485, 0.8177635298774327, 0.35768999002433816,
                   0.7545951919107605, 0.7602693339366691]
    
        for band in range(0, subtile.shape[-1]):
            mins = min_all[band]
            maxs = max_all[band]
            subtile[..., band] = np.clip(subtile[..., band], mins, maxs)
            midrange = (maxs + mins) / 2
            rng = maxs - mins
            standardized = (subtile[..., band] - midrange) / (rng / 2)
            subtile[..., band] = standardized
        return subtile


def _transform(x):
    # Go from monthly images to quarterly medians
    

    # Re-create the range of the non-uint16 indices
    x[..., -1] *= 2
    x[..., -1] -= 0.7193834232943873
    
    x[..., -2] -= 0.09731556326714398
    x[..., -3] -= 0.4973397113668104,
    x[..., -4] -= 0.1409399364817101

    med = np.median(x, axis = 0)
    x = np.reshape(x, (4, 3,x.shape[1], x.shape[2], x.shape[3]))
    x = np.median(x, axis = 1, overwrite_input = True)
    #print(x.shape, med.shape)
    return np.concatenate([x, med[np.newaxis]], axis = 0)


def dataset_transform(input):
    input = _normalize(_transform(input))
    input = np.moveaxis(input, -1, 1)
    return input


def test_transform(input):
    med = np.median(input, axis = 0)
    input = np.reshape(input, (4, 3,input.shape[1], input.shape[2], input.shape[3]))
    input = np.median(input, axis = 1, overwrite_input = True)
    input = np.concatenate([input, med[np.newaxis]], axis = 0)
    #input = np.median(input, axis = 0)
    input = _normalize(input)
    input = np.moveaxis(input, -1, 0)
    input = np.reshape(input, (input.shape[0]* input.shape[1], input.shape[2], input.shape[3]))
    return input


class Dataset(Dataset):
    def __init__(self, dataset_path, transform):
        self.dataset_path = dataset_path
        # Create a mapping from class label to a unique integer.        
        self.datapoints = os.listdir(self.dataset_path + 'input/')
        self.datapoints = [x for x in self.datapoints if x[-4:] == '.hkl']
        self.percs = make_y_array(self.datapoints)
        self.datapoints = [self.datapoints[i] for i in self.percs]
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, 'input/')
        target_path = os.path.join(self.dataset_path, 'target/')
        input = hkl.load(img_path + self.datapoints[idx]).astype(np.float32) / 65535
        #input = input[:, 1:-1, 1:-1, :]
        # input here is 46 x 46
        size = 28
        bord = (78 - size) // 2
        input = input[:, bord:-bord, bord:-bord, :]
        input = self.transform(input)

        # The output should be 
        output = np.load(target_path + self.datapoints[idx][:-4] + ".npy").astype(np.float32)
        output = np.clip(output, 0, 1)
        if output.shape[0] > 14:
            clip = (output.shape[0] - 14) // 2
            output = output[clip:-clip, clip:-clip]
        return input, output

    def __len__(self):
        return len(self.datapoints)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", dest = 'size', type = int)
    parser.add_argument("--epochs", dest = 'epochs', type = int, default = 75)
    parser.add_argument("--train", dest = 'train', type = bool, default = False)
    parser.add_argument("--batch_size", dest = 'batch_size', type = int, default = 32)
    args = parser.parse_args()

    model = TTCModel(args.size)


    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Initializing model with {pytorch_total_params} params')

    TTCData = Dataset('/Volumes/Macintosh HD/Users/work/Documents/ttc-training-data/', dataset_transform)
    train_dataloader = DataLoader(
        TTCData,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    print(f'Initializing dataset with {len(train_dataloader)*16} length')
    model = model.to('mps')
    #optimizer = adabound.AdaBound(model.parameters(), lr=6e-4, final_lr=0.06)
    #optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_steps = len(train_dataloader) * 100
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    if args.train:
        print(f"Training data for {args.epochs} epochs with {args.batch_size} batch size")
        for epoch in range(1, args.epochs):
            print("Starting Epoch...", epoch)
            batch_count = 0
            losses = []
            TTCData = Dataset('/Volumes/Macintosh HD/Users/work/Documents/ttc-training-data/', dataset_transform)
            train_dataloader = DataLoader(
                TTCData,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8,
            )
            for data, target in tqdm(train_dataloader):
                data = data.to('mps')
                target = target.to('mps')

                output, loss = model(data,
                                    target,
                                    train = True,
                                    drop_prob = np.clip(epoch * 0.01, 0, 0.5))
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.detach().cpu())
                if batch_count % 250 == 0:
                    print(f"{batch_count}, Train Loss = {np.mean(losses)}")
                with warmup_scheduler.dampening():
                    lr_scheduler.step()
                batch_count += 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': 0.16,
                }, 'modelgrumed.pt')

