# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 22:43:00 2023

@author: sb3682
"""
import os
import sys
import time
import argparse
import logging

import torch
import torch.nn as nn
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from data import dataset_v2
import modules
import util
from data.noise import noise_torch
from data import fr_v2
from data.loss import fnr
import random
from torchsummary import summary

def train_frequency_representation(args, fr_module, fr_optimizer, fr_criterion_fr, fr_criterion_ae, fr_scheduler, train_loader, val_loader,
                                   xgrid, epoch, tb_writer):
    """
    Train the frequency-representation module for one epoch
    """
    epoch_start_time = time.time()
    fr_module.train()
    loss_train_fr = 0
    loss_train_ae = 0
    loss_train = 0
    
    for batch_idx, (noisy_signal, clean_signal, target_fr, target_fr_2, freq) in enumerate(train_loader):
        if args.use_cuda:
            noisy_signal, clean_signal, target_fr, target_fr_2= noisy_signal.cuda(), clean_signal.cuda(), target_fr.cuda(), target_fr_2.cuda()
        batch = clean_signal.size(0)
        # clean_signal = clean_signal.view(batch,1,-1)
        # noisy_signal = noisy_signal.view(batch,1,-1)
        # print(clean_signal.shape)
        # print(noisy_signal.shape)
        output_ae,output_fr = fr_module(noisy_signal)
        loss_ae = fr_criterion_ae(output_ae, clean_signal)
        loss_fr = fr_criterion_fr(output_fr, target_fr)
        loss = loss_ae + 2*loss_fr
        loss.backward()
        fr_optimizer.step()
        fr_optimizer.zero_grad()
        loss_train_fr += loss_fr.data.item()
        loss_train_ae += loss_ae.data.item()
        loss_train += loss.data.item()

    fr_module.eval()
    loss_val_ae, loss_val_fr, loss_val, fnr_val = 0, 0, 0, 0
    
    for batch_idx, (noisy_signal, clean_signal, target_fr, target_fr_2, freq) in enumerate(val_loader):
        if args.use_cuda:
            noisy_signal, clean_signal, target_fr, target_fr_2 = noisy_signal.cuda(), clean_signal.cuda(), target_fr.cuda(), target_fr_2.cuda()
        batch = clean_signal.size(0)
        # clean_signal = clean_signal.view(batch,1,-1)
        # noisy_signal = noisy_signal.view(batch,1,-1)
        with torch.no_grad():
            output_ae, output_fr = fr_module(noisy_signal)
        
        loss_ae = fr_criterion_ae(output_ae, clean_signal)
        loss_fr = fr_criterion_fr(output_fr, target_fr)
        loss = loss_ae + 3*loss_fr
        loss_val_ae += loss_ae.data.item()
        loss_val_fr += loss_fr.data.item()
        loss_val += loss.data.item()
        nfreq = (freq >= -0.5).sum(dim=1)
        f_hat = fr_v2.find_freq(output_fr.cpu().detach().numpy(), nfreq, xgrid, args.max_n_freq)
        fnr_val += fnr(f_hat, freq.cpu().numpy(), args.signal_dim)

    loss_train_fr /= args.n_training
    loss_val_fr /= args.n_validation
    fnr_val *= 100 / args.n_validation
    loss_train_ae /= args.n_training
    loss_val_ae /= args.n_validation
    loss_train /= args.n_training
    loss_val /= args.n_validation
    
    tb_writer.add_scalar('ae_l2_training', loss_train_ae, epoch)
    tb_writer.add_scalar('fr_l2_training', loss_train_fr, epoch)
    tb_writer.add_scalar('total_l2_training', loss_train, epoch)
    
    tb_writer.add_scalar('ae_l2_validation', loss_val_ae, epoch)
    tb_writer.add_scalar('fr_l2_validation', loss_val_fr, epoch)
    tb_writer.add_scalar('total_l2_validation', loss_val, epoch)
    tb_writer.add_scalar('fr_FNR', fnr_val, epoch)

    fr_scheduler.step(loss_val_fr)
    logger.info("Epochs: %d / %d, AE_train_loss %.2f, FR_train_loss %.2f, Total_train_Loss %.2f, AE_val_loss %.2f, FR_val_loss %.2f, Total_val_loss %.2f,, FNR %.2f %%",
                epoch, args.n_epochs_fr + args.n_epochs_fc, loss_train_ae, loss_train_fr, loss_train, loss_val_ae, loss_val_fr, loss_val,
                fnr_val)


def train_frequency_counting(args, fr_module, fc_module, fc_optimizer, fc_criterion, fc_scheduler, train_loader,
                             val_loader, epoch, tb_writer):
    """
    Train the frequency-counting module for one epoch
    """
    epoch_start_time = time.time()
    fr_module.eval()
    fc_module.train()
    loss_train_fc, acc_train_fc = 0, 0
    for batch_idx, (clean_signal, target_fr, freq) in enumerate(train_loader):
        if args.use_cuda:
            clean_signal, target_fr, freq = clean_signal.cuda(), target_fr.cuda(), freq.cuda()
        noisy_signal = noise_torch(clean_signal, args.snr, args.noise)
        nfreq = (freq >= -0.5).sum(dim=1)
        if args.use_cuda:
            nfreq = nfreq.cuda()
        if args.fc_module_type == 'classification':
            nfreq = nfreq - 1
        with torch.no_grad():
            output_fr = fr_module(noisy_signal)
            output_fr = output_fr.detach()
        output_fc = fc_module(output_fr)
        if args.fc_module_type == 'regression':
            output_fc = output_fc.view(output_fc.size(0))

            nfreq = nfreq.float()
        loss_fc = fc_criterion(output_fc, nfreq)
        if args.fc_module_type == 'classification':
            estimate = output_fc.max(1)[1]
        else:
            estimate = torch.round(output_fc)
        acc_train_fc += estimate.eq(nfreq).sum().cpu().item()
        loss_train_fc += loss_fc.data.item()

        fc_optimizer.zero_grad()
        loss_fc.backward()
        fc_optimizer.step()

    loss_train_fc /= args.n_training
    acc_train_fc *= 100 / args.n_training

    fc_module.eval()

    loss_val_fc = 0
    acc_val_fc = 0
    for batch_idx, (noisy_signal, _, target_fr, freq) in enumerate(val_loader):
        if args.use_cuda:
            noisy_signal, target_fr = noisy_signal.cuda(), target_fr.cuda()
        nfreq = (freq >= -0.5).sum(dim=1)
        if args.use_cuda:
            nfreq = nfreq.cuda()
        if args.fc_module_type == 'classification':
            nfreq = nfreq - 1
        with torch.no_grad():
            output_fr = fr_module(noisy_signal)
            output_fc = fc_module(output_fr)
        if args.fc_module_type == 'regression':
            output_fc = output_fc.view(output_fc.size(0))
            nfreq = nfreq.float()
        loss_fc = fc_criterion(output_fc, nfreq)
        if args.fc_module_type == 'regression':
            estimate = torch.round(output_fc)
        elif args.fc_module_type == 'classification':
            estimate = torch.argmax(output_fc, dim=1)

        acc_val_fc += estimate.eq(nfreq).sum().item()
        loss_val_fc += loss_fc.data.item()

    loss_val_fc /= args.n_validation
    acc_val_fc *= 100 / args.n_validation

    fc_scheduler.step(acc_val_fc)

    tb_writer.add_scalar('fc_loss_training', loss_train_fc, epoch - args.n_epochs_fr)
    tb_writer.add_scalar('fc_loss_validation', loss_val_fc, epoch - args.n_epochs_fr)
    tb_writer.add_scalar('fc_accuracy_training', acc_train_fc, epoch - args.n_epochs_fr)
    tb_writer.add_scalar('fc_accuracy_validation', acc_val_fc, epoch - args.n_epochs_fr)

    logger.info("Epochs: %d / %d, Time: %.1f, Training fc loss: %.2f, Vadidation fc loss: %.2f, "
                "Training accuracy: %.2f %%, Validation accuracy: %.2f %%",
                epoch, args.n_epochs_fr + args.n_epochs_fc, time.time() - epoch_start_time, loss_train_fc,
                loss_val_fc, acc_train_fc, acc_val_fc)

def set_fr_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'psnet':
        net = PSnet(signal_dim=args.signal_dim, fr_size=args.fr_size, n_filters=args.fr_n_filters,
                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers, kernel_size=args.fr_kernel_size)
    elif args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)
        net=net.cuda() 
        print(net)
        summary(net, input_size=(2,args.signal_dim))
    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net


def set_fc_module(args):
    """
    Create a frequency-counting module
    """
    assert args.fr_size % args.fc_downsampling == 0, \
        'The downsampling factor (fc_downsampling) does not divide the frequency representation size (fr_size)'
    net = None
    if args.fc_module_type == 'regression':
        net = FrequencyCountingModule(n_output=1, n_layers=args.fc_n_layers, n_filters=args.fc_n_filters,
                                      kernel_size=args.fc_kernel_size, fr_size=args.fr_size,
                                      downsampling=args.fc_downsampling, kernel_in=args.fc_kernel_in)
    elif args.fc_module_type == 'classification':
        net = FrequencyCountingModule(n_output=args.max_num_freq, n_layers=args.fc_n_layers,
                                      n_filters=args.fc_n_filters)
    else:
        NotImplementedError('Counter module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net

class PSnet(nn.Module):
    def __init__(self, signal_dim=50, fr_size=1000, n_filters=8, inner_dim=100, n_layers=3, kernel_size=3):
        super().__init__()
        self.fr_size = fr_size
        self.num_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim, bias=False)
        mod = []

        if torch.__version__ >= "1.7.0":
            conv_padding =  "same"
        elif torch.__version__ >= "1.5.0":
            conv_padding = kernel_size // 2
        else:
            conv_padding = kernel_size - 1

        for n in range(n_layers):
            in_filters = n_filters if n > 0 else 1
            mod += [
                nn.Conv1d(in_channels=in_filters, out_channels=n_filters, kernel_size=kernel_size,
                          stride=1, padding=conv_padding, bias=False),
                nn.BatchNorm1d(n_filters),
                nn.ReLU()
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.Linear(inner_dim * n_filters, fr_size, bias=True)

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, 1, -1)
        x = self.mod(x).view(bsz, -1)
        output = self.out_layer(x)
        return output


class FrequencyRepresentationModule(nn.Module):
    def __init__(self, signal_dim=200, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        signal_dim = 256
        ae_filters = 64 
        kernel_size = 3 
        self.stride = 2
        self.output_padding = 1

        self.padding_conv = (((signal_dim//2)-1)*self.stride - signal_dim + (kernel_size-1) + 1)//2 + 1
        self.padding_trans = (2*signal_dim - (signal_dim-1)*self.stride - (kernel_size-1) - 1 - self.output_padding)//(-2)
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv1d(2,ae_filters,kernel_size,self.stride,self.padding_conv),
            nn.BatchNorm1d(ae_filters),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
            nn.Conv1d(ae_filters,ae_filters,kernel_size,self.stride,self.padding_conv),
            nn.BatchNorm1d(ae_filters),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
            nn.Conv1d(ae_filters,ae_filters,kernel_size,self.stride,self.padding_conv),
            nn.BatchNorm1d(ae_filters),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(ae_filters, ae_filters, kernel_size, self.stride, self.padding_trans, self.output_padding),
            nn.BatchNorm1d(ae_filters),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
            nn.ConvTranspose1d(ae_filters, ae_filters, kernel_size, self.stride, self.padding_trans, self.output_padding),
            nn.BatchNorm1d(ae_filters),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
            nn.ConvTranspose1d(ae_filters, 2, kernel_size, self.stride, self.padding_trans, self.output_padding),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
            )
        
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        # self.in_layer = nn.Linear(signal_dim, inner_dim * n_filters, bias=False)
        self.in_layer = nn.Sequential(
            nn.Conv1d(2, 128*n_filters, kernel_size=256, stride = 128, padding=0,
                                       bias=False), #stride = 26
            nn.BatchNorm1d(num_features=128*n_filters),
            )
        
        self.down1 = nn.Sequential(
            nn.Conv1d(n_filters, n_filters*2,kernel_size,padding='same'),
            nn.BatchNorm1d(n_filters*2),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.Conv1d(n_filters*2, n_filters*4,kernel_size,self.stride, self.padding_conv),
            nn.BatchNorm1d(n_filters*4),
            nn.ReLU(),
            )
        self.down3 = nn.Sequential(
            nn.Conv1d(n_filters*4, n_filters*8,kernel_size,self.stride,self.padding_conv),
            nn.BatchNorm1d(n_filters*8),
            nn.ReLU(),
            )
        self.down4 = nn.Sequential(
            nn.Conv1d(n_filters*8, n_filters*16,kernel_size,self.stride,self.padding_conv),
            nn.BatchNorm1d(n_filters*16),
            nn.ReLU(),
            )
        self.equal1 = nn.Sequential(
            nn.Conv1d(n_filters*16, n_filters*32,kernel_size,padding='same'),
            nn.BatchNorm1d(n_filters*32),
            nn.ReLU(),
            )
        self.equal2 = nn.Sequential(
            nn.Conv1d(n_filters*32, n_filters*16,kernel_size,padding='same'),
            nn.BatchNorm1d(n_filters*16),
            nn.ReLU(),
            )
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(n_filters*16, n_filters*8,kernel_size,self.stride,self.padding_trans,self.output_padding),
            nn.BatchNorm1d(n_filters*8),
            nn.ReLU(),
            )
        self.cat1 = nn.Sequential(
            nn.ConvTranspose1d(n_filters*16, n_filters*8,kernel_size,padding=1),
            nn.BatchNorm1d(n_filters*8),
            nn.ReLU(),
            )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(n_filters*8, n_filters*4,kernel_size,self.stride,self.padding_trans,self.output_padding),
            nn.BatchNorm1d(n_filters*4),
            nn.ReLU(),
            )
        self.cat2 = nn.Sequential(
            nn.ConvTranspose1d(n_filters*8, n_filters*4,kernel_size,padding=1),
            nn.BatchNorm1d(n_filters*4),
            nn.ReLU(),
            )
        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(n_filters*4, n_filters*2,kernel_size,self.stride,self.padding_trans,self.output_padding),
            nn.BatchNorm1d(n_filters*2),
            nn.ReLU(),
            )
        self.cat3 = nn.Sequential(
            nn.ConvTranspose1d(n_filters*4, n_filters*2,kernel_size,padding=1),
            nn.BatchNorm1d(n_filters*2),
            nn.ReLU(),
            )
        self.up4 = nn.Sequential(
            nn.ConvTranspose1d(n_filters*2, n_filters,kernel_size,padding=1),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            )
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, kernel_out, stride=upsampling,
                                            padding= (kernel_out - upsampling + 1) // 2, output_padding=0, bias=False)

    def forward(self, inp):  
        bsz = inp.size(0)
        # inp = inp.view(bsz, 1, -1)
        encoded = self.encoder(inp)
        # x1 = self.equal(inp)
        x1 = self.decoder(encoded)
        # x_auto = x1.view(bsz,-1) 
        x = self.in_layer(x1) #.view(bsz, self.n_filters, -1)
        x = x.view(bsz, self.n_filters, -1)
        xd1 = self.down1(x)
        xd2 = self.down2(xd1)
        xd3 = self.down3(xd2)
        xd4 = self.down4(xd3)
        xe1 = self.equal1(xd4)
        xe2 = self.equal2(xe1)
        xu1 = self.up1(xe2)
        xcat1 = torch.cat((xu1,xd3),1)
        xcat1 = self.cat1(xcat1)
        xu2 = self.up2(xcat1)
        xcat2 = torch.cat((xu2,xd2),1)
        xcat2 = self.cat2(xcat2)
        xu3 = self.up3(xcat2)
        xcat3 = torch.cat((xu3,xd1),1)
        xcat3 = self.cat3(xcat3)
        xu4 = self.up4(xcat3)
        x2 = self.out_layer(xu4).view(bsz, -1)
        return x1,x2


class FrequencyCountingModule(nn.Module):
    def __init__(self, n_output, n_layers, n_filters, kernel_size, fr_size, downsampling, kernel_in):
        super().__init__()
        mod = [nn.Conv1d(1, n_filters, kernel_in, stride=downsampling, padding=kernel_in - downsampling,
                             padding_mode='circular')]
        for i in range(n_layers):
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size - 1, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        mod += [nn.Conv1d(n_filters, 1, 1)]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.Linear(fr_size // downsampling, n_output)

    def forward(self, inp):   
        bsz = inp.size(0)
        inp = inp[:, None]
        x = self.mod(inp)
        x = x.view(bsz, -1)
        y = self.out_layer(x)
        # y[y < 0.1] = 0
        return y

parser = argparse.ArgumentParser()

# basic parameters
parser.add_argument('--output_dir', type=str, default='./checkpoint/experiment_name', help='output directory')
parser.add_argument('--no_cuda', action='store_true', help="avoid using CUDA when available")

# dataset parameters
parser.add_argument('--batch_size', type=int, default=512, help='batch size used during training')
parser.add_argument('--signal_dim', type=int, default=256, help='dimensionof the input signal')
parser.add_argument('--signal_dim_rand', type=int, default=64, help='dimension of the variational input signal')
parser.add_argument('--fr_size', type=int, default=2048, help='size of the frequency representation')
parser.add_argument('--max_n_freq', type=int, default=10,
                    help='for each signal the number of frequencies is uniformly drawn between 1 and max_n_freq')
parser.add_argument('--min_sep', type=float, default=0.5, 
                    help='minimum separation between spikes, normalized by signal_dim')
##################
parser.add_argument('--distance', type=str, default='random', help='distance distribution between spikes')
##################
parser.add_argument('--amplitude', type=str, default='normal_floor', help='spike amplitude distribution')
parser.add_argument('--floor_amplitude', type=float, default=1, help='minimum amplitude of spikes')
parser.add_argument('--noise', type=str, default='gaussian_blind', help='kind of noise to use')
#parser.add_argument('--snr', type=float, default=np.exp(np.log(10) * float(50) / 10), help='snr parameter') # 0,5,10,20,30,40,50 dB
parser.add_argument('--snr', type=float, default=['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'], help='snr parameter')
# frequency-representation (fr) module parameters
parser.add_argument('--fr_module_type', type=str, default='fr', help='type of the fr module: [fr | psnet]')
parser.add_argument('--fr_n_layers', type=int, default=10, help='number of convolutional layers in the fr module')
parser.add_argument('--fr_n_filters', type=int, default=16, help='number of filters per layer in the fr module')
parser.add_argument('--fr_kernel_size', type=int, default=3,
                    help='filter size in the convolutional blocks of the fr module')
parser.add_argument('--fr_kernel_out', type=int, default=26, help='size of the conv transpose kernel')
parser.add_argument('--fr_inner_dim', type=int, default=128, help='dimension after first linear transformation')
parser.add_argument('--fr_upsampling', type=int, default=16,
                    help='stride of the transposed convolution, upsampling * inner_dim = fr_size')

# frequency-counting (fc) module parameters
parser.add_argument('--fc_module_type', type=str, default='regression', help='[regression | classification]')
parser.add_argument('--fc_n_layers', type=int, default=25, help='number of layers in the fc module')
parser.add_argument('--fc_n_filters', type=int, default=16, help='number of filters per layer in the fc module')
parser.add_argument('--fc_kernel_size', type=int, default=3,
                    help='filter size in the convolutional blocks of the fc module')
parser.add_argument('--fc_downsampling', type=int, default=8, help='stride of the first convolutional layer')
parser.add_argument('--fc_kernel_in', type=int, default=25, help='kernel size of the first convolutional layer')

# kernel parameters used to generate the ideal frequency representation
####################
parser.add_argument('--kernel_type', type=str, default='gaussian',
                    help='type of kernel used to create the ideal frequency representation [gaussian, triangle or closest]')
####################
parser.add_argument('--triangle_slope', type=float, default=4000,
                    help='slope of the triangle kernel normalized by signal_dim')
parser.add_argument('--gaussian_std', type=float, default=0.05,
                    help='std of the gaussian kernel normalized by signal_dim')

# training parameters
parser.add_argument('--n_training', type=int, default=500000, help='# of training data')
parser.add_argument('--n_validation', type=int, default=20000, help='# of validation data')
parser.add_argument('--lr_fr', type=float, default=0.0003,
                    help='initial learning rate for adam optimizer used for the frequency-representation module')
parser.add_argument('--lr_fc', type=float, default=0.0002,
                    help='initial learning rate for adam optimizer used for the frequency-counting module')
parser.add_argument('--n_epochs_fr', type=int, default=200, help='number of epochs used to train the fr module')
parser.add_argument('--n_epochs_fc', type=int, default=0, help='number of epochs used to train the fc module')
parser.add_argument('--save_epoch_freq', type=int, default=10,
                    help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--numpy_seed', type=int, default=100)
parser.add_argument('--torch_seed', type=int, default=76)

args = parser.parse_args()

if torch.cuda.is_available() and not args.no_cuda:
    args.use_cuda = True
else:
    args.use_cuda = False

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    
logger = logging.getLogger(__name__)
    
file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'run.log'))
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

tb_writer = SummaryWriter(args.output_dir)
util.print_args(logger, args)

np.random.seed(args.numpy_seed) # create a particular set of random numbers (same set of numbers are randomly generated over time)
torch.manual_seed(args.torch_seed)

train_loader = dataset_v2.make_train_data(args)
val_loader = dataset_v2.make_eval_data(args)

# frequency_representation_module
fr_module = set_fr_module(args)
fr_optimizer = torch.optim.Adam(fr_module.parameters(), lr=args.lr_fr)
fr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fr_optimizer, 'min', patience=7, factor=0.2, verbose=True)
start_epoch = 1

# frequency_counting_module
fc_module = set_fc_module(args)
fc_optimizer = torch.optim.Adam(fc_module.parameters(), lr=args.lr_fc)
fc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fc_optimizer, 'min', patience=5, factor=0.7, verbose=True)

logger.info('[Network] Number of parameters in the frequency-representation module : %.3f M' % (
            util.model_parameters(fr_module) / 1e6))

fr_criterion_fr = torch.nn.MSELoss(reduction='sum')
fr_criterion_ae = torch.nn.MSELoss(reduction='sum')
#fc_criterion = torch.nn.MSELoss(reduction='sum')

xgrid = np.linspace(-0.5, 0.5, args.fr_size, endpoint=False)

for epoch in range(start_epoch, args.n_epochs_fr):

    if epoch < args.n_epochs_fr:
        train_frequency_representation(args=args, fr_module=fr_module, fr_optimizer=fr_optimizer, fr_criterion_fr=fr_criterion_fr, fr_criterion_ae=fr_criterion_ae,
                                       fr_scheduler=fr_scheduler, train_loader=train_loader, val_loader=val_loader,
                                       xgrid=xgrid, epoch=epoch, tb_writer=tb_writer)
    # else:
    #     train_frequency_counting(args=args, fr_module=fr_module, fc_module=fc_module,
    #                              fc_optimizer=fc_optimizer, fc_criterion=fc_criterion,
    #                              fc_scheduler=fc_scheduler, train_loader=train_loader,
    #                              val_loader=val_loader, epoch=epoch, tb_writer=tb_writer)

    if epoch % args.save_epoch_freq == 0 or epoch == 199:
        util.save(fr_module, fr_optimizer, fr_scheduler, args, epoch, args.fr_module_type)
    #     util.save(fc_module, fc_optimizer, fc_scheduler, args, epoch, 'fc')
        