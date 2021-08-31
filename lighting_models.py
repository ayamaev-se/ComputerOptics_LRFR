import pytorch_lightning as pl
from torch.nn import *
from torch import nn
import torch
import unet_ext.unet
import numpy as np
from torch_radon import Radon
import matplotlib.pyplot as plt
from models import UNetBlock
import torch.fft
import pytorch_ssim


ssim_loss = pytorch_ssim.SSIM().cuda()
global_epoch = 0
global_train_step = 0

def channels_save(tensor, path):
    v = tensor[0].detach().cpu().numpy()
    mv = np.max(v)
    miv = np.min(v)
    fig, axs = plt.subplots(2, v.shape[0], figsize=(10 * v.shape[0], 10))
    for i in range(0, v.shape[0]):
        axs[0, i].imshow(v[i], vmin=miv, vmax=mv)
        axs[1, i].imshow(v[i])
    plt.savefig(path)
    plt.close()

angles = np.linspace(0, np.pi, 64, endpoint=False)
big_angles = np.linspace(0, np.pi, 512, endpoint=False)
radon = Radon(1024, angles)
big_radon = Radon(1024, big_angles)
radon_small = Radon(512, angles)
#bypased radon autoencoder
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class InterpolatedNet(pl.LightningModule):

    def __init__(self):

        super().__init__()

        self.ramp = Parameter(torch.ones((513,), requires_grad=True))
        self.ramp2 = Parameter(torch.ones((513,), requires_grad=True))

        self.post = UNetBlock(3, 128, 1, 9)
        self.post_complex = Sequential(
            Conv2d(2, 32, (5,5), padding=2), PReLU(),
            Conv2d(32, 32, (5,5), padding=2), PReLU(),
            Conv2d(32, 2, (1,1))
        )
        self.sig = Sigmoid()

        self.pre_complex = Sequential(
            Conv2d(2, 32, (5,5), padding=2), PReLU(),
            Conv2d(32, 32, (5,5), padding=2), PReLU(),
            Conv2d(32, 32, (5,5), padding=2), PReLU(),
            Conv2d(32, 32, (5,5), padding=2), PReLU(),
            Conv2d(32, 32, (5,5), padding=2), PReLU(),
            Conv2d(32, 4, (5,5), padding=2), PReLU(),
        )

        self.mul = Parameter(torch.ones((512,257), requires_grad=True))
        self.mul2 = Parameter(torch.ones((512,257), requires_grad=True))
        self.mul3 = Parameter(torch.ones((512,257), requires_grad=True))
        self.mul4 = Parameter(torch.ones((512,257), requires_grad=True))
        self.normal_p = Parameter(torch.ones((1,), requires_grad=True))

    def test_forward(self, x, y):
        return self.forward(x)

    def pre(self, x):
        ff = torch.fft.rfftn(x, dim=(2,3))

        real = ff.real
        imag = ff.imag

        ffpp = self.pre_complex(torch.cat([real, imag], dim=1))

        real = ffpp[:,:1]
        imag = ffpp[:,1:2]

        real = torch.unsqueeze(real, -1)
        imag = torch.unsqueeze(imag, -1)

        add = torch.view_as_complex(torch.cat([real, imag], dim=4).clone()) * self.sig(self.mul3)


        real = ffpp[:,2:3]
        imag = ffpp[:,3:4]

        real = torch.unsqueeze(real, -1)
        imag = torch.unsqueeze(imag, -1)

        sec = torch.view_as_complex(torch.cat([real, imag], dim=4).clone()) * self.sig(self.mul4)
        return torch.fft.irfftn(add, dim=(2,3)) + x, torch.fft.irfftn(sec, dim=(2,3))

    def complex_post(self, x):
        # if(len(x.shape) == 3):
        #     x = torch.unsqueeze(x, 1)
        #     y = torch.unsqueeze(y, 1)

        pr, sec = self.pre(x)
        
        return self.hard_post(x, pr, sec)

    def hard_post(self, x, pr, sec):
        # if(len(x.shape) == 3):
        #     x = torch.unsqueeze(x, 1)
        #     y = torch.unsqueeze(y, 1)
        
        pp = self.post(torch.cat([x, pr, sec], dim=1))
        ff = torch.fft.rfftn(pp, dim=(2,3))

        real = ff.real
        imag = ff.imag

        ffpp = self.post_complex(torch.cat([real, imag], dim=1))

        real = ffpp[:,:1]
        imag = ffpp[:,1:]

        real = torch.unsqueeze(real, -1)
        imag = torch.unsqueeze(imag, -1)

        add = torch.view_as_complex(torch.cat([real, imag], dim=4).clone())

        ff = (ff + self.sig(self.mul2) * add) * self.sig(self.mul)
        pp = torch.fft.irfftn(ff, dim=(2,3))
        return pp

    def forward(self, x):
        # sino_x = self.make_sin(x)

        # sino_y = torch.fft.rfft(sino_x, dim=-1)
        # sino_y = sino_y * self.ramp
        # sino_y = torch.fft.irfft(sino_y, dim=-1)

        # m = radon.backprojection(sino_y)
        # m = m[:,:,256:1024-256,256:1024-256]
        
        m = x * self.normal_p
        dm = self.complex_post(m)
        rec = m + dm

        return rec

    

    def make_sin(self, x):
        rec_inp = torch.nn.functional.pad(x, (256,256,256,256))
        return radon.forward(rec_inp) / rec_inp.shape[-1]

    def encoder(self, x):
        sino_x = self.make_sin(x)

        sino_y = torch.fft.rfft(sino_x, dim=-1)
        sino_y = sino_y * self.ramp
        sino_y = torch.fft.irfft(sino_y, dim=-1)

        sino_rec = radon.backprojection(sino_y)
        sino_rec = sino_rec[:,:,256:1024-256,256:1024-256]

        return sino_rec

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=1e-3)
        co = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25000 * 20)
        self.co = co
        self.my_opt = optimizer
        # return optimizer
        return [optimizer], [self.co]

    def training_step(self, train_batch, batch_idx):
        global global_train_step
        global ssim_loss
        x, y = train_batch
        # x = y

        if(len(x.shape) == 3):
            x = torch.unsqueeze(x, 1)
            y = torch.unsqueeze(y, 1)
        
        m = x * self.normal_p
        pr, sec = self.pre(m)
        dm = self.hard_post(m, pr, sec)
        rec = m + dm

        # rec_loss = torch.mean((rec - y) ** 2)
        ymav = y.max().item()
        if ymav < 1e-5:
            ymav = 1

        mse_loss = torch.mean((rec - y) ** 2)
        rec_loss = -ssim_loss(rec / ymav, y / ymav)
        dm_zero_loss = torch.mean(self.make_sin(dm) ** 2)
        pr_loss = torch.mean((pr - y) ** 2)

        loss = dm_zero_loss * 0.1 + mse_loss + pr_loss + rec_loss

        # for param_group in self.my_opt.param_groups:
        #     param_group['lr'] = 1e-3

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('interp', pr_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('dm', dm_zero_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('rec', -rec_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', get_lr(self.my_opt), on_step=True, on_epoch=False, prog_bar=True)

        if (batch_idx % 300 == 0):
            vmin = y[0,0].min().item()
            vmax = y[0,0].max().item()

            fig, axs = plt.subplots(1, 9, figsize=(30,5))
            axs[0].imshow(x[0,0].detach().cpu().numpy(), cmap='gray', vmin=vmin)
            axs[0].set_title('x')

            axs[1].imshow(pr[0,0].detach().cpu().numpy(), cmap='gray', vmin=vmin)
            axs[1].set_title('Interpolating')

            axs[2].imshow(rec[0,0].detach().cpu().numpy(), cmap='gray', vmin=vmin)
            axs[2].set_title('Full rec')

            axs[3].imshow(y[0,0].detach().cpu().numpy(), cmap='gray', vmin=vmin)
            axs[3].set_title('y')

            axs[4].imshow(dm[0,0].detach().cpu().numpy(), cmap='gray')
            axs[4].set_title('dm')

            axs[5].imshow(self.sig(self.mul3).detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axs[5].set_title('Pre Furie multiplayer')


            axs[6].imshow(self.sig(self.mul).detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axs[6].set_title('Furie 2d multiplyer')


            axs[7].imshow(self.sig(self.mul2).detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axs[7].set_title('Additive furie 2d multiplyer')


            axs[8].imshow(sec[0,0].detach().cpu().numpy(), cmap='gray')
            axs[8].set_title('Secondary')

            plt.savefig(f'epoch_imgs/{global_train_step}.png')
            plt.close()

            global_train_step += 1

        self.co.step()
        return loss

    def validation_step(self, batch, batch_idx):
        global global_epoch
        global ssim_loss
        x, y = batch
        # x = y

        if(len(x.shape) == 3):
            x = torch.unsqueeze(x, 1)
            y = torch.unsqueeze(y, 1)

        m = x * self.normal_p
        pr, sec = self.pre(m)
        dm = self.hard_post(m, pr, sec)
        rec = m + dm

        # rec_loss = torch.mean((rec - y) ** 2)
        ymav = y.max().item()
        if ymav < 1e-5:
            ymav = 1

        mse_loss = torch.mean((m - y) ** 2)
        rec_loss = -ssim_loss(rec / ymav, y / ymav)
        dm_zero_loss = torch.mean(self.make_sin(dm) ** 2)

        loss = rec_loss

        if batch_idx == 3:
            vmin = y[0,0].min().item()
            vmax = y[0,0].max().item()

            fig, axs = plt.subplots(1, 6, figsize=(25,5))
            axs[0].imshow(x[0,0].detach().cpu().numpy(), cmap='gray', vmin=vmin)
            axs[0].set_title('x')

            axs[1].imshow(pr[0,0].detach().cpu().numpy(), cmap='gray', vmin=vmin)
            axs[1].set_title('Interpolating')

            axs[2].imshow(rec[0,0].detach().cpu().numpy(), cmap='gray', vmin=vmin)
            axs[2].set_title('Full rec')

            axs[3].imshow(y[0,0].detach().cpu().numpy(), cmap='gray', vmin=vmin)
            axs[3].set_title('y')

            axs[4].imshow(dm[0,0].detach().cpu().numpy(), cmap='gray')
            axs[4].set_title('dm')

            axs[5].plot(self.ramp.detach().cpu().numpy())
            axs[5].set_title('Furie multiplyer')

            plt.savefig(f'phantom_imgs/{global_epoch}.png')
            plt.close()

            global_epoch += 1
 
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
