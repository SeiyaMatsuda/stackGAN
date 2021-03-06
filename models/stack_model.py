import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from .common import *


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block

class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, imp_dim, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.imp_dim = imp_dim
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf * 4 + nef, ndf * 4),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True))
        self.TF_layer = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=4),
            )
        self.imp_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf * 4 * 4 * 4, self.imp_dim))

    def forward(self, h_code, c_code=None):
        # conditioning output
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
            h_c_code = self.outlogits(h_c_code)
        else:
            h_c_code = h_code

        x_TF = self.TF_layer(h_c_code)
        x_imp = self.imp_layer(h_c_code)
        return torch.squeeze(x_TF),  torch.squeeze(x_imp)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

# ############# Networks for stageI GAN #############
class STAGE1_G(nn.Module):
    def __init__(self, weight, latent_size=100, char_num=26, w2v_dimension=300, num_dimension=300, attention=False, device=torch.device("cuda")):
        super(STAGE1_G, self).__init__()
        self.char_dim = char_num
        self.gf_dim = 128 * 8
        self.ninput = latent_size + num_dimension + self.char_dim
        self.c_dim = num_dimension
        self.w2v_dim = w2v_dimension
        self.z_dim = latent_size
        self.attention = attention
        self.weight = weight
        self.device = device
        self.define_module()

    def define_module(self):
        ngf = self.gf_dim
        ninput = self.ninput
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.emb_layer = ImpEmbedding(self.weight, deepsets=False, device=self.device)
        self.CA_layer = Conditioning_Augumentation(self.w2v_dim, self.c_dim, device=self.device)

        self.fc =  nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf // 4, ngf //8)
        # -> 1 x 32 x 32
        self.img = nn.Sequential(
            conv3x3(ngf // 8, 1),
            nn.Tanh())


    def forward(self, noise, y_char, y_imp):
        noise1, noise2=torch.split(noise, [self.z_dim, self.c_dim], dim=1)
        c_code = self.emb_layer(y_imp)
        c_code, mu, logvar = self.CA_layer(c_code, noise2)
        c_code = torch.cat((noise1, c_code, y_char), dim=1)
        h_code = self.fc(c_code)
        h_code = h_code.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        # state size 3 x 32 x 32
        fake_img = self.img(h_code)
        return fake_img, mu, logvar


class STAGE1_D(nn.Module):
    def __init__(self, imp_num=1574, char_num=26,  num_dimension=300, device=torch.device("cuda")):
        super(STAGE1_D, self).__init__()
        self.df_dim = 64
        self.char_dim = char_num
        self.imp_dim = imp_num
        self.c_dim = num_dimension
        self.ef_dim = self.char_dim
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(1+self.char_dim, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 8 x 8
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(0.2, inplace=True)
            # state size (ndf * 4) x 4 x 4)
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, self.c_dim, self.imp_dim, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(ndf, self.c_dim, self.imp_dim, bcondition=False)



    def forward(self, image, y_char):
        y_char = y_char.view(y_char.size(0), y_char.size(1), 1, 1).repeat(1, 1, image.size(2), image.size(3))
        img_embedding = self.encode_img(torch.cat((image, y_char), axis=1))
        return img_embedding

# ############# Networks for stageII GAN #############
class STAGE2_G(nn.Module):
    def __init__(self, STAGE1_G, weight, latent_size=100, char_num=26, w2v_dimension = 300, num_dimension=300, attention=False, device=torch.device("cuda")):
        super(STAGE2_G, self).__init__()
        self.char_dim = char_num
        self.gf_dim = 128
        self.w2v_dim = w2v_dimension
        self.c_dim = num_dimension
        self.z_dim = latent_size
        self.r_num = 4
        self.attention = attention
        self.weight = weight
        self.device = device
        self.define_module()
        self.STAGE1_G = STAGE1_G
        # fix parameters of stageI GAN
        for param in self.STAGE1_G.parameters():
            param.requires_grad = False
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.r_num):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.emb_layer = ImpEmbedding(self.weight, deepsets=False, device=self.device)
        self.CA_layer = Conditioning_Augumentation(self.w2v_dim, self.c_dim, device=self.device)

        # ngf x 32 x 32--> 2ngf ??? 16 ??? 16
        self.encoder = nn.Sequential(
            conv3x3(1, ngf//8),
            nn.ReLU(True),
            nn.Conv2d(ngf//8, ngf//4, 4, 2, 1),
            nn.BatchNorm2d(ngf//4),
            nn.ReLU(True))
        self.hr_joint = nn.Sequential(
            conv3x3(self.c_dim + ngf//4, ngf//4),
            nn.BatchNorm2d(ngf//4),
            nn.ReLU(True))
        self.residual = self._make_layer(ResBlock, ngf//4)
        # --> ngf//8 x 32 x 32
        self.upsample1 = upBlock(ngf//4, ngf//8)
        # --> ngf//16 x 64 x 64
        self.upsample2 = upBlock(ngf//8, ngf//16)
        # --> 1 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 1),
            nn.Tanh())

    def forward(self, noise, y_char, y_imp):
        noise1, noise2 = torch.split(noise, [self.z_dim, self.c_dim], dim=1)
        stage1_img, _, _ = self.STAGE1_G(noise, y_char, y_imp)
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img)
        c_code= self.emb_layer(y_imp)
        c_code, mu, logvar = self.CA_layer(c_code, noise2)
        c_code = c_code.view(-1, self.c_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 16, 16)

        i_c_code = torch.cat([encoded_img, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)

        fake_img = self.img(h_code)
        return fake_img, mu, logvar


class STAGE2_D(nn.Module):
    def __init__(self, imp_num=1574, char_num=26, num_dimension=300, device=torch.device("cuda")):
        super(STAGE2_D, self).__init__()
        self.df_dim = 64
        self.char_dim = char_num
        self.c_dim = num_dimension
        self.ef_dim = self.char_dim
        self.imp_dim = imp_num
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(1 + self.char_dim, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf * 4) x 8 x 8)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf * 8) x 4 x 4)
            conv3x3(ndf * 8, ndf * 4),
            nn.BatchNorm2d(ndf * 4),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(0.2, inplace=True),  #
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, self.c_dim, self.imp_dim, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(ndf, self.c_dim, self.imp_dim, bcondition=False)

    def forward(self, image, y_char):
        y_char = y_char.view(y_char.size(0), y_char.size(1), 1, 1).repeat(1, 1, image.size(2), image.size(3))
        img_embedding = self.encode_img(torch.cat((image, y_char), axis=1))
        return img_embedding