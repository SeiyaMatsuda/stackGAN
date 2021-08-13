import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from .common import *


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True),
        conv3x3(out_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True),
    )
    return block

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
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
    def __init__(self, weight, latent_size=512, char_num=26, num_dimension=300, attention=False, device=torch.device("cuda")):
        super(STAGE1_G, self).__init__()
        self.char_dim = char_num
        self.gf_dim = 512
        self.ninput = latent_size + num_dimension + self.char_dim
        self.c_dim = num_dimension
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
        self.CA_layer = Conditioning_Augumentation(self.c_dim, self.c_dim, device=self.device)

        self.encoder = nn.Sequential(conv3x3(ninput, ngf),
                                    nn.BatchNorm2d(ngf),
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
        y_imp = self.emb_layer(y_imp)
        c_code, mu, logvar = self.CA_layer(y_imp)
        c_code = torch.cat((c_code, y_char), dim=1)
        c_code = c_code.view(c_code.size(0), c_code.size(1), 1, 1).repeat(1,1,4,4)
        noise = noise.view(-1, self.z_dim, 4, 4)
        h_code = torch.cat((noise, c_code), dim=1)
        h_code = self.encoder(h_code)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        # state size 3 x 32 x 32
        fake_img = self.img(h_code)
        return fake_img, mu, logvar


class STAGE1_D(nn.Module):
    def __init__(self, imp_num=1574, char_num=26, device=torch.device("cuda")):
        super(STAGE1_D, self).__init__()
        self.df_dim = 256
        self.char_dim = char_num
        self.imp_dim = imp_num
        self.ef_dim = self.char_dim
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            conv3x3(1, ndf//4),
            nn.BatchNorm2d(ndf // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((ndf // 4, ndf // 4)),
            # state size. (ndf//4) x 16 x 16
            conv3x3(ndf//4, ndf//2),
            nn.BatchNorm2d(ndf//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((ndf//2, ndf//2)),
            # state size (ndf//2) x 8 x 8
            conv3x3(ndf//2, ndf),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((ndf, ndf))
            # state size ndf x  4 x 4
        )

        self.layer_TF_char = nn.Sequential(
            conv3x3(ndf, ndf),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((ndf, ndf))
        )

        self.layer_TF = nn.Sequential(
            nn.Conv2d(ndf, 1, kernel_size=1, stride=1))

        self.layer_char = nn.Sequential(
            nn.Conv2d(ndf, nef, kernel_size=1, stride=1))

        self.layer_imp = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(ndf * 4 * 4, self.imp_dim))


    def forward(self, image):
        img_embedding = self.encode_img(image)
        x_TF_char = self.layer_TF_char(img_embedding)
        x_TF = self.layer_TF(x_TF_char)
        x_char = self.layer_char(x_TF_char)
        x_imp = self.layer_imp(img_embedding)

        return torch.squeeze(x_TF), torch.squeeze(x_char), torch.squeeze(x_imp)

# ############# Networks for stageII GAN #############
class STAGE2_G(nn.Module):
    def __init__(self, STAGE1_G, weight, latent_size=512, char_num=26, num_dimension=300, attention=False, device=torch.device("cuda")):
        super(STAGE2_G, self).__init__()
        self.char_dim = char_num
        self.gf_dim = 512
        self.imp_dim = num_dimension
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
        # ngf x 32 x 32--> 2ngf ✕ 16 ✕ 16
        self.encoder = nn.Sequential(
            conv3x3(1, ngf//8),
            nn.ReLU(True),
            nn.Conv2d(ngf//8, ngf//4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//4),
            nn.ReLU(True))
        self.hr_joint = nn.Sequential(
            conv3x3(self.ef_dim + ngf//4, ngf//4),
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
        _, stage1_img, _, _ = self.STAGE1_G(noise, y_char, y_imp)
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img)
        c_code, mu, logvar = self.emb_layer(y_imp)
        c_code = c_code.view(-1, self.imp_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 16, 16)
        char_code = y_char.view(-1, self.imp_dim, 1, 1)
        char_code = char_code.repeat(1, 1, 16, 16)

        i_c_code = torch.cat([encoded_img, c_code, char_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)

        fake_img = self.img(h_code)
        return fake_img, mu, logvar


class STAGE2_D(nn.Module):
    def __init__(self, imp_num=1574, char_num=26, device=torch.device("cuda")):
        super(STAGE2_D, self).__init__()
        self.df_dim = 512
        self.char_dim = char_num
        self.ef_dim = self.char_dim
        self.imp_dim = imp_num
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(1, ndf//8, 4, 2, 1, bias=False),  # 32 * 32 * nd//8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf//8, ndf//4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf//4),
            nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf//4
            nn.Conv2d(ndf//4, ndf//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf//2),
            nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf//2
            nn.Conv2d(ndf//2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf
            # conv3x3(ndf//2, ndf),
            # nn.BatchNorm2d(ndf),
            # nn.LeakyReLU(0.2, inplace=True)   # 4 * 4 * ndf
        )

        self.layer_TF = nn.Sequential(
            nn.Conv2d(ndf, 1, kernel_size=4, stride=4))

        self.layer_char = nn.Sequential(
            nn.Conv2d(ndf, self.char_dim, kernel_size=4, stride=4))

        self.layer_imp = nn.Sequential(
            nn.Conv2d(ndf, self.imp_dim, kernel_size=4, stride=4))

    def forward(self, image):
        img_embedding = self.encode_img(image)
        x_TF = self.layer_TF(img_embedding)
        x_char = self.layer_char(img_embedding)
        x_imp = self.layer_imp(img_embedding)
        return x_TF, x_char, x_imp