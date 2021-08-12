import torch
from torch import nn
from .common import *
from .self_attention import Self_Attn
import numpy as np
from mylib import tile_like

class ConvModuleG(nn.Module):
    '''
    Args:
        out_size: (int), Ex.: 16 (resolution)
        inch: (int),  Ex.: 256
        outch: (int), Ex.: 128
    '''

    def __init__(self, out_size, inch, outch, first=False):
        super().__init__()

        if first:
            layers = [
                Conv2d(inch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                Conv2d(outch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),

            ]

        else:
            layers = [
                nn.Upsample((out_size, out_size), mode='nearest'),
                Conv2d(inch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                Conv2d(outch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvModuleD(nn.Module):
    '''
    Args:
        out_size: (int), Ex.: 16 (resolution)
        inch: (int),  Ex.: 256
        outch: (int), Ex.: 128
    '''

    def __init__(self, out_size, inch, outch, char_num=26, imp_num=1574, final=False):
        super().__init__()
        self.final = final
        if final:
            layers = [
                Minibatch_std(),  # final block only
                Conv2d(inch + 1, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                Conv2d(outch, outch, 4, padding=0),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            layer_TF = [nn.Conv2d(outch, 1, 1, padding=0)]
            layer_char = [nn.Conv2d(outch, char_num, 1, padding=0)]
            layer_imp = [nn.Flatten(),
                         nn.Dropout(p=0.5),
                         nn.Linear(outch * 4 * 4, imp_num),]

            self.layer_TF = nn.Sequential(*layer_TF)
            self.layer_char = nn.Sequential(*layer_char)
            self.layer_imp = nn.Sequential(*layer_imp)
        else:
            layers = [
                Conv2d(inch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                Conv2d(outch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                nn.AdaptiveAvgPool2d((out_size, out_size)),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x_ = self.layers(x)
        if self.final:
            x_TF = torch.squeeze(self.layer_TF(x_))
            x_char = torch.squeeze(self.layer_char(x_))
            x_imp = self.layers[:-2](x)
            x_imp = torch.squeeze(self.layer_imp(x_imp))
        else:
            x_TF = x_
            x_char = None
            x_imp = None
        return x_TF, x_char, x_imp

class Generator(nn.Module):
    def __init__(self, weight, latent_size=256, char_num=26, num_dimension=300, attention=False,
                 device=torch.device("cuda")):
        super().__init__()

        # conv modules & toRGBs
        self.attention = attention
        scale = 1
        inchs = np.array([latent_size + char_num + num_dimension, 256, 128, 64, 32, 16], dtype=np.uint32) * scale
        outchs = np.array([256, 128, 64, 32, 16, 8], dtype=np.uint32) * scale
        sizes = np.array([4, 8, 16, 32, 64, 128], dtype=np.uint32)
        firsts = np.array([True, False, False, False, False, False], dtype=np.bool)
        blocks, toRGBs, attn_blocks = [], [], []
        for idx, (s, inch, outch, first) in enumerate(zip(sizes, inchs, outchs, firsts)):
            blocks.append(ConvModuleG(s, inch, outch, first))
            if attention:
                attn_blocks.append(Attention(outch, num_dimension, len(sizes) - (idx + 1)))
        toRGBs = nn.Conv2d(8, 1, 1, padding=0)
        self.emb_layer = ImpEmbedding(weight, deepsets=False, device=device)
        self.CA_layer = Conditioning_Augumentation(num_dimension + char_num, latent_size, device=device)
        self.blocks = nn.ModuleList(blocks)
        self.toRGBs = nn.ModuleList(toRGBs)
        if attention:
            self.attn_blocks = nn.ModuleList(attn_blocks)
            self.attribute_embed = nn.Embedding(num_dimension, 128)
            attrid = torch.tensor([i for i in range(num_dimension)])
            self.attrid = attrid.view(1, attrid.size(0))

        self.size = sizes

def forward(self, x, y_char, y_imp, res, eps=1e-7, emb=True):
    # to image
    n, c = x.shape
    x = x.reshape(n, c // 16, 4, 4)
    if emb:
        y_sc = self.emb_layer(y_imp)
    y_cond = torch.cat([y_sc, y_char], dim=1)
    y_cond = y_cond.reshape(y_cond.size(0), y_cond.size(1), 1, 1)
    y_cond = y_cond.expand(y_cond.size(0), y_cond.size(1), 4, 4)
    # attribute embedding
    if self.attention:
        attrid = self.attrid.repeat(x.size(0), 1).to(y_imp.device)
        attr_raw = self.attribute_embed(attrid)
        y_emb = y_imp.unsqueeze(2) * attr_raw
    # for the highest resolution
    x = torch.cat([x, y_cond], axis=1)

    for i in range(len(self.blocks)):
        x = self.blocks[i](x)
        if self.attention:
            x = self.attn_blocks[i](x, y_emb)

    mu, logvar = None, None
    return torch.tanh(x), mu, logvar

class ACDiscriminator(nn.Module):
    def __init__(self, weight, mask, img_size = 64,  num_dimension = 300, imp_num = 1574, char_num = 26 , mode = 'CP', emb = 'w2v'):
        super(ACDiscriminator, self).__init__()
        self.num_dimension = num_dimension
        self.imp_num = imp_num
        self.img_size = img_size
        self.char_num = char_num
        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1 + char_num, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2,inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True)
            )


        self.fc_TF = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(128 * 16 * 16, 1024)),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

        self.fc_class =nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(128 * 16 * 16, 1024)),
            # nn.BatchNorm1d(1024),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.imp_num)
        )
        self.fc_char = nn.Linear(128 * 16 * 16, char_num)

        self.init_weights()


    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()

    def forward(self, img, label, char_class):
        char = char_class.view(char_class.size(0), char_class.size(1), 1, 1).expand(-1, -1, self.img_size, self.img_size)
        x = self.layer1(torch.cat([img, char], dim=1))
        x = self.layer2(x)
        x = x.view(-1, 128 * 16 * 16)
        x_TF = self.fc_TF(x)
        x_class = self.fc_class(x)
        return x_TF, x_class

class CGenerator(nn.Module):
    def __init__(self,  weights, mask, z_dim = 300, num_dimension = 300, imp_num = 1574 , char_num = 26, attention = True, mode = 'C', emb='w2v'):
        super(CGenerator, self).__init__()
        self.z_dim = z_dim
        self.char_num = char_num
        self.imp_num = imp_num
        self.emb = emb
        if self.emb=='w2v':
            self.w2v_layer = ImpEmbedding(weights, mask, sum_weight=False, deepsets=False)
        elif self.emb =='one-hot':
            num_dimension = imp_num
        self.num_dimension = num_dimension
        #self.Emb_layer = CustomEmbedding(num_dimension, 64, spectral_norm=True)
        self.layer1 = nn.Sequential(
            nn.Linear(self.z_dim + self.char_num , 1500),
            nn.BatchNorm1d(1500),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer2 = nn.Sequential(
            nn.Linear(self.num_dimension, 1500),
            nn.BatchNorm1d(1500),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer3 = nn.Sequential(
            nn.Linear(3000, 128 * 16 * 16),
            nn.BatchNorm1d(128 * 16 * 16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # チャネル数を128⇒64に変える。
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True))
    # self.layer4 = Up(128, 64, num_dimension, 1, attention=True)

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()

    def forward(self, noise, labels, char_class, w2v = True):
        y_1 = self.layer1(torch.cat([noise, char_class], dim=1))  # (100,1,1)⇒(300,1,1)
        # 印象情報のw2v
        if self.emb=='w2v':
            if w2v:
                attr = self.w2v_layer(labels)
            else:
                attr = labels
        elif self.emb == 'one-hot':
            attr = labels
        #印象情報のEmbedding
        # impression_id = torch.LongTensor(list(range(self.num_dimension)))
        # impression_id = impression_id.view(1, impression_id.size(0))
        # impression_id = impression_id.repeat(len(noise), 1)
        # impression_row = self.Emb_layer(impression_id.to(labels.device))
        # impression_feature = attr.unsqueeze(2) * impression_row

        y_2 = self.layer2(attr)  # (300,1,1)⇒(1500,1,1)
        x = torch.cat([y_1, y_2], 1)  # y_1 + y_2=(1800,1,1)
        x = self.layer3(x)  # (1800,1,1)⇒(512*8,1,1)
        x = x.view(-1, 128, 16, 16)  # (512,8,8)
        x = self.layer4(x)  # (512,8,8)⇒(256,16,16)
        x = self.layer5(x)  # (256,16,16)⇒(128,32,32)
        return x

class CDiscriminator(nn.Module):
    def __init__(self, weight, mask, img_size=64, num_dimension=300, imp_num=1574, char_num=26, mode = 'C', emb = 'w2v'):
        super(CDiscriminator, self).__init__()
        self.imp_num = imp_num
        self.img_size = img_size
        self.char_num = char_num
        self.emb = emb
        if self.emb=='w2v':
            self.w2v_layer = ImpEmbedding(weight, mask, sum_weight=False)
        elif self.emb=='one-hot':
            num_dimension = 300
            self.dic_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(1574, num_dimension)),
            nn.LeakyReLU(0.2,inplace=True))
        self.num_dimension = num_dimension
        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1 + self.char_num + self.num_dimension, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc_TF = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(128 * 16 * 16, 1024)),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()

    def forward(self, img, label, char_class):
        char = char_class.view(char_class.size(0), char_class.size(1), 1, 1).expand(-1, -1, self.img_size, self.img_size)
        if self.emb=='w2v':
            attr = self.w2v_layer(label)
        elif self.emb=='one-hot':
            attr = self.dic_layer(label)
        attr = attr.view(attr.size(0), attr.size(1), 1, 1).expand(-1, -1, self.img_size, self.img_size)
        x = self.layer1(torch.cat([img, char, attr], dim=1))
        x = self.layer2(x)
        x = x.view(-1, 128 * 16 * 16)
        x_TF = self.fc_TF(x)
        return x_TF
