import torch.nn.functional as F
from mylib import *
from options import *
import numpy as np
import gc
import tqdm
import torch.autograd as autograd
from dataset import *
def gradient_penalty(netD, real, fake, char_class_oh, batch_size, gamma=1):
    device = real.device
    alpha = torch.rand(batch_size, 1, 1, 1, requires_grad=True).to(device)
    x = alpha*real + (1-alpha)*fake
    d_ = netD.forward(x, char_class_oh)[0]
    g = torch.autograd.grad(outputs=d_, inputs=x,
                            grad_outputs=torch.ones(d_.shape).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    g = g.reshape(batch_size, -1)
    return ((g.norm(2,dim=1)/gamma-1.0)**2).mean()

def train(param):
    # paramの変数
    opts = param["opts"]
    stage = param["stage"]
    test_z = param["z"]
    label_weight = param['label_weight']
    pos_weight = param['pos_weight']
    DataLoader = param["DataLoader"]
    G_model = param["G_model"]
    D_model = param["D_model"]
    G_optimizer = param["G_optimizer"]
    D_optimizer = param["D_optimizer"]
    log_dir = param['log_dir']
    iter_start = param["iter_start"]
    ID = param['ID']
    writer = param['writer']
    ##training start
    G_model.train()
    D_model.train()
    iter = iter_start
    #lossの初期化
    D_running_TF_loss = 0
    G_running_TF_loss = 0
    D_running_cl_loss = 0
    G_running_cl_loss = 0
    real_acc = []
    fake_acc = []
    class_mAP = []
    #Dataloaderの定義
    databar = tqdm.tqdm(DataLoader)
    #バッチごとの計算
    #マルチクラス分類
    bce_loss = torch.nn.BCEWithLogitsLoss(weight=label_weight, pos_weight=pos_weight).to(opts.device)
    mse_loss = torch.nn.MSELoss().to(opts.device)
    ca_loss = CALoss()

    for batch_idx, samples in enumerate(databar):
        real_img, char_class, labels = samples['img_target'], samples['charclass_target'], samples['multi_embed_label_target']
        if stage == 1:
            real_img = F.adaptive_avg_pool2d(real_img, (32, 32))
        # バッチの長さを定義
        batch_len = real_img.size(0)
        #デバイスの移
        real_img,  char_class = \
            real_img.to(opts.device), char_class.to(opts.device)
        # 文字クラスのone-hotベクトル化
        char_class_oh = torch.eye(opts.char_num)[char_class].to(opts.device)
        # 印象語のベクトル化
        labels_oh = Multilabel_OneHot(labels, len(ID), normalize=False).to(opts.device)
        # training Generator
        #画像の生成に必要なノイズ作成
        z = torch.randn(batch_len, opts.latent_size).to(opts.device)
        ##画像の生成に必要な印象語ラベルを取得
        _, D_real_class = D_model(real_img, char_class_oh)
        gen_label = F.sigmoid(D_real_class.detach())
        # ２つのノイズの結合
        fake_img, mu, logvar = G_model(z, char_class_oh, gen_label)
        D_fake_TF,  D_fake_class = D_model(fake_img, char_class_oh)
        # Wasserstein lossの計算
        G_TF_loss = -torch.mean(D_fake_TF)
        if iter >= opts.classifier_decay:
            # 印象語分類のロス
            G_class_loss = mse_loss(F.sigmoid(D_fake_class), gen_label)
            # CAにおける損失
            G_loss = G_TF_loss + G_class_loss
        else:
            G_loss = G_TF_loss
            G_class_loss = torch.tensor([0])
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        G_running_TF_loss += G_TF_loss.item()
        G_running_cl_loss += G_class_loss.item()

        #training Discriminator
        #Discriminatorに本物画像を入れて順伝播⇒Loss計算
        D_real_TF, D_real_class = D_model(real_img, char_class_oh)
        # 生成用のラベル
        gen_label = F.sigmoid(D_real_class.detach())
        D_real_loss = -torch.mean(D_real_TF)
        fake_img, _, _ = G_model(z, char_class_oh, gen_label)
        D_fake_TF = D_model(fake_img.detach(), char_class_oh)[0]
        D_fake_loss = torch.mean(D_fake_TF)
        gp_loss = gradient_penalty(D_model, real_img.data, fake_img.data, char_class_oh, real_img.shape[0])
        loss_drift = (D_real_TF ** 2).mean()
        #Wasserstein lossの計算
        D_TF_loss = D_fake_loss + D_real_loss + 10 * gp_loss
        # 印象語分類のロス
        D_class_loss = bce_loss(D_real_class, labels_oh)
        #印象語分類のmAP
        C_mAP = mean_average_precision(torch.sigmoid(D_real_class), labels_oh)[1]
        D_loss = D_TF_loss + D_class_loss + 0.001 * loss_drift
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        D_running_TF_loss += D_TF_loss.item()
        D_running_cl_loss += D_class_loss.item()
        class_mAP.append(C_mAP)
        ##caliculate accuracy
        real_pred = 1 * (torch.sigmoid(D_real_TF) > 0.5).detach().cpu()
        fake_pred = 1 * (torch.sigmoid(D_fake_TF) > 0.5).detach().cpu()
        real_TF = torch.ones(real_pred.size(0))
        fake_TF = torch.zeros(fake_pred.size(0))
        r_acc = (real_pred == real_TF).float().sum().item()/len(real_pred)
        f_acc = (fake_pred == fake_TF).float().sum().item()/len(fake_pred)
        real_acc.append(r_acc)
        fake_acc.append(f_acc)

        ##tensor bord
        writer.add_scalars("TF_loss", {'D_TF_loss': D_TF_loss, 'G_TF_loss': G_TF_loss}, iter)
        writer.add_scalars("class_loss", {'D_class_loss': D_class_loss, 'G_class_loss': G_class_loss}, iter)
        writer.add_scalars("Acc", {'real_acc': r_acc, 'fake_acc': f_acc}, iter)

        iter += 1

        if iter % 250 == 0:
            test_label = ['decorative', 'big', 'shading', 'manuscript', 'ghost']
            test_emb_label = [[ID[key]] for key in test_label]
            label = Multilabel_OneHot(test_emb_label, len(ID), normalize=False)
            save_path = os.path.join(log_dir, 'img_iter_%05d_%02d✕%02d.png' % (iter, real_img.size(2), real_img.size(3)))
            visualizer(save_path, G_model, test_z, opts.char_num, label, opts.device)

    D_running_TF_loss /= len(DataLoader)
    G_running_TF_loss /= len(DataLoader)
    D_running_cl_loss /= len(DataLoader)
    G_running_cl_loss /= len(DataLoader)
    class_mAP = np.array(class_mAP).mean(axis=0)
    real_acc = sum(real_acc)/len(real_acc)
    fake_acc = sum(fake_acc)/len(fake_acc)
    check_point = {'G_net': G_model.module.state_dict(),
                   'G_optimizer': G_optimizer.state_dict(),
                   'D_net': D_model.module.state_dict(),
                   'D_optimizer': D_optimizer.state_dict(),
                   'D_epoch_TF_losses': D_running_TF_loss,
                   'G_epoch_TF_losses': G_running_TF_loss,
                   'D_epoch_cl_losses': D_running_cl_loss,
                   'G_epoch_cl_losses': G_running_cl_loss,
                   'epoch_cl_mAP': class_mAP,
                   'epoch_real_acc': real_acc,
                  'epoch_fake_acc':fake_acc,
                   "iter_finish":iter,
                   }
    return check_point