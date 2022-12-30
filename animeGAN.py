import torch
from dataset import AnimeDataSet
from torch import nn
from torch import optim
import itertools
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from torchvision.transforms import InterpolationMode
import cv2
import numpy as np
from network import Generator,Discriminator
from vgg import VGG19
import time as t
from tqdm import tqdm
from utils import *
class AnimeGANs(object):
    def __init__(self,args):
        #定义配置
        self.cpu_count=args.cpu_count
        self.istrain=args.istrain
        self.istest=args.istest
        self.init_train=args.init_train
        self.retrain=args.retrain
        self.epochs=args.epochs
        self.init_epochs=args.init_epochs
        self.data_dir=args.data_dir
        self.dataset=args.dataset
        self.result_dir=args.result_dir
        self.save_time=args.save_interval
        self.checkpoint_dir=args.checkpoint_dir
        self.save_image_dir=args.save_image_dir
        self.G_lr=args.lr_g
        self.D_lr = args.lr_d
        self.init_lr=args.init_lr
        self.batch_size=args.batch_size
        self.d_noise=args.d_noise
        self.device=args.device
        #定义模型
        self.G=Generator(dataset=args.dataset).to(self.device)
        self.D=Discriminator(args).to(self.device)
        self.vgg19=VGG19().to(self.device).eval()
        self.vgg19.load_state_dict(torch.load('/Users/mac/Downloads/AnimeGAN/vgg19.pth'))
        #定义优化器
        if self.init_train:
            self.G_optim = optim.Adam(self.G.parameters(), lr=self.init_lr, betas=(0.5, 0.999))
        else:
            self.G_optim=optim.Adam(self.G.parameters(),lr=self.G_lr,betas=(0.5, 0.999))
        self.D_optim=optim.Adam(self.D.parameters(), lr=self.D_lr, betas=(0.5, 0.999))
        #定义损失函数
        self.huber = nn.SmoothL1Loss().to(self.device)
        self.content_loss = nn.L1Loss().to(self.device)
        self.gram_loss = nn.L1Loss().to(self.device)
        self.color_loss = nn.L1Loss().to(self.device)
        self.gan_loss = args.gan_loss
        self.wadvg = args.wadvg
        self.wadvd = args.wadvd
        self.wcon = args.wcon
        self.wgra = args.wgra
        self.wcol = args.wcol
        self.adv_type = args.gan_loss
        #noise
        self.gaussian_mean=torch.tensor(0.0)
        self.gaussian_std=torch.tensor(0.1)
    #高斯噪声
    def gaussian_noise(self):
        return torch.normal(self.gaussian_mean, self.gaussian_std)
    #数据加载。。。。
    def load_data(self):
        trans = [transforms.Resize(286, InterpolationMode.BICUBIC),
                     transforms.CenterCrop(256),
                     transforms.RandomHorizontalFlip(0.5),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        trans_gray = [transforms.Resize(286, InterpolationMode.BICUBIC),
                 transforms.CenterCrop(256),
                 transforms.RandomHorizontalFlip(0.5),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5), (0.5))]
        self.data_loader = DataLoader(AnimeDataSet(root=self.data_dir,mode=self.dataset,trans=trans,trans_gray=trans_gray),
                                  batch_size=self.batch_size, shuffle=True,num_workers=self.cpu_count,
                                  pin_memory=True,drop_last=True)
    #输入数据
    def set_inputs(self,input):
        self.source_img=input['source'].to(self.device)
        self.style_img = input['style'].to(self.device)
        self.stg_img = input['style_gray'].to(self.device)
        self.smg_img = input['smooth_gray'].to(self.device)
    #训练
    def train(self):
        print("training on 4090")
        print('=================train start!=========================')
        self.load_data()#读取图像
        start_time = t.time()
        for step in tqdm(range(1,self.init_epochs+1)):
            for i,data in enumerate(self.data_loader):
                count=len(self.data_loader)
                if step>(self.epochs//2):
                    self.G_optim.param_groups['lr'][0]-=self.G_lr/(self.epochs//2)
                    self.D_optim.param_groups['lr'][0]-=self.D_lr/(self.epochs//2)
                    print("学习率已更新")
                self.G.train()
                self.set_inputs(data)
                # train G with  content loss only
                if  not self.init_train:
                    #========================D=================
                    self.D.train()
                    self.D_optim.zero_grad()
                    fake_img=self.G(self.source_img).detach()
                    if self.d_noise:
                        fake_img += gaussian_noise()
                        self.style_img += gaussian_noise()
                        self.stg_img += gaussian_noise()
                        self.smg_img += gaussian_noise()
                    fake_d = self.D(fake_img)
                    real_anime_d = self.D(self.style_img)
                    real_anime_gray_d = self.D(self.stg_img)
                    real_anime_smg_gray_d = self.D(self.smg_img)
                    #loss
                    real_anime_d=torch.mean(torch.square(real_anime_d-1.0))
                    fake_d=torch.mean(torch.square(fake_d))#lsgan
                    real_anime_gray_d = torch.mean(torch.square(real_anime_gray_d))
                    real_anime_smg_gray_d= torch.mean(torch.square(real_anime_smg_gray_d))
                    loss_d=self.wadvd*(real_anime_d+fake_d+real_anime_gray_d+0.2*real_anime_smg_gray_d)
                    loss_d.backward()
                    self.D_optim.step()
                    #=========================train g=================
                    self.G_optim.zero_grad()
                    fake_img=self.G(self.source_img)
                    fake_d=self.D(fake_img)
                    fake_teture=self.vgg19(fake_img)
                    real_teture=self.vgg19(self.source_img)
                    styg_teture=self.vgg19(self.stg_img)
                    adv_loss=torch.mean(torch.square(fake_d - 1.0))
                    con_loss=self.content_loss(fake_teture,real_teture)
                    gram_loss=self.gram_loss(gram(styg_teture),gram(fake_teture))
                    source_yuv=rgb_to_yuv(self.source_img)
                    fake_yuv=rgb_to_yuv(fake_img)
                    col_loss=(self.color_loss(source_yuv[:,:,:,0],fake_yuv[:,:,:,0])+self.huber(source_yuv[:,:,:,1],fake_yuv[:,:,:,1])+\
                             self.huber(source_yuv[:,:,:,2],fake_yuv[:,:,:,2]))
                    loss_g=adv_loss*self.wadvg+con_loss*self.wcon+gram_loss*self.wgra+col_loss*self.wcol
                    loss_g.backward()
                    self.G_optim.step()
                else:
                    self.G_optim.zero_grad()
                    fake_img = self.G(self.source_img)
                    real_con = self.vgg19(self.source_img)
                    fake_con = self.vgg19(fake_img)
                    loss_con = self.content_loss(fake_con, real_con.detach())
                    loss_con.backward()
                    self.G_optim.step()
                t_end=t.time()
                if self.init_train:
                    print(
                        f"epoch:[{step}/{self.epochs}],iter:[{i}/{count}],loss_G:{loss_con},G_lr:{self.G_optim.param_groups[0]['lr']},time:{time_change(t_end -start_time)}")
                else:
                    print(
                        f"epoch[{step}/{self.epochs}],iter[{i}/{count}],loss_G:{loss_G},loss_D1:{loss_D1},loss_D2:{loss_D2},G_lr:{self.G_optim.param_groups[0]['lr']},time:{time_change(end_time - start_time)}")
            if self.save_image_dir%1==0:
                train_sample_num = 5
                data_loader = self.load_data()
                style= np.zeros((self.hw * 3, 0, 3))
                self.G.eval(), self.D.eval()
                for _ in range(train_sample_num):
                    for i, data in tqdm(enumerate(data_loader)):
                        break
                    real_img = data['source'].to(self.device)
                    style_img = data['style'].to(self.device)
                    fake_ing=self.G(real_img)  # 生成假图
                    style = np.concatenate((style, np.concatenate((RGB2BGR(denorm(real_img[0])),
                                                                     RGB2BGR(denorm(style_img[0])),
                                                                     RGB2BGR(denorm(fake_img[0]))), 0)), 1)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'anime_%06d.png' % step), style)
                print("测试图像生成成功！")
                self.save_model()
                # 保存模型

    def save_model(self):
        params = {}
        params["G"] = self.G.state_dict()
        params["D"] = self.D.state_dict()
        torch.save(params, os.path.join(self.result_dir, self.dataset,self.checkpoint_dir,f'checkpoints_{self.dataset}.pth'))
        print("保存模型成功！")
        # 加载模型

    def load_model(self):
        params = torch.load(os.path.join(self.result_dir,self.dataset,self.checkpoint_dir, f'checkpoints_{self.dataset}.pth'))
        self.G.load_state_dict(params['G'])
        self.D.load_state_dict(params['D'])
        print("加载模型成功！")
#保存生成图像
    def test(self):
        self.load_model()
        data_loader=self.load_data()
        self.G.eval()
        for i ,data in tqdm(enumerate(data_loader)):
            real_img=data['source'].to(self.device)
            fake_img=self.G(real_img)# 生成假图
            fake_img=RGB2BGR(denorm(fake_a[0]))
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test/img','style_%06d.png' % i), fake_img)
            if i==3000:
                break
        print("测试图像生成成功！")












