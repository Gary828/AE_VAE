import sys
import torch

class myAE(torch.nn.Module):
    def __init__(self, d_0, d_1, d_2, d_3, d_4):
        super(myAE, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Linear(d_0, d_1),
            torch.nn.ReLU(inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2),
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Linear(d_2, d_3),
            torch.nn.ReLU(inplace=True)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Linear(d_3, d_4),
            torch.nn.Tanh()
        )
    def Encoder(self, H_0):
        H_1 = self.conv1(H_0)
        H_2 = self.conv2(H_1)
        return H_2

    def Decoder(self, H_2):
        H_3 = self.conv3(H_2)
        H_4 = self.conv4(H_3)
        return H_4

    def forward(self, H_0):
        Latent_Representation = self.Encoder(H_0)
        Features_Reconstrction = self.Decoder(Latent_Representation)
        return Latent_Representation, Features_Reconstrction

class myVAE(torch.nn.Module):
    def __init__(self, d_0, d_1, d_2, d_3, d_4):
        super(myVAE, self).__init__()

        self.conv1 = torch.nn.Sequential(

            torch.nn.Linear(d_0, d_1),
            torch.nn.ReLU(inplace=True)
        )

        # VAE有两个encoder，一个用来学均值，一个用来学方差
        self.conv2_mean = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2)

        )
        self.conv2_std = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Linear(d_2, d_3),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Linear(d_3, d_4),
            torch.nn.Tanh()
        )

    def Encoder(self, H_0):
        H_1 = self.conv1(H_0)
        H_2_mean = self.conv2_mean(H_1)
        H_2_std = self.conv2_std(H_1)
        return H_2_mean, H_2_std

    def Reparametrization(self, H_2_mean, H_2_std):
        # randn 就是标准正态分布， rand就是{0,1}之间的均匀分布
        eps = torch.rand_like(H_2_std)
        # H_2_std 并不是方差，而是：H_2_std = log(σ)
        std = torch.exp(H_2_std)
        Latent_Representation = eps * std + H_2_mean
        return Latent_Representation

    # 解码隐变量
    def Decoder(self, Latent_Representation):
        H_3 = self.conv3(Latent_Representation)
        Features_Reconstruction = self.conv4(H_3)
        return Features_Reconstruction

    # 计算重构值和隐变量z的分布参数
    def forward(self, H_0):
        H_2_mean, H_2_std = self.Encoder(H_0)
        Latent_Representation = self.Reparametrization(H_2_mean, H_2_std)
        Features_Reconstruction = self.Decoder(Latent_Representation)
        return Latent_Representation, Features_Reconstruction, H_2_mean, H_2_std



class mySAE(torch.nn.Module):
    def __init__(self, Input_Dim, Middle_Dim, Output_Dim, bias=False):
        super(mySAE, self).__init__()

        # 直接把encoder和decoder写在这里也可以，网络结构比较简单
        self.Encoder = torch.nn.Sequential(
            torch.nn.Linear(Input_Dim, Middle_Dim),
            torch.nn.ReLU(inplace=True)
        )

        self.Decoder = torch.nn.Sequential(
            torch.nn.Linear(Middle_Dim, Output_Dim),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, H_0):
        Latent_Representation = self.Encoder(H_0)
        Features_Reconstrction = self.Decoder(Latent_Representation)
        return Latent_Representation, Features_Reconstrction