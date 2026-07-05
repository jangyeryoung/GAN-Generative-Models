import torch
import torch.nn as nn

class vanilla_G(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.img_size = img_size
        self.G = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=256), 
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=self.img_size * self.img_size),
            nn.Tanh() # 출력값을 -1 ~ 1 사이로 정규화
        )
    def forward(self, x): # [batch, z_dim]
        batch_size = x.shape[0]
        out = self.G(x) # 랜덤 노이즈를 이미지 데이터로 변환
        out = out.view(batch_size, 1, self.img_size, self.img_size)
        return out

class vanilla_D(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.D = nn.Sequential(
            nn.Linear(in_features=self.img_size * self.img_size, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid(), 
        )

    def forward(self, x): # [batch, 1, img_size, img_size]
        batch_size = x.shape[0]
        out = x.view(batch_size, -1) # x input shape: [batch, 1, 28, 28] -> [batch, 784]로 평탄화
        out = self.D(out) # [batch, 1]
        return out

class G_Loss(nn.Module):
    def __init__(self, device):
        super(G_Loss, self).__init__()
        self.device = device
        self.criterion = nn.BCELoss() #BCEL는 두가지 선택지 중 얼마나 잘 맞췄는지 평가

    def forward(self, fake): 
        ones = torch.ones_like(fake).to(self.device) 
        g_loss = self.criterion(fake, ones) 
        return g_loss


class D_Loss(nn.Module):
    def __init__(self, device):
        super(D_Loss, self).__init__()
        self.device = device
        self.criterion = nn.BCELoss()

    def forward(self, D_real, D_fake):
        ones = torch.ones_like(D_real).to(self.device)  # Discriminator가 실제 이미지를 실제 이미지로 판별하도록 1로 채워진 레이블 생성
        zeros = torch.zeros_like(D_fake).to(self.device)  # Discriminator가 생성된 이미지를 가짜 이미지로 판별하도록 0으로 채워진 레이블 생성

        d_real_loss = self.criterion(D_real, ones)  # 실제 이미지가 실제 이미지로 판별되도록 Binary Cross-Entropy Loss 계산
        d_fake_loss = self.criterion(D_fake, zeros)  # 생성된 이미지가 가짜 이미지로 판별되도록 Binary Cross-Entropy Loss 계산
        d_loss = d_real_loss + d_fake_loss  

        return d_loss, d_real_loss, d_fake_loss
