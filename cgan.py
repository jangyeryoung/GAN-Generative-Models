import torch
import torch.nn as nn

class conditional_G(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.img_size = img_size
        self.G = nn.Sequential(
            nn.Linear(in_features=z_dim * 2, out_features=256),  # input size is z_dim + condition size
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=self.img_size * self.img_size),
            nn.Tanh() # -1~1 범위 출력
        )

    def forward(self, x, c): # x는 노이즈, c는 조건
        batch_size = x.shape[0]
        c = c.unsqueeze(1).expand(x.size()) 
        x = torch.cat((x, c), dim=1)  
        out = self.G(x)
        out = out.view(batch_size, 1, self.img_size, self.img_size)
        return out


class conditional_D(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.D = nn.Sequential(
            nn.Linear(in_features=self.img_size * self.img_size * 2, out_features=1024),
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

    def forward(self, x, c): # x는 이미지, c는 조건
        batch_size = x.shape[0]
        out = x.view(batch_size, -1)
        c = c.unsqueeze(1).expand(out.size()) 
        out = torch.cat((out, c), dim=1) 
        out = self.D(out)  # [batch, 1]
        return out
