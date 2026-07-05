import torch.nn as nn

class dcgan_G(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.img_size = img_size
        self.G = nn.Sequential(
            # 이미지를 점점 키우는 역할
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=64, kernel_size=7,
                               stride=1, padding=0, bias=False), # [64, 7, 7]
            nn.BatchNorm2d(64), 
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4,
                               stride=2, padding=1, bias=False), # [32, 14, 14]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4,
                               stride=2, padding=1, bias=False), # [1, 28, 28]
            nn.Tanh() # 최종 이미지 -1 ~ 1로 정규화
        )

        # 네트워크 초기화
        for m in self.modules() :
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, 1, 1)  # 입력 노이즈를 ConvTranspose2d 입력 형태에 맞춰 [batch_size, z_dim, 1, 1] 크기로 변환
        out = self.G(x) # 업샘플링을 통해 이미지 생성
        return out # [batch_size, 1, 28, 28] 형태로 생성된 이미지 반환

class dcgan_D(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.D = nn.Sequential(
            # 점진적으로 이미지를 압축하는 역할
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4,
                      stride=2, padding=1, bias=False), # [32, 14, 14]
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                      stride=2, padding=1, bias=False), # [64, 7, 7]
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7,
                      stride=1, padding=0, bias=False), # [1, 1, 1]
            nn.Sigmoid(),
        )

        # 네트워크 초기화
        for m in self.modules() :
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        batch_size = x.shape[0] # [b, 1, 28, 28]
        out = self.D(x) # 진짜/가짜 확률 계산 후 출력 [batch, 1, 1, 1]
        out = out.squeeze() 
        return out 
