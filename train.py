import torch
import torch.optim as optim
from models.gan import vanilla_G, vanilla_D
from models.dcgan import dcgan_G, dcgan_D
from models.cgan import conditional_G, conditional_D


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_type = 'vanilla' # 'vanilla', 'dcgan', 'cgan' 중 선택
z_dim = 100
img_size = 28

if model_type == 'vanilla':
    G = vanilla_G(z_dim, img_size).to(device)
    D = vanilla_D(img_size).to(device)
elif model_type == 'dcgan':
    G = dcgan_G(z_dim, img_size).to(device)
    D = dcgan_D(img_size).to(device)
elif model_type == 'cgan':
    G = conditional_G(z_dim, img_size).to(device)
    D = conditional_D(img_size).to(device)

lr = 0.0002
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

def train_step(real_imgs, labels=None):
    batch_size = real_imgs.size(0)
    real_imgs = real_imgs.to(device)
    if labels is not None: labels = labels.to(device)

    optimizer_D.zero_grad()  
    noise = torch.randn(batch_size, z_dim, device=device)
    
    if model_type == 'cgan':
        fake_imgs = G(noise, labels)
        d_real = D(real_imgs, labels)
        d_fake = D(fake_imgs.detach(), labels)
    else:
        fake_imgs = G(noise)
        d_real = D(real_imgs)
        d_fake = D(fake_imgs.detach())
        
    d_loss, d_real_loss, d_fake_loss = d_loss_fn(d_real, d_fake)
    d_loss.backward()
    optimizer_D.step()
    
    optimizer_G.zero_grad()
    
    if model_type == 'cgan':
        d_fake_for_g = D(fake_imgs, labels)
    else:
        d_fake_for_g = D(fake_imgs)
        
    g_loss = g_loss_fn(d_fake_for_g)
    g_loss.backward()
    optimizer_G.step()
    
    return d_loss.item(), g_loss.item()

print(f"Training pipeline for {model_type} GAN is fully configured.")