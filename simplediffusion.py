# 가져오기 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from einops import rearrange #pip install einops 
from typing import  List 
import random 
import math 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 
from timm.utils import ModelEmaV3 #pip install timm 
from tqdm import tqdm #pip install tqdm 
import matplotlib.pyplot as plt #pip install matplotlib 
import torch.optim as optim 
import numpy as np 
import os
from torchvision.datasets import MNIST

import subprocess
from einops import rearrange
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
import time
from torch.cuda.amp import autocast
from torchvision.models import inception_v3
from PIL import Image
import warnings



# 임베딩 단계
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        return embeds[:, :, None, None]
    
# residual 블럭
# Residual Blocks
class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x 
    
# 어텐션
class Attention(nn.Module):
    def __init__(self, C: int, num_heads:int , dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q,k,v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q,k,v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')
 
# Unet 층   
class UnetLayer(nn.Module):
    def __init__(self, 
            upscale: bool, 
            attention: bool, 
            num_groups: int, 
            dropout_prob: float,
            num_heads: int,
            C: int):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x

# Unet 클래스    
class UNET(nn.Module):
    def __init__(self,
            Channels: List = [64, 128, 256, 512, 512, 384],
            Attentions: List = [False, True, False, False, False, True],
            Upscales: List = [False, False, False, True, True, True],
            num_groups: int = 32,
            dropout_prob: float = 0.1,
            num_heads: int = 8,
            input_channels: int = 1,
            output_channels: int = 1,
            time_steps: int = 1000): #! DEFAULT 인자 사용
        super().__init__()
        self.num_layers = len(Channels) #! layer 6개
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        out_channels = (Channels[-1]//2)+Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels)) #! 512
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t):
        x = self.shallow_conv(x) #! 1-> 64
        residuals = []
        for i in range(self.num_layers//2): #! layer 절반
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings)
            residuals.append(r)
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))
    
# 스케줄러
class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]
    
# 시드 설정
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)  
    random.seed(seed)
    
# 훈련
def train(batch_size: int=64,
          num_time_steps: int=1000, #! timestep 1000 으로 설정
          num_epochs: int=15, #! 75
          seed: int=-1,
          ema_decay: float=0.9999,  
          lr=2e-5,
          checkpoint_path: str=None):
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    train_dataset = datasets.MNIST(root='./data', train=True, download=True,transform=transforms.ToTensor()) #! download = True 로 설정
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = UNET().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')

    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x,_) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            x = x.cuda()
            x = F.pad(x, (2,2,2,2))
            t = torch.randint(0,num_time_steps,(batch_size,))
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size,1,1,1).cuda()
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            output = model(x, t) #! 스케줄러로 노이즈 씌운 데이터랑 time stamp
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
        print(f'Epoch {i+1} | Loss {total_loss / (60000/batch_size):.5f}')

    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
    torch.save(checkpoint, 'checkpoints/ddpm_checkpoint') #! epoch 다 돌고 난 이후에 저장
    
save_num=0

# 시각화 코드
def display_reverse(images: List):
    global save_num
    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.show()
    # 저장
    os.makedirs('result', exist_ok=True) # result 디렉토리 생성

    plt.axis('off')  # Hide axis for a cleaner image
    plt.savefig(f'result/{save_num}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    save_num+=1

def inference(checkpoint_path: str=None,
              num_time_steps: int= 1000,
              ema_decay: float=0.9999,
              num_images : int = 10):
    checkpoint = torch.load(checkpoint_path)
    model = UNET().cuda()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    generated_images = []
    times = []  # 각 이미지 생성 시간을 저장

    with torch.no_grad():
        model = ema.module.eval()
        for i in range(num_images):
            torch.cuda.synchronize()  # GPU 작업 완료 대기
            start_time = time.time()  # 시작 시간 기록

            z = torch.randn(1, 1, 32, 32).cuda()
            for t in tqdm(reversed(range(1, num_time_steps)), desc=f'Generating image {i+1}/{num_images}'):
                t = [t]
                temp = (scheduler.beta[t] / ((torch.sqrt(1 - scheduler.alpha[t])) * (torch.sqrt(1 - scheduler.beta[t])))).cuda()
                z = (1 / (torch.sqrt(1 - scheduler.beta[t])).cuda()) * z - (temp * model(z, t).cuda())
                e = torch.randn(1, 1, 32, 32).cuda()
                z = z + (e * torch.sqrt(scheduler.beta[t]).cuda())

            torch.cuda.synchronize()  # GPU 작업 완료 대기
            end_time = time.time()  # 종료 시간 기록

            elapsed_time = end_time - start_time
            times.append(elapsed_time)  # 시간을 저장
            print(f"Image {i+1} generated in {elapsed_time:.4f} seconds.")

            # 최종 이미지 처리
            temp = scheduler.beta[0].cuda() / ((torch.sqrt(1 - scheduler.alpha[0].cuda())) * (torch.sqrt(1 - scheduler.beta[0].cuda())))
            final_image = (1 / (torch.sqrt(1 - scheduler.beta[0].cuda()))) * z - (temp * model(z, [0]).cuda())
            final_image = rearrange(final_image.squeeze(0), "c h w -> h w c").cpu().numpy()

            if final_image.shape[2] == 1:  # 채널이 1개인 경우
                final_image = np.repeat(final_image, 3, axis=2)

            final_image = (final_image - final_image.min()) / (final_image.max() - final_image.min())  # 정규화
            generated_images.append(final_image)

    avg_time = sum(times) / len(times)
    print(f"\nAverage generation time per image: {avg_time:.4f} seconds.")
    return generated_images, times
    
# 스케줄러
class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]
    
# 시드 설정
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
# 훈련 ㄱㄱ
def train(batch_size: int=64,
          num_time_steps: int=1000, #! timestep 1000 으로 설정
          num_epochs: int=15, #! 75
          seed: int=-1,
          ema_decay: float=0.9999,  
          lr=2e-5,
          checkpoint_path: str=None):
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    train_dataset = datasets.MNIST(root='./data', train=True, download=True,transform=transforms.ToTensor()) #! download = True 로 설정
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = UNET().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')

    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x,_) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            x = x.cuda()
            x = F.pad(x, (2,2,2,2))
            t = torch.randint(0,num_time_steps,(batch_size,))
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size,1,1,1).cuda()
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            output = model(x, t) #! 스케줄러로 노이즈 씌운 데이터랑 time stamp
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
        print(f'Epoch {i+1} | Loss {total_loss / (60000/batch_size):.5f}')

    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
    torch.save(checkpoint, 'checkpoints/ddpm_checkpoint') #! epoch 다 돌고 난 이후에 저장
    
save_num=0

# 시각화 코드
def display_reverse(images: List):
    global save_num
    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.show()
    # 저장
    os.makedirs('result', exist_ok=True) # result 디렉토리 생성

    plt.axis('off')  # Hide axis for a cleaner image
    plt.savefig(f'result/{save_num}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    save_num+=1


def inference_and_loss_plot(checkpoint_path: str = None,
                       num_time_steps: int = 1000,
                       ema_decay: float = 0.9999,
                       num_images : int = 10):
    checkpoint = torch.load(checkpoint_path)
    model = UNET().cuda()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    generated_images = []  # FID 계산을 위해 생성된 이미지를 저장
    all_losses = []  # 각 이미지에 대한 loss 값을 저장

    with torch.no_grad():
        model = ema.module.eval()
        for i in range(num_images):  # 10개의 이미지를 생성
            z = torch.randn(1, 1, 32, 32).cuda()
            losses = []  # 각 이미지의 time step에 따른 loss 값을 저장

            for t in tqdm(reversed(range(1, num_time_steps)), desc=f'Generating image {i+1}/{num_images}'):
                t_tensor = [t]
                output = model(z, t_tensor)
                loss = torch.mean((output - z) ** 2).item()  # MSE 계산
                losses.append(loss)

                temp = (scheduler.beta[t_tensor].cuda() / ((torch.sqrt(1 - scheduler.alpha[t_tensor].cuda())) *
                        (torch.sqrt(1 - scheduler.beta[t_tensor].cuda()))))
                z = (1 / (torch.sqrt(1 - scheduler.beta[t_tensor].cuda()))) * z - (temp * output)

                e = torch.randn(1, 1, 32, 32).cuda()
                z = z + (e * torch.sqrt(scheduler.beta[t_tensor].cuda()))

            # 마지막 타임스텝에서 생성된 이미지를 저장
            temp = scheduler.beta[0].cuda() / ((torch.sqrt(1 - scheduler.alpha[0].cuda())) * (torch.sqrt(1 - scheduler.beta[0].cuda())))
            final_image = (1 / (torch.sqrt(1 - scheduler.beta[0].cuda()))) * z - (temp * model(z, [0]).cuda())
            final_image = rearrange(final_image.squeeze(0), "c h w -> h w c").cpu().numpy()

            # 이미지의 차원을 (H, W, 3) 형태로 맞추기
            if final_image.shape[2] == 1:  # 채널이 1개인 경우
                final_image = np.repeat(final_image, 3, axis=2)

            # 0~1 사이로 정규화
            final_image = (final_image - final_image.min()) / (final_image.max() - final_image.min())
            generated_images.append(final_image)
            all_losses.append(losses)

    # Loss plot
    plt.figure(figsize=(12, 8))
    for i, losses in enumerate(all_losses):
        plt.plot(range(1, num_time_steps), losses, label=f'Image {i+1}')
    plt.xlabel('Time step')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss vs Time step for 10 generated images')
    plt.legend()
    plt.grid()
    plt.savefig('loss_vs_time_step.png')
    plt.close()

    return generated_images

def compute_fid_score(real_images_dir, generated_images_dir, batch_size=50):
    try:
        # FID 점수 계산
        print(f"Calculating FID with batch_size={batch_size}, real_images_dir={real_images_dir}, generated_images_dir={generated_images_dir}")
        result = subprocess.run(
            ['python', '-m', 'pytorch_fid', real_images_dir, generated_images_dir, '--batch-size', str(batch_size)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        output = result.stdout.decode('utf-8')
        fid_score = float(output.strip().split()[-1])
    except subprocess.CalledProcessError as e:
        print("Error occurred while calculating FID score:", e.stderr.decode('utf-8'))
        print(f"Command: {e.cmd}")
        raise RuntimeError("FID 계산 실패")
    except (ValueError, IndexError) as e:
        print("Error parsing FID score from output:", e)
        raise RuntimeError("FID 점수 추출 실패")
    
    return fid_score

def save_real_images(real_images_dir, num_images=100):
    # MNIST 데이터셋을 다운로드하고 로드
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(root='./data', train=True, download=True, transform=transform)

    # 실험적인 real 이미지 저장 폴더 생성
    os.makedirs(real_images_dir, exist_ok=True)

    # 첫 num_images개의 이미지를 real_images_dir에 저장
    for i in range(100, num_images):
        image, _ = dataset[i]  # 이미지를 얻음
        image = image.squeeze(0).numpy()  # 채널 차원을 제거하고 numpy로 변환
        plt.imsave(os.path.join(real_images_dir, f'real_image_{i}.png'), image, cmap='gray')
    print(f'Saved {num_images} real images to {real_images_dir}')

def save_generated_images(generated_images, generated_images_dir):
    # 디렉토리 생성 (존재하지 않으면 생성)
    os.makedirs(generated_images_dir, exist_ok=True)
    
    # 디렉토리에 이미 저장된 이미지 파일 이름들 확인
    existing_files = [f for f in os.listdir(generated_images_dir) if f.startswith('generated_image_') and f.endswith('.png')]
    
    # 기존 파일의 최대 번호를 확인
    if existing_files:
        existing_numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
        start_index = max(existing_numbers) + 1
    else:
        start_index = 0  # 디렉토리가 비어 있으면 0부터 시작
    
    # 생성된 이미지를 추가로 저장
    for idx, image in enumerate(generated_images):
        # 이미지 데이터 타입을 float32로 변환
        image = np.array(image, dtype=np.float32)
        
        # 새로운 이름 생성
        image_name = f'generated_image_{start_index + idx}.png'
        
        # 이미지 저장
        plt.imsave(os.path.join(generated_images_dir, image_name), image, cmap='gray')
    
    print(f'Saved {len(generated_images)} generated images to {generated_images_dir}, starting from index {start_index}.')

    
    
def calculate_weight_magnitude(model):
    """
    훈련된 모델의 weight magnitude를 계산합니다.
    
    Args:
        model (torch.nn.Module): 훈련된 모델
    
    Returns:
        dict: 각 레이어의 weight magnitude를 저장한 딕셔너리
    """
    weight_magnitudes = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:  # 가중치만 선택
            magnitude = param.abs().mean().item()  # 절대값의 평균 계산
            weight_magnitudes[name] = magnitude
    
    return weight_magnitudes

    
def plot_weight_magnitudes_separate(weight_magnitudes, save_path='up_down_weight_magnitude.png'):
    """
    weight magnitude를 down과 up 레이어로 나누어 시각화하고 저장합니다.
    
    Args:
        weight_magnitudes (dict): 각 레이어의 weight magnitude를 저장한 딕셔너리
        save_path (str): 저장할 파일 경로
    """
    layers = list(weight_magnitudes.keys())
    magnitudes = list(weight_magnitudes.values())

    # down과 up 레이어를 구분
    down_indices = [i for i, layer in enumerate(layers) if 'Layer1' in layer or 'Layer2' in layer or 'Layer3' in layer]
    up_indices = [i for i, layer in enumerate(layers) if 'Layer4' in layer or 'Layer5' in layer or 'Layer6' in layer]

    # down과 up의 weight magnitude 리스트
    down_magnitudes = [magnitudes[i] for i in down_indices]
    up_magnitudes = [magnitudes[i] for i in up_indices]
    down_layers = [layers[i] for i in down_indices]
    up_layers = [layers[i] for i in up_indices]

    plt.figure(figsize=(12, 6))

    # down 레이어는 빨간색
    plt.barh(down_layers, down_magnitudes, color='red', label='Down Layers')

    # up 레이어는 파란색
    plt.barh(up_layers, up_magnitudes, color='blue', label='Up Layers')

    plt.xlabel('Weight Magnitude')
    plt.ylabel('Layer')
    plt.title('Weight Magnitude per Layer (Up vs Down)')
    plt.legend()
    plt.grid(axis='x')
    plt.tight_layout()

    # 글씨 크기 조정
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=8)

    # 레이블 간격 조정을 위해 글씨가 겹치지 않도록 자동 조정
    plt.subplots_adjust(left=0.3)

    # plot을 파일로 저장
    plt.savefig(save_path)
    plt.close()
    print(f"Weight magnitude plot saved to {save_path}")


def plot_fisher_information_per_layer_weight_legend(fisher_information_per_timestep):
    """
    각 레이어의 weight별로 Fisher Information을 time step에 따라 시각화합니다.
    전체 레이어에 대해 y 축을 동일한 스케일로 설정하고, time step마다 독립된 선과 분간하기 쉬운 색상으로 표시합니다.
    """
    # Fisher 정보에서 첫 타임스텝의 데이터를 사용해 레이어 및 weight 이름을 추출
    example_timestep = list(fisher_information_per_timestep.keys())
    layer_names = list(fisher_information_per_timestep[example_timestep[0]].keys())

    # 각 레이어별로 서브플롯 생성
    fig, axs = plt.subplots(2, 3, figsize=(25, 15))
    axs = axs.ravel()

    # 구분하기 쉬운 색상 팔레트 설정
    color_map = plt.cm.get_cmap('tab20', len(fisher_information_per_timestep))
    colors = color_map(np.linspace(0, 1, len(fisher_information_per_timestep)))
    plt.gca().set_prop_cycle(cycler('color', colors))
    for idx, layer_name in enumerate(layer_names):
        ax = axs[idx]

        # 모든 타임스텝에 대해 해당 레이어의 weight 정보를 수집하여 플로팅
        for t_idx, (t, layer_info) in enumerate(fisher_information_per_timestep.items()):
            if layer_name in layer_info:
                
                weight_names = list(layer_info[layer_name].keys())
                fisher_values = [layer_info[layer_name][weight_name] for weight_name in weight_names]
            
                # 불필요한 부분 제거: "Layer#n."과 ".weight"
                simplified_weight_names = [name.replace(f"{layer_name}.", "").replace(".weight", "") for name in weight_names]

                # 각 time step을 독립적인 선으로 표현하고, 색상을 구분하기 쉬운 팔레트로 설정
                ax.plot(simplified_weight_names, fisher_values, marker='o', color=colors[t_idx], label=f'Timestep {t}')

        # 그래프 설정
        ax.set_xlabel(f'Weights in {layer_name}')
        ax.set_ylabel('Fisher Information Value')
        ax.set_title(f'Fisher Information for Weights in {layer_name}')
        # ax.set_ylim(0, global_max)  # 모든 플롯의 y축 스케일을 동일하게 설정
        ax.set_yscale('log')  # 로그 스케일 설정
        # ax.tick_params(axis='x', labelrotation=30)  # x축 레이블 각도 조정
        ax.tick_params(axis='x', labelsize=8)       # x축 레이블 폰트 크기 조정
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(-0.05, 1), fontsize='small')    
    plt.tight_layout(pad=5.0)
    plt.savefig('fisher_information_value_final_legend.png')
    plt.show()
    
def plot_fisher_information_per_layer_weight_gradation(fisher_information_per_timestep):
    """
    각 레이어의 weight별로 Fisher Information을 time step에 따라 시각화합니다.
    전체 레이어에 대해 y 축을 동일한 스케일로 설정하고, time step의 변화에 따라 그라데이션 색상을 적용합니다.
    """
    # Fisher 정보에서 첫 타임스텝의 데이터를 사용해 레이어 및 weight 이름을 추출
    example_timestep = list(fisher_information_per_timestep.keys())
    layer_names = list(fisher_information_per_timestep[example_timestep[0]].keys())

    # 각 레이어별로 서브플롯 생성
    fig, axs = plt.subplots(2, 3, figsize=(25, 15))
    axs = axs.ravel()

    for idx, layer_name in enumerate(layer_names):
        ax = axs[idx]
        color_map = plt.cm.viridis(np.linspace(0, 1, len(fisher_information_per_timestep)))
        
        # 모든 타임스텝에 대해 해당 레이어의 weight 정보를 수집하여 플로팅
        for t_idx, (t, layer_info) in enumerate(fisher_information_per_timestep.items()):
            if layer_name in layer_info:
                weight_names = list(layer_info[layer_name].keys())
                fisher_values = [layer_info[layer_name][weight_name] for weight_name in weight_names]
                # 불필요한 부분 제거: "Layer#n."과 ".weight"
                simplified_weight_names = [name.replace(f"{layer_name}.", "").replace(".weight", "") for name in weight_names]

                ax.plot(simplified_weight_names, fisher_values, marker='o', color=color_map[t_idx], label=f'Timestep {t}' if t_idx == 0 else "")

        # 그래프 설정
        ax.set_xlabel(f'Weights in {layer_name}')
        ax.set_ylabel('Fisher Information Value')
        ax.set_title(f'Fisher Information for Weights in {layer_name}')
        ax.set_yscale('log')  # 로그 스케일 설정
        ax.tick_params(axis='x', labelrotation=30)  # x축 레이블 각도 조정
        ax.tick_params(axis='x', labelsize=8)       # x축 레이블 폰트 크기 조정
        ax.grid(True)

    
    plt.tight_layout(pad=5.0)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # 하단 컬러바 여백 확보를 위해 조정
    # 컬러바 범위 설정
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])  # [left, bottom, width, height]
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), cax=cbar_ax, label='Timestep Progression', orientation='horizontal')
    
    
    plt.savefig(f'fisher_information_value_final_gradation.png')

def apply_fisher_threshold_and_half_weights(model, fisher_information, threshold=1.0):
    
    with torch.no_grad():
        for layer_name, weights_info in fisher_information.items():
            for weight_name, fisher_value in weights_info.items():
                if fisher_value < threshold:
                    # 모델에서 해당 weight를 찾아 half precision으로 변환
                    for name, param in model.named_parameters():
                        if name == weight_name:
                            # print(f"Applying half precision to {name} with Fisher value {fisher_value} at timestep {timestep}")
                            param.data = param.data.half()  # 데이터를 절반 정밀도(float16)로 변환
                            break
    return model

def calculate_fisher_information_for_plot(model, z, scheduler, num_time_steps, per=50): # timestep 정규화 없이 수행
    
    fisher_information_per_timestep = {1: {}}
    fisher_information_per_timestep.update({t: {} for t in range(per, num_time_steps, per)})
    criterion = nn.MSELoss()
    model.eval()

    for t in tqdm(reversed(range(1, num_time_steps)), desc='Calculating Fisher Information per Timestep'):
        t_tensor = [t]
        
        # `per` 간격마다만 Fisher 정보를 저장하므로 이때만 `z`를 새로 복사하여 `requires_grad=True` 설정
        if  t % per == 0 or t == 1: # t % per == 0 or 
            z_temp = z.detach().clone().requires_grad_(True)
            model.zero_grad()
        else:
            z_temp = z.detach()  # Fisher 정보 계산 필요 없을 때는 `requires_grad=False`

        if t % per == 0 or t == 1 :
            output = model(z_temp, t_tensor)
        else:
            with torch.no_grad():
                output = model(z_temp, t_tensor)
            

        # `per` 간격마다 손실 및 Fisher 정보 계산
        if  t % per == 0 or t == 1: # t % per == 0 or 
            # 손실 계산 및 그래디언트 계산
            loss = criterion(output, z_temp)
            loss.backward(retain_graph=False)
            z_temp.grad = None
            torch.cuda.empty_cache()


            # 각 레이어와 weight별로 Fisher 정보 저장
            layer_fisher_values = {}
            for name, param in model.named_parameters():
                if 'Layer' in name and 'weight' in name and param.requires_grad:
                    layer_name = name.split('.')[0]
                    weight_name = name

                    # Fisher 정보 계산
                    fisher_value = param.grad.data.pow(2).mean().item() # 각 weight 값의 제곱의 평균
                    layer_fisher_values[(layer_name, weight_name)] = fisher_value

            # Fisher 정보 계산 후 그래디언트 해제
            z_temp.grad = None
            model.zero_grad()
        torch.cuda.empty_cache()
        # 노이즈 업데이트
        temp = (scheduler.beta[t_tensor].cuda() / ((torch.sqrt(1 - scheduler.alpha[t_tensor].cuda())) *
                                                   (torch.sqrt(1 - scheduler.beta[t_tensor].cuda()))))
        z = (1 / (torch.sqrt(1 - scheduler.beta[t_tensor].cuda()))) * z - (temp * output)
        torch.cuda.empty_cache()
    return fisher_information_per_timestep

def calculate_fisher_information(model, z, scheduler, num_time_steps, per=50):
    
    fisher_information_per_timestep = {1: {}}
    fisher_information_per_timestep.update({t: {} for t in range(per, num_time_steps, per)})
    criterion = nn.MSELoss()
    model.eval()

    for t in tqdm(reversed(range(1, num_time_steps)), desc='Calculating Fisher Information per Timestep'):
        t_tensor = [t]
        
        # `per` 간격마다만 Fisher 정보를 저장하므로 이때만 `z`를 새로 복사하여 `requires_grad=True` 설정
        if  t % per == 0 or t == 1:
            z_temp = z.detach().clone().requires_grad_(True)
            model.zero_grad()
        else:
            z_temp = z.detach()  # Fisher 정보 계산 필요 없을 때는 `requires_grad=False`

        if t % per == 0 or t == 1 :
            output = model(z_temp, t_tensor)
        else:
            with torch.no_grad():
                output = model(z_temp, t_tensor)
            

        # `per` 간격마다 손실 및 Fisher 정보 계산
        if  t % per == 0 or t == 1: # t % per == 0 or 
            # 손실 계산 및 그래디언트 계산
            loss = criterion(output, z_temp)
            loss.backward(retain_graph=False)
            z_temp.grad = None
            torch.cuda.empty_cache()


            # 각 레이어와 weight별로 Fisher 정보 저장
            layer_fisher_values = {}
            for name, param in model.named_parameters():
                if 'Layer' in name and 'weight' in name and param.requires_grad:
                    layer_name = name.split('.')[0]
                    weight_name = name

                    # Fisher 정보 계산
                    fisher_value = param.grad.data.pow(2).mean().item() # 각 weight 값의 제곱의 평균
                    layer_fisher_values[(layer_name, weight_name)] = fisher_value

            # 해당 timestep에서 최대값 구하기
            max_fisher_value = max(layer_fisher_values.values(), default=1.0)  # max가 0일 때 대비하여 기본값 1.0 설정

            #! Fisher 정보를 정규화하여 저장 - timestep 별 정규화
            for (layer_name, weight_name), fisher_value in layer_fisher_values.items():
                if layer_name not in fisher_information_per_timestep[t]:
                    fisher_information_per_timestep[t][layer_name] = {}
                fisher_information_per_timestep[t][layer_name][weight_name] = fisher_value / max_fisher_value

            # Fisher 정보 계산 후 그래디언트 해제
            z_temp.grad = None
            model.zero_grad()
        torch.cuda.empty_cache()
        # 노이즈 업데이트
        temp = (scheduler.beta[t_tensor].cuda() / ((torch.sqrt(1 - scheduler.alpha[t_tensor].cuda())) *
                                                   (torch.sqrt(1 - scheduler.beta[t_tensor].cuda()))))
        z = (1 / (torch.sqrt(1 - scheduler.beta[t_tensor].cuda()))) * z - (temp * output)
        torch.cuda.empty_cache()
    return fisher_information_per_timestep

# weight 파라미터 크기 분석
def analyze_weight_size(ckpt_path) :
    checkpoint = torch.load(ckpt_path)
    # 파라미터 개수와 용량, 자료형 확인
    total_params = 0
    total_size_in_bytes = 0
    param_dtypes = {}

    for param_name, param_tensor in checkpoint['weights'].items():
        param_count = param_tensor.numel()          # 파라미터 개수
        param_size = param_tensor.element_size()    # 각 파라미터의 바이트 크기
        param_dtype = param_tensor.dtype            # 자료형
        
        # 개수, 용량 합산
        total_params += param_count
        total_size_in_bytes += param_count * param_size
        
        # 자료형 별로 개수를 저장
        if param_dtype in param_dtypes:
            param_dtypes[param_dtype] += param_count
        else:
            param_dtypes[param_dtype] = param_count

    # 결과 출력
    print(f"Total Parameter Count: {total_params}")
    print(f"Total Size in MB: {total_size_in_bytes / (1024 ** 2):.2f} MB")
    print("Parameter Data Types and Counts:")
    for dtype, count in param_dtypes.items():
        print(f"  {dtype}: {count}")



def main():
    timesteps = 1000
    per=5 # 시각화 분석을 위하여 몇 타임스텝 주기로 fisher information 을 수집할 것인지 결정
    num_images = 5000 # 생성할 이미지 갯수
    cut = 1
    
    # 비교군 담을 fid 배열
    fids = []


    # 실제 이미지와 생성된 이미지 폴더 설정
    real_images_dir = './real_images'
    
    generated_images_dir = f'./generated_images' # 원본 모델에서 생성한 이미지

    # 실제 이미지를 저장 (MNIST 데이터셋 사용)
    save_real_images(real_images_dir, num_images=5000) # 한번만 실행해도 됨
    
    # 원래 model 실행
    original_ckpt_path = 'checkpoints/ddpm_checkpoint'    

    #! 비교군 1 - 원본 모델
    # 모델을 사용해 이미지를 생성
    generated_images, times = inference(original_ckpt_path, num_time_steps = timesteps, num_images = num_images) # 처음 추론

    # 생성된 이미지를 저장
    os.makedirs(generated_images_dir, exist_ok=True)
    save_generated_images(generated_images, generated_images_dir)
    
    checkpoint = torch.load(original_ckpt_path)
    model = UNET().cuda()
    scheduler = DDPM_Scheduler()
    z = torch.randn(1, 1, 32, 32).cuda()
    
    # FID 점수 계산
    fid_score = compute_fid_score(real_images_dir, generated_images_dir)
    fids.append(fid_score)
    
    #! 비교군 2 - 전체 양자화 
    fisher_information = calculate_fisher_information(model, z, scheduler, num_time_steps=timesteps, per=5)
    
    # fisher information 시각화 코드
    calculate_fisher_information_for_plot = calculate_fisher_information(model, z, scheduler, num_time_steps=timesteps, per=5)
    plot_fisher_information_per_layer_weight_gradation(calculate_fisher_information_for_plot)
    
    threshold = 10000000000000000000000000000
    quant_generated_images_dir = 'quant_whole' # 이미지 저장할 곳
    os.makedirs(quant_generated_images_dir, exist_ok=True)
    
    quant_ckpt_dir = 'quant_whole_ckpt' # 모델 저장할 곳
    os.makedirs(quant_ckpt_dir, exist_ok=True)
    save_path = os.path.join(quant_ckpt_dir, 'quant_whole.pt')
    
    # print(fisher_information)
    # print(fisher_information[1])
    
    quantized_model = apply_fisher_threshold_and_half_weights(model, fisher_information=fisher_information[cut], threshold=threshold)
    torch.save({'weights': quantized_model.state_dict(), 'ema': checkpoint['ema']}, save_path)
    
    # 모델을 사용해 이미지를 생성
    generated_images, times = inference(save_path, num_time_steps = timesteps, num_images = num_images)

    # 생성된 이미지 저장
    save_generated_images(generated_images, quant_generated_images_dir)

    # FID 점수 계산
    fid_score = compute_fid_score(real_images_dir, quant_generated_images_dir)
    fids.append(fid_score)

    print('-------------------original-------------------')
    print(f'quant FID Score: {fids[0]}')
    analyze_weight_size(original_ckpt_path)
    
    
    print('------------------whole quantized-------------------')
    print(f'quant FID Score: {fids[1]}')
    analyze_weight_size(save_path)
    
        
if __name__ == '__main__':
    # FutureWarning을 무시
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()