from simplediffusion import *
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

def main():
    timesteps = 1000
    num_images = 5000 # 생성할 이미지 갯수
    cut = 1

    # 실제 이미지와 생성된 이미지 폴더 설정
    real_images_dir = './real_images'
    
    # 원래 model 실행
    original_ckpt_path = 'checkpoints/ddpm_checkpoint'    

    
    checkpoint = torch.load(original_ckpt_path)
    model = UNET().cuda()
    scheduler = DDPM_Scheduler()
    z = torch.randn(1, 1, 32, 32).cuda()
    
    fisher_information = calculate_fisher_information(model, z, scheduler, num_time_steps=timesteps, per=5)
    
    # fisher information 시각화 코드
    plot_fisher_information_per_layer_weight_gradation(fisher_information)
    
    threshold = 1
    quant_generated_images_dir = f'images/quant_threshold/{threshold}' # 이미지 저장할 곳
    os.makedirs(quant_generated_images_dir, exist_ok=True)
    
    quant_ckpt_dir = f'ckpt/quant_threshold_{threshold}_ckpt' # 모델 저장할 곳
    os.makedirs(quant_ckpt_dir, exist_ok=True)
    save_path = os.path.join(quant_ckpt_dir, f'threshold_{threshold}.pt')
    
    
    quantized_model = apply_fisher_threshold_and_half_weights(model, fisher_information=fisher_information[cut], threshold=threshold)
    torch.save({'weights': quantized_model.state_dict(), 'ema': checkpoint['ema']}, save_path)
    
    # 모델을 사용해 이미지를 생성
    generated_images, times = inference(save_path, num_time_steps = timesteps, num_images = num_images)

    # 생성된 이미지 저장
    save_generated_images(generated_images, quant_generated_images_dir)

    # FID 점수 계산
    fid = compute_fid_score(real_images_dir, quant_generated_images_dir)
    
    print(f'-------------------quantized threshold {threshold}-------------------')
    print(f'quant FID Score: {fid}')
    analyze_weight_size(save_path)

    
        
if __name__ == '__main__':
    # FutureWarning을 무시
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()