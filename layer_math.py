from simplediffusion import *
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os


def calculate_fisher(model, z, scheduler, num_time_steps, per=50, cut=1):
    
    fisher_information_per_timestep = {cut: {}}
    criterion = nn.MSELoss()
    model.eval()

    for t in tqdm(reversed(range(1, num_time_steps)), desc='Calculating Fisher Information per Timestep'):
        t_tensor = [t]
        
        # `per` 간격마다만 Fisher 정보를 저장하므로 이때만 `z`를 새로 복사하여 `requires_grad=True` 설정
        if t == cut: 
            torch.cuda.synchronize()  # GPU 동기화
            torch.cuda.empty_cache()
            z_temp = z.detach().clone().requires_grad_(True)
            model.zero_grad()
            
        else:
            z_temp = z.detach()  # Fisher 정보 계산 필요 없을 때는 `requires_grad=False`
            torch.cuda.empty_cache()

        if t > cut + 1 :
            with torch.no_grad():
                output = model(z_temp, t_tensor)
        else:
            output = model(z_temp, t_tensor)
        torch.cuda.empty_cache()
        if t == cut:
            loss = criterion(output, z_temp)
            loss.backward()
            torch.cuda.empty_cache()  # 그래디언트 계산 후 캐시 정리

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
            torch.cuda.empty_cache()  # 그래디언트 계산 후 캐시 정리

        # 노이즈 업데이트
        temp = (scheduler.beta[t_tensor].cuda() / ((torch.sqrt(1 - scheduler.alpha[t_tensor].cuda())) *
                                                   (torch.sqrt(1 - scheduler.beta[t_tensor].cuda()))))
        z = (1 / (torch.sqrt(1 - scheduler.beta[t_tensor].cuda()))) * z - (temp * output)

    return fisher_information_per_timestep


#! 분포 저장 함수
def plot_weight_distribution(model, save_path="layerwise_weight_histograms_separate.png"):

    # 각 레이어의 가중치를 저장할 딕셔너리 초기화
    layer_weight_data = {f"Layer{i}": {} for i in range(1, 7)}

    # 모델의 가중치 이름과 값을 순회
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            for i in range(1, 7):  # Layer1 ~ Layer6 검사
                if f"Layer{i}" in name:
                    layer_weight_data[f"Layer{i}"][name] = param.detach().cpu().numpy().flatten()

    # 서브플롯 설정 (2행 x 3열)
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs = axs.ravel()  # 2D 배열의 축을 1D로 변환

    # 구분하기 쉬운 색상 팔레트 설정 (seaborn의 컬러 팔레트 사용)
    color_palette = sns.color_palette("husl", n_colors=20)

    for idx, (layer_name, weight_dict) in enumerate(layer_weight_data.items()):
        if weight_dict:  # 데이터가 존재하는 레이어만 처리
            ax = axs[idx]
            for color_idx, (weight_name, weights) in enumerate(weight_dict.items()):
               
                ax.hist(
                    weights, bins=50, alpha=0.6,
                    color=color_palette[color_idx % len(color_palette)], histtype='stepfilled', label=weight_name
                )
            ax.set_title(f"Weight Distribution: {layer_name}", fontsize=10)
            ax.set_xlabel("Weight Value", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.set_yscale('log')
            ax.grid(True)
            ax.legend(fontsize=7, loc='upper right')

    # 빈 서브플롯 숨기기
    for idx in range(len(layer_weight_data), len(axs)):
        axs[idx].axis('off')

    # 전체 레이아웃 조정 및 저장
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Layer-wise distinct-colored weight histograms saved to {save_path}")
    
def calculate_layerwise_fisher_statistics(fisher_information, timestep=1):
   
    layer_statistics = {}

    if timestep not in fisher_information:
        raise ValueError(f"Timestep {timestep} not found in fisher_information")

    for layer_name, weights_info in fisher_information[timestep].items():
        fisher_values = torch.tensor(list(weights_info.values()))
        layer_mean = fisher_values.mean().item()
        layer_std = fisher_values.std().item()
        layer_statistics[layer_name] = {"mean": layer_mean, "std": layer_std}

    return layer_statistics # layer 별 mean, std

def apply_layerwise_fisher_quantization(model, fisher_information, layer_statistics, k=1.5, timestep=1, alpha=0.5):
   
    with torch.no_grad():
        timestep = timestep
        for layer_name, weights_info in fisher_information[timestep].items():
            if layer_name not in layer_statistics:
                continue

            # Layer별 mean과 std 가져오기
            layer_mean = layer_statistics[layer_name]["mean"]
            layer_std = layer_statistics[layer_name]["std"]

            # Adaptive k 값 계산
            if layer_std == 0:  # 분산이 0인 경우
                k = 1.0
            else:
                k = (layer_mean / layer_std) * alpha

            # k 값 제한
            k = max(0.1, min(k, 2.0))  # 0.1 ≤ k ≤ 2.0

            # Threshold 계산
            threshold_min = layer_mean - k * layer_std # 평균, 표준편차로 계산, 이 때 k는 
            threshold_max = layer_mean + k * layer_std

            print(f"Layer {layer_name}: mean={layer_mean:.6e}, std={layer_std:.6e}, "
                  f"k={k:.2f}, threshold_min={threshold_min:.6e}, threshold_max={threshold_max:.6e}")

            # Weight에 양자화 및 Threshold 적용
            for weight_name, fisher_value in weights_info.items():
                if 'gnorm' in weight_name:
                    continue
                for name, param in model.named_parameters():
                    if name == weight_name:
                        # Threshold 범위에 따라 weight 클리핑
                        param.data = torch.clamp(param.data, threshold_min, threshold_max)

                        # Fisher 값에 따라 양자화
                        if fisher_value < threshold_min:
                            param.data = param.data.half()
                        else:
                            param.data = param.data.float()
                        break

    return model


def main():
    
    timesteps = 1000
    per = 1000
    cut = 1
    
    real_images_dir = './real_images'
    cut = 1
    category='layer_math'
    
    # layer_math 매개변수 설정
    k=1
    alpha=0.5
    num_images = 5000 # 생성할 이미지 갯수
    
    quant_generated_images_dir = f'images/{category}/adaptive_k_{k}_alpha_{alpha}' # 이미지 저장할 곳
    os.makedirs(quant_generated_images_dir, exist_ok=True)
    
    quant_ckpt_dir = f'ckpt/{category}_ckpt' # 모델 저장할 곳
    os.makedirs(quant_ckpt_dir, exist_ok=True)
    save_path = os.path.join(quant_ckpt_dir, f'adaptive_k_{k}_alpha_{alpha}.pt')
    
    
    # fisher 추출
    original_ckpt_path = 'checkpoints/ddpm_checkpoint'
    checkpoint = torch.load(original_ckpt_path)
    model = UNET().cuda()
    
    scheduler = DDPM_Scheduler()
    z = torch.randn(1, 1, 32, 32).cuda()
    
    
    fisher_information = calculate_fisher_information(model, z, scheduler, num_time_steps=timesteps, per=per)
    for timestep, layers in fisher_information.items():
        if timestep == cut:
            layer_statistics = calculate_layerwise_fisher_statistics(fisher_information, timestep=cut)
            quantized_model = apply_layerwise_fisher_quantization(model, fisher_information, layer_statistics, k=k, alpha=alpha)
            torch.save({'weights': quantized_model.state_dict(), 'ema': checkpoint['ema']}, save_path)
            break
        
        
    # 모델을 사용해 이미지를 생성
    generated_images, times = inference(save_path,  num_time_steps = timesteps, num_images = num_images)
    save_generated_images(generated_images, quant_generated_images_dir)
   
   
    fid_score = compute_fid_score(real_images_dir, quant_generated_images_dir)
    print('-------------------layer_math-------------------')
    print(f'quant FID Score: {fid_score}')
    analyze_weight_size(save_path)
    
    
    
if __name__ == '__main__':  
    # FutureWarning을 무시
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()