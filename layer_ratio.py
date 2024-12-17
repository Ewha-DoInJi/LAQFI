from simplediffusion import *
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

def apply_fp16_quantization(model, fisher_information_per_timestep, threshold_ratio):
    """
    Fisher 정보에 기반하여 timestep 1의 레이어별로 주어진 비율의 threshold 값을 적용하여 
    weight를 절반으로 양자화한 모델을 반환합니다.

    Args:
        model (torch.nn.Module): 모델.
        fisher_information_per_timestep (dict): 각 time step과 레이어의 weight별 Fisher 정보 값을 포함한 딕셔너리.
        threshold_ratio (float): 임계값 비율 (0.0 ~ 1.0)로, 하위 비율의 Fisher 정보를 걸러내기 위한 값.

    Returns:
        torch.nn.Module: 양자화된 모델.
    """
    # Fisher 정보에서 timestep 1만 사용
    timestep_1_info = fisher_information_per_timestep.get(1, {})
    
    # Fisher 값 기반으로 threshold 계산
    thresholds = {}
    for layer_name, weights_info in timestep_1_info.items():
        fisher_values = list(weights_info.values())  # 모든 weight의 Fisher 값 가져오기
        fisher_values.sort(reverse=True)  # 내림차순 정렬
        cutoff_index = int(len(fisher_values) * threshold_ratio)  # 비율에 따라 cutoff 계산
        threshold_value = fisher_values[cutoff_index] if cutoff_index < len(fisher_values) else fisher_values[-1]
        thresholds[layer_name] = threshold_value  # 각 레이어의 threshold 저장

    # 양자화 적용
    with torch.no_grad():
        quantized_count = {}  # 각 레이어의 양자화된 가중치 개수를 저장
        
        for layer_name, threshold_value in thresholds.items():
            quantized_count[layer_name] = 0  # 초기화
            for weight_name, fisher_value in timestep_1_info[layer_name].items():
                print(f'        layer_name = {layer_name}, weight_name={weight_name} fisher_value = {fisher_value} / threshold_value = {threshold_value}')
                if fisher_value < threshold_value:  # threshold보다 작은 weight만 양자화
                    for name, param in model.named_parameters():
                        if name == weight_name:  # weight 이름이 일치하면
                            print(f"Applying half to {name} with Fisher value {fisher_value} (Threshold: {threshold_value})")
                            param.data = param.data.half()  # FP16으로 변환
                            quantized_count[layer_name] += 1  # 양자화된 weight 개수 증가
                            break
        # 각 레이어의 양자화된 weight 개수 출력
        print("\nQuantization Summary:")
        for layer_name, count in quantized_count.items():
            print(f"  Layer {layer_name}: Quantized {count} weights")
        
    return model

# results 값을 CSV 파일로 저장
def save_a_result_to_csv(results, filename="./quant-fp16-final-results.csv"):
    """
    단일 행 형태의 결과 데이터를 CSV 파일에 저장.
    
    Args:
        results (list): 단일 행 데이터 (예: [threshold_ratio, elapsed_time, quant_fid_score, quant_mean_is, quant_std_is]).
        filename (str): 저장할 파일 이름 (기본값: "results.csv").
    """
    headers = ["threshold_ratio", "elapsed_time", "quant_fid_score", "quant_mean_is", "quant_std_is"]
    file_exists = os.path.isfile(filename)  # 파일 존재 여부 확인

    # 파일 열기 (쓰기 모드 또는 추가 모드)
    with open(filename, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        
        # 파일이 없는 경우에만 헤더 작성
        if not file_exists:
            writer.writerow(headers)
        
        # 데이터를 행 단위로 추가
        writer.writerow(results)

    print(f"Results appended to {filename}" if file_exists else f"Results saved to {filename}")
    
def main():
    timesteps = 1000
    num_images = 5000 # 생성할 이미지 갯수
    
    
    timesteps = 1000
    per = 1000
    cut = 1
    
    real_images_dir = './real_images'
    cut = 1
    category='layer_ratio'

    # simple diffusion 모델 실행
    original_ckpt_path = 'checkpoints/ddpm_checkpoint'
    # original_checkpoint = torch.load(original_ckpt_path)
    
    ##### 양자화 모델 생성 및 저장
    threshold_ratio = 0.15
        
    quant_generated_images_dir = f'images/{category}/{threshold_ratio}' # 이미지 저장할 곳
    os.makedirs(quant_generated_images_dir, exist_ok=True)
    
    quant_ckpt_dir = f'ckpt/{category}_ckpt' # 모델 저장할 곳
    os.makedirs(quant_ckpt_dir, exist_ok=True)
    save_path = os.path.join(quant_ckpt_dir, f'{threshold_ratio}.pt')
    
    
    # fisher 추출
    original_ckpt_path = 'checkpoints/ddpm_checkpoint'
    model = UNET().cuda()
    
    scheduler = DDPM_Scheduler()
    z = torch.randn(1, 1, 32, 32).cuda()
    #####
    
    original_checkpoint = torch.load(original_ckpt_path)
    model = UNET().cuda()
    scheduler = DDPM_Scheduler()
    z = torch.randn(1, 1, 32, 32).cuda()
    fisher_information = calculate_fisher_information(model, z, scheduler, num_time_steps=timesteps)
    

    quantized_model = apply_fp16_quantization(model, fisher_information, threshold_ratio=threshold_ratio)
    torch.save({'weights': quantized_model.state_dict(), 'ema': original_checkpoint['ema']}, save_path)

    # 양자화 이미지 생성 및 저장
    generated_images, time = inference(save_path, num_time_steps = timesteps, num_images = num_images)
    os.makedirs(quant_generated_images_dir, exist_ok=True)
    save_generated_images(generated_images, quant_generated_images_dir)


    fid_score = compute_fid_score(real_images_dir, quant_generated_images_dir, batch_size=50)
    print('-------------------layer ratio-------------------')
    print(f'quant FID Score: {fid_score}')
    analyze_weight_size(save_path)

    
if __name__ == '__main__':
    # FutureWarning을 무시
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()