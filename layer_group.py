from simplediffusion import *
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os




# =========================================================
# Helper Functions
# =========================================================
def initialize_fisher_information_structures(cut_time_step, group_config):
    """
    Initialize Fisher information structures based on group configuration.
    """
    mean_fisher_information = {cut_time_step: {group: {} for group in group_config}}
    group_count = {cut_time_step: {group: {} for group in group_config}}
    return mean_fisher_information, group_count


def update_fisher_information(mean_fisher_info, group_count, t, group, layer_name, fisher_value):
    """
    Update Fisher information and group count for a given layer.
    """
    if layer_name not in mean_fisher_info[t][group]:
        mean_fisher_info[t][group][layer_name] = 0.0
        group_count[t][group][layer_name] = 0
    mean_fisher_info[t][group][layer_name] += fisher_value
    group_count[t][group][layer_name] += 1


def compute_mean_fisher_information(mean_fisher_info, group_count, t):
    """
    Compute mean Fisher information by dividing total Fisher values by count.
    """
    for group_name in mean_fisher_info[t]:
        for layer_name in mean_fisher_info[t][group_name]:
            fisher_value = mean_fisher_info[t][group_name][layer_name]
            fisher_count = group_count[t][group_name][layer_name]
            mean_fisher_info[t][group_name][layer_name] = fisher_value / fisher_count


# =========================================================
# mean fisher information 구하기
# =========================================================
def calculate_mean_fisher_information_per_timestep_by_group(
    model, z, scheduler, num_time_steps, group_config, cut_time_step=1, per=50
):
    """
    Calculate Fisher information per timestep by group, dynamically determined by group_config.
    """
    criterion = nn.MSELoss()
    model.eval()

    # Initialize Fisher information structures based on group config
    mean_fisher_info, group_count = initialize_fisher_information_structures(cut_time_step, group_config)

    for t in tqdm(reversed(range(1, num_time_steps)), desc='Calculating Fisher Information'):
        t_tensor = [t]
        
        # `z` initialization for grad calculation
        z_temp = z.detach().clone().requires_grad_(t == cut_time_step)  # Only require gradient at cut_time_step
        model.zero_grad()

        # Forward pass and loss calculation
        with torch.no_grad() if t > cut_time_step + 1 else torch.enable_grad():
            output = model(z_temp, t_tensor)
        loss = criterion(output, z_temp) if t == cut_time_step else None

        if t == cut_time_step:
            loss.backward()

            # Update Fisher information for each layer and group
            for name, param in model.named_parameters():
                if 'Layer' in name and 'weight' in name and param.requires_grad:
                    layer_name = name.split('.')[0]
                    group = assign_group(layer_name, group_config)

                    fisher_value = torch.mean(param**2).item()
                    update_fisher_information(mean_fisher_info, group_count, t, group, layer_name, fisher_value)

            compute_mean_fisher_information(mean_fisher_info, group_count, t)

        # Noise update step
        temp = scheduler.beta[t_tensor].cuda() / (
            torch.sqrt(1 - scheduler.alpha[t_tensor].cuda()) *
            torch.sqrt(1 - scheduler.beta[t_tensor].cuda())
        )
        z = z / torch.sqrt(1 - scheduler.beta[t_tensor].cuda()) - (temp * output)

    return mean_fisher_info


# =========================================================
# 레이어가 속한 그룹 찾기
# =========================================================
def assign_group(layer_name, group_config):
    """
    Assign layers to specific groups based on dynamic group configuration.
    """
    for group, layers in group_config.items():
        if any(layer_name.startswith(layer) for layer in layers):
            return group
    return 'unknown'  # Default group if not matched


# =========================================================
# threshold 그룹별 양자화
# =========================================================
def apply_threshold_to_model(model, fisher_information, threshold_group):
    """
    Apply thresholding for quantizing model weights based on Fisher information.
    """
    with torch.no_grad():
        for group_name, weights_info in fisher_information.items():
            
            # 현재 layer가 속한 그룹의 threshold 가져오기
            threshold = threshold_group[group_name]
            
            for weight_name, fisher_value in weights_info.items():
                ## threshold (평균) 보다 아래면 양자화로 날리기
                if fisher_value < threshold:
                    # fisher에서 구한 weight_name과 일치한 모델 파라미터명찾기
                    # bias가 아닌 가중치만 양자화
                    for name, param in model.named_parameters():
                        #print(name)
                        if name == weight_name:
                            param.data = param.data.half()       # 가중치 반으로 양자화
                            break
    return model

# =========================================================
# threshold 평균 구하기
# =========================================================
def calculate_threshold(mean_fisher_info, t):
    """
    Calculate threshold values for Fisher information by group.
    """
    threshold_group = {}
    for group_name in mean_fisher_info[t]:
        fisher_value = sum(mean_fisher_info[t][group_name].values())
        layer_count = len(mean_fisher_info[t][group_name])
        threshold_group[group_name] = fisher_value / layer_count if layer_count > 0 else 0
    return threshold_group


# =========================================================
# Group별 threshold 평균 구하기 최종 함수 (main함수에 작성해도 됨)
# =========================================================
def get_threshold_group(model, z, scheduler, num_time_steps, checkpoint, cut_time_step=1, group_config=None):
    # Calculate Fisher information
    mean_fisher_info = calculate_mean_fisher_information_per_timestep_by_group(
        model=model, z=z, scheduler=scheduler, num_time_steps=num_time_steps,
        group_config=group_config, cut_time_step=cut_time_step
    )
    
    # Calculate threshold values for quantization
    threshold_group = calculate_threshold(mean_fisher_info, cut_time_step)
    
    return threshold_group





# =========================================================
# 전체 fisher 구하기
# =========================================================
def calculate_fisher_information_per_timestep_by_layer(model, z, scheduler, num_time_steps, per=50, cut_time_step=1, group_config=None):
    # Fisher 정보 초기화
    fisher_information_per_timestep = {cut_time_step: {key: {} for key in group_config.keys()}} # 그룹에 따른 초기화
    
    fisher_information_per_timestep.update({t: {} for t in range(per, num_time_steps, per)})
    criterion = nn.MSELoss()
    model.eval()

    # 그룹별 레이어를 기반으로 Fisher 정보 계산
    for t in tqdm(reversed(range(1, num_time_steps)), desc='Calculating Fisher Information per Timestep'):
        t_tensor = [t]

        # `per` 간격마다만 Fisher 정보를 저장하므로 이때만 `z`를 새로 복사하여 `requires_grad=True` 설정
        if t % per == 0 or t == 1:  # t % per == 0 or 
            z_temp = z.detach().clone().requires_grad_(True)
            model.zero_grad()
        else:
            z_temp = z.detach()  # Fisher 정보 계산 필요 없을 때는 `requires_grad=False`

        if t % per == 0 or t == 1:
            output = model(z_temp, t_tensor)
        else:
            with torch.no_grad():
                output = model(z_temp, t_tensor)

        # `per` 간격마다 손실 및 Fisher 정보 계산
        if t % per == 0 or t == 1:
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

                    # group_config에 맞춰 group_name 설정
                    group_name = get_group_name_from_config(layer_name, group_config)

                    # Fisher 정보 계산
                    fisher_value = param.grad.data.pow(2).mean().item()  # 각 weight 값의 제곱의 평균
                    
                    # 그룹별로 Fisher 값 저장
                    layer_fisher_values[(group_name, weight_name)] = fisher_value

            # 해당 timestep에서 최대값 구하기
            max_fisher_value = max(layer_fisher_values.values(), default=1.0)  # max가 0일 때 대비하여 기본값 1.0 설정

            # Fisher 정보를 정규화하여 저장 - timestep 별 정규화
            for (group_name, weight_name), fisher_value in layer_fisher_values.items():
                if group_name not in fisher_information_per_timestep[t]:
                    fisher_information_per_timestep[t][group_name] = {}
                fisher_information_per_timestep[t][group_name][weight_name] = fisher_value / max_fisher_value

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

def get_group_name_from_config(layer_name, group_config):
    """
    주어진 레이어 이름을 기반으로 group_config에서 적절한 group_name을 반환하는 함수.
    """
    for group_name, layers in group_config.items():
        if layer_name in layers:
            return group_name
    return 'unknown'  # 그룹에 속하지 않는 레이어는 'unknown'으로 처리

    
def main():
    """
    Main function to calculate Fisher information, compute thresholds, and apply quantization.
    """
    
    timesteps = 1000
    per = 1000
    cut_time_step = 1
    
    real_images_dir = './real_images'
    cut = 1
    category='layer_group'
    num_images = 5000 # 생성할 이미지 갯수
    
    ## 여기서 레이어 그룹 지정하기!
    group_config = {
        'uplayer': ['Layer6'],
        'middlelayer': ['Layer4', 'Layer5'],
        'downlayer': ['Layer1', 'Layer2', 'Layer3']
    }
    """
    group_config = {
        'uplayer': ['Layer4', 'Layer5', 'Layer6'],
        'downlayer': ['Layer1', 'Layer2', 'Layer3']
    } 
    """
    """
    group_config = {
        'uplayer': ['Layer3', 'Layer4', 'Layer5', 'Layer6'],
        'downlayer': ['Layer1', 'Layer2']
    }
    """
    
    quant_generated_images_dir = f'images/{category}/{group_config}' # 이미지 저장할 곳
    os.makedirs(quant_generated_images_dir, exist_ok=True)
    
    quant_ckpt_dir = f'ckpt/{category}_ckpt' # 모델 저장할 곳
    os.makedirs(quant_ckpt_dir, exist_ok=True)
    save_path = os.path.join(quant_ckpt_dir, f'{group_config}.pt')
    
    
    # fisher 추출
    original_ckpt_path = 'checkpoints/ddpm_checkpoint'
    checkpoint = torch.load(original_ckpt_path)
    model = UNET().cuda()
    
    scheduler = DDPM_Scheduler()
    z = torch.randn(1, 1, 32, 32).cuda()

    
    ## 레이어 그룹별 threshold 구하기!
    threshold_group = get_threshold_group(model, z, scheduler, timesteps, checkpoint, cut_time_step, group_config)
		
    
    #######################################################
    ## 3. 양자화 & 양자화 모델 저장
    #######################################################
    
    # Layer 별 전체 피셔 구하기
    fisher_information = calculate_fisher_information_per_timestep_by_layer(
        model=model,
        z=z,
        scheduler=scheduler,
        num_time_steps=timesteps,
        per=50,
        cut_time_step=cut_time_step,
        group_config=group_config
    )

    # 모델에 양자화 적용
    for timestep, fisher_information_per_layer in fisher_information.items():
        if timestep == cut_time_step:  # time step이 1이면 (이 때 score가 나쁘지 않음)
            quantized_model = apply_threshold_to_model(model, fisher_information_per_layer, threshold_group)
            torch.save({'weights': quantized_model.state_dict(), 'ema': checkpoint['ema']}, save_path)
    
    
    # 모델을 사용해 이미지를 생성
    generated_images, times = inference(save_path,  num_time_steps = timesteps, num_images = num_images)
    save_generated_images(generated_images, quant_generated_images_dir)
    
    fid_score = compute_fid_score(real_images_dir, quant_generated_images_dir)
    print('-------------------layer_group-------------------')
    print(f'quant FID Score: {fid_score}')
    
    # 양자화 모델 메모리 계산
    analyze_weight_size(save_path)
    
    
if __name__ == '__main__':  
    # FutureWarning을 무시
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()