# Layer-Adaptive Quantization on Diffusion Model using Fisher Information

## 전체 코드 파일 개요
- `simplediffusion.py`
  - ddpm 모델 구현 및 훈련
  - generated image 생성
  - 생성된 이미지 저장
  - FID score 계산
  - 비교군(원본 모델, 전체 양자화) fid 및 메모리 측정
- `whole_threshold.py`: 전체 임계값 설정 후 양자화, FID 및 메모리 측정
- `layer_group.py`: layer 그룹별 임계값 설정 후 양자화, FID 및 메모리 측정
- `layer_ratio.py`: layer별 임계값 비율 설정 후 양자화, FID 및 메모리 측정
- `layer_math.py`: layer별 평균 분산 적응형 계수 임계값 설정 후 양자화, FID 및 메모리 측정


## 코드 실행 순서 및 방법

### 환경 세팅
1. 서버 접속 <br>
   서버 계정 정보는 메일을 통해 공유드렸습니다!
   
2. conda 가상환경 설정하기 (LAQFI dir 위치에서 실행)
  - conda 가상 환경 생성: `conda env create -f env.yml`
  - conda 가상 환경 활성화: `conda activate sd_env`

### 실험 시작
1. `simplediffusion.py` 실행
2. 양자화 실험 <br>
  - 단일 임곗값 설정 실험: `python3 whole_threshold.py`
  - 레이어  그룹별 임곗값 설정 실험: `python3 layer_ratio.py`
  - 레이어별 임곗값 설정 실험
    - layer 별 임계값 비율 설정: `python3 layer_group.py`
    - layer별 평균 분산 적응형 계수 임계값 설정: `python3 layer_math.py`
