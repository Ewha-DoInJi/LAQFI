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
