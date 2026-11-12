# DPO from scratch

<a href="https://colab.research.google.com/github/dain77-d/dpo-from-scratch/blob/main/colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" /></a>

**Direct Preference Optimization** (Rafailov et al., NeurIPS 2023, [arXiv 2305.18290](https://arxiv.org/abs/2305.18290))를
라이브러리(TRL 등) 없이 직접 구현하고, 논문 6.1절의 **IMDb 감성 제어 실험**을 작은 규모로 재현합니다.

## 핵심 아이디어

RLHF는 보상 모델을 학습한 뒤 PPO로 정책을 최적화합니다. DPO는 KL 제약이 걸린 보상 최대화 문제의 최적해를 이용해서
보상을 정책과 기준 모델의 로그 확률 비로 표현합니다. 그 결과 **선호 쌍만으로 분류 손실 하나를 최소화**하면 됩니다.

$$\mathcal{L}_\text{DPO} = -\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\Big)$$

구현은 [`dpo.py`](dpo.py)의 약 40줄이 전부입니다. 손실, 응답 토큰만의 로그 확률 합, 패딩 처리를 담고 있습니다.

## 실험 설계 (논문 6.1절 축소판)

| 단계 | 내용 |
|---|---|
| 1. SFT | distilgpt2를 IMDb 리뷰로 미세조정 → 기준 정책 π_ref |
| 2. 선호 데이터 | 리뷰 앞 8토큰을 프롬프트로, π_ref에서 32토큰 완성문 2개 샘플링 → 감성 분류기(SST-2 DistilBERT) 점수가 높은 쪽을 chosen |
| 3. DPO | β별로 학습하며 주기적으로 평가 |
| 4. 평가 | 보상 = 분류기 logit(긍정) − logit(부정), KL = 정책 샘플에서 log π − log π_ref 평균 → **보상-KL 프런티어** |

구현하면서 신경 쓴 점:
- **dropout을 끈 채로 학습**합니다. 켜 두면 0 스텝에서도 π ≠ π_ref가 되어 손실이 log 2보다 크게 시작합니다. 시험 실행에서 실제로 0.79가 나와 발견했습니다.
- KL 추정이 맞으려면 정책에서 그대로 샘플링해야 하므로 `top_k=0`을 씁니다. transformers 기본값은 top_k=50입니다.
- π_ref의 로그 확률은 학습 중 변하지 않으므로 한 번만 계산해 둡니다.

## 결과

> CPU(distilgpt2) 실행 결과를 정리하는 중입니다. GPU(gpt2, β 4개) 결과는 Colab으로 추가할 예정입니다.

## 실행

```bash
pip install -r requirements.txt
python test_dpo.py        # 손실 단위 테스트
python run.py --quick     # 작은 규모로 끝까지 한 번 (CPU 수 분)
python run.py             # 본 실행. GPU가 있으면 자동으로 큰 설정
```

또는 위 **Open in Colab** 배지를 누르고 런타임을 T4 GPU로 바꾼 뒤 모두 실행하면 됩니다.
