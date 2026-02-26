# Qwen VL OCR

Qwen2-VL / Qwen2.5-VL 비전-언어 모델을 활용한 이미지 OCR 도구입니다.

## 요구사항

```bash
pip install torch transformers qwen-vl-utils
```

- Python 3.8+
- CUDA 지원 GPU (권장)

## 사용법

```bash
python qwen_ocr.py --image <이미지 경로> [옵션]
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--image` | (필수) | 입력 이미지 경로 |
| `--model` | `Qwen/Qwen2-VL-2B-Instruct` | 사용할 모델 ID |
| `--mode` | `layout` | OCR 모드 (`plain` / `layout` / `json`) |
| `--max_new_tokens` | `2048` | 최대 생성 토큰 수 |

### OCR 모드

- **plain** – 이미지 내용을 자연어로 설명
- **layout** – 줄바꿈과 공백을 유지하며 텍스트 추출 후 요약 제공
- **json** – `{"full_text": "...", "lines": [...]}` 형식의 JSON으로 추출

### 예시

```bash
# 기본 (layout 모드)
python qwen_ocr.py --image sample/image.png

# JSON 추출
python qwen_ocr.py --image sample/text.png --mode json

# 더 큰 모델 사용
python qwen_ocr.py --image sample/image.png --model Qwen/Qwen2-VL-7B-Instruct --mode layout
```

## 지원 모델

| 모델 ID | 크기 | 비고 |
|---------|------|------|
| `Qwen/Qwen2-VL-2B-Instruct` | 2B | 가볍고 빠름 (기본값) |
| `Qwen/Qwen2-VL-7B-Instruct` | 7B | 성능/속도 균형 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | 문서 인식 강화 |
| `Qwen/Qwen2.5-VL-32B-Instruct` | 32B | 최고 성능, 고사양 필요 |

## 프로젝트 구조

```
OCR/
├── qwen_ocr.py      # 메인 스크립트
├── README.md
└── sample/
    ├── image.png    # 샘플 이미지
    └── text.png     # 샘플 텍스트 이미지
```
