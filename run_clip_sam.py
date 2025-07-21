import os
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from segment_anything import build_sam, SamAutomaticMaskGenerator
import clip
# ___________________________________________________


# 경로 설정
# ___________________________________________________

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# SAM 모델 가중치 파일 경로
SAM_CHECKPOINT = os.path.join(PROJECT_DIR, "sam_vit_h_4b8939.pth")

# 테스트 이미지 경로 (testImageFolder 폴더 내, 여기서 테스트 이미지 수정)
IMAGE_PATH = os.path.join(PROJECT_DIR, "testImageFolder", "testImage_001.jpg")
# ___________________________________________________


# 디바이스 설정. 가능하면 GPU 사용, 보드 구성 보고 if문 없애도 됨.
# ___________________________________________________

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
# ___________________________________________________


# SAM 모델 생성 및 마스크 생성기 초기화
# ___________________________________________________

sam = build_sam(checkpoint=SAM_CHECKPOINT).to(device)
mask_generator = SamAutomaticMaskGenerator(sam)
# ___________________________________________________


# 이미지 로드 및 전처리. BGR -> RGB 변환
# ___________________________________________________
bgr = cv2.imread(IMAGE_PATH)
if bgr is None:                                       #경로 오류
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
# ___________________________________________________


# 마스크 생성
# ___________________________________________________
masks = mask_generator.generate(rgb)
print("마스크 개수:", len(masks))
# ___________________________________________________


# 마스크 영역 crop 함수
# ___________________________________________________
def crop_with_mask(image_array, mask_bool):
    out = np.zeros_like(image_array)
    out[mask_bool] = image_array[mask_bool]
    return Image.fromarray(out)

def xywh2xyxy(box):
    x, y, w, h = box
    return (x, y, x + w, y + h)

# Crop 이미지 생성
image_pil = Image.fromarray(rgb)
crops = [crop_with_mask(rgb, m["segmentation"]).crop(xywh2xyxy(m["bbox"])) for m in masks]
# ___________________________________________________


# CLIP 모델 로드
# ___________________________________________________
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# ___________________________________________________


# CLIP 유사도 계산 함수
# ___________________________________________________
def clip_scores(images, text_prompt):
    imgs = torch.stack([preprocess(im).to(device) for im in images])
    txt = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        img_f = clip_model.encode_image(imgs).float()
        txt_f = clip_model.encode_text(txt).float()
    img_f /= img_f.norm(dim=-1, keepdim=True)
    txt_f /= txt_f.norm(dim=-1, keepdim=True)
    probs = (100 * img_f @ txt_f.T).softmax(dim=0) #이미지, 텍스트 사이 유사도 비교, 점수계산
    return probs[:, 0].cpu().numpy()  # (N,)
# ___________________________________________________


# prompt에 가장 유사한 마스크 선택 및 시각화
# ___________________________________________________
search_text = "traffic light"  # 예시 텍스트 프롬프트
scores = clip_scores(crops, search_text)
thres = 0.05                   #이 이상 점수이면 맞다고 판단. 실제 테스트 돌려보며 값 수정 필요
sel_idx = [i for i, s in enumerate(scores) if s > thres]
print("선택된 마스크 인덱스:", sel_idx)

# 시각화
overlay = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)
color = (255, 0, 0, 200)  # 빨간색 반투명

for i in sel_idx:
    mask_img = Image.fromarray(masks[i]["segmentation"].astype(np.uint8) * 255)
    draw.bitmap((0, 0), mask_img, fill=color)

result = Image.alpha_composite(image_pil.convert('RGBA'), overlay)

plt.figure(figsize=(8, 8))
plt.imshow(result)
plt.axis('off')
plt.title(f"Prompt: {search_text}")
plt.show()