import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
import cv2
import torch
import os
from ..models.pipeline_tools import encode_images, prepare_text_input
import json
import math

try:
    import wandb
except ImportError:
    wandb = None

def f(r, T=0.6, beta=0.1):
    return np.where(r < T, beta + (1 - beta) / T * r, 1)


def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]

    if mask.sum() < 10:
        return 0,h,0,w
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return (y1,y2,x1,x2)



def expand_bbox(mask, yyxx, ratio, min_crop=0):
    y1,y2,x1,x2 = yyxx
    H,W = mask.shape[0], mask.shape[1]

    yyxx_area = (y2-y1+1) * (x2-x1+1)
    r1 = yyxx_area / (H * W)
    r2 = f(r1)
    ratio = math.sqrt(r2 / r1)

    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)
    h = max(h,min_crop)
    w = max(w,min_crop)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)


def pad_to_square(image, pad_value = 255, random = False):
    H,W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0,padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if len(image.shape) == 2: 
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2))
        else:
            pad_param = ((padd_1, padd_2), (0, 0))
    elif len(image.shape) == 3: 
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2), (0, 0))
        else:
            pad_param = ((padd_1, padd_2), (0, 0), (0, 0))

    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
    return image


def expand_image_mask(image, mask, ratio=1.4):
    h,w = image.shape[0], image.shape[1]
    H,W = int(h * ratio), int(w * ratio) 
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W -w) // 2)
    w2 = W -w - w1

    pad_param_image = ((h1,h2),(w1,w2),(0,0))
    pad_param_mask = ((h1,h2),(w1,w2))
    image = np.pad(image, pad_param_image, 'constant', constant_values=255)
    mask = np.pad(mask, pad_param_mask, 'constant', constant_values=0)
    return image, mask

def box2squre(image, box):
    H,W = image.shape[0], image.shape[1]
    y1,y2,x1,x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h,w = y2-y1, x2-x1

    if h >= w:
        x1 = cx - h//2
        x2 = cx + h//2
    else:
        y1 = cy - w//2
        y2 = cy + w//2
    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)

def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 2 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if (self.total_steps % self.save_interval == 0 or self.total_steps == 1) and self.total_steps < 15500:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals   or self.total_steps == 1
        if self.total_steps % self.sample_interval == 0 or self.total_steps == 1:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                self.total_steps,
                f"{self.save_path}/{self.run_name}/output_diptych",
                f"lora_{self.total_steps}",
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        steps,
        save_path,
        file_name,
    ):
        seed = 42
        size = (768, 768)
        
        # --- 설정 부분 ---
        csv_path = "/content/drive/MyDrive/[불량헌터스] 광고 이미지 텍스트 불량 감지 시스템/Dataset/phase2/phase2_train_0116.csv" # CSV 파일 경로
        base_dir = "/content/drive/MyDrive/[불량헌터스] 광고 이미지 텍스트 불량 감지 시스템/Dataset/phase2" # 데이터가 실제로 위치한 최상위 폴더 경로
        num_samples = 5 # 매 샘플링 시점마다 인퍼런스 돌릴 행 개수
        # ----------------

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        def final_inference(csv_path, base_dir, n):
            save_subdir = os.path.join(save_path, f"{file_name}_seed{seed}")
            os.makedirs(save_subdir, exist_ok=True)

            # CSV 로드 및 상위 n개 추출
            df = pd.read_csv(csv_path)
            test_df = df.head(n)

            for index, row in test_df.iterrows():
                # 경로 결합 (CSV의 ./ 경로 대응을 위해 lstrip 처리)
                source_image_path = os.path.join(base_dir, row['tar_image'].lstrip('./'))
                ref_image_path = os.path.join(base_dir, row['ref_image'].lstrip('./'))
                ref_mask_path = os.path.join(base_dir, row['ref_masked'].lstrip('./'))
                mask_image_path = os.path.join(base_dir, row['tar_masked'].lstrip('./'))

                source_image_filename = os.path.basename(source_image_path)

                if os.path.exists(mask_image_path):
                    print(f"Generating sample {index+1}/{n}: {source_image_filename}...")

                    # 이미지 로드
                    ref_image = cv2.imread(ref_image_path)
                    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
                    tar_image = cv2.imread(source_image_path)
                    tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
                    
                    # 마스크 처리
                    ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:, :, 0]
                    tar_mask = (cv2.imread(mask_image_path) > 128).astype(np.uint8)[:, :, 0]

                    if tar_mask.shape != tar_image.shape[:2]:
                        tar_mask = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))

                    # --- 기존 전처리 로직 시작 ---
                    ref_box_yyxx = get_bbox_from_mask(ref_mask)
                    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
                    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)
                    
                    y1, y2, x1, x2 = ref_box_yyxx
                    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
                    ref_mask_crop = ref_mask[y1:y2, x1:x2]
                    
                    ratio = 1.3
                    masked_ref_image, _ = expand_image_mask(masked_ref_image, ref_mask_crop, ratio=ratio)
                    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)

                    # Zoom in target
                    tar_box_yyxx = get_bbox_from_mask(tar_mask)
                    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)
                    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=2)
                    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
                    
                    ty1, ty2, tx1, tx2 = tar_box_yyxx_crop
                    old_tar_image = tar_image.copy()
                    tar_image_crop = tar_image[ty1:ty2, tx1:tx2, :]
                    tar_mask_crop = tar_mask[ty1:ty2, tx1:tx2]

                    H1, W1 = tar_image_crop.shape[0], tar_image_crop.shape[1]
                    tar_mask_sq = pad_to_square(tar_mask_crop, pad_value=0)
                    tar_mask_resize = cv2.resize(tar_mask_sq, size)

                    masked_ref_image_resize = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
                    
                    # Flux Redux Prior
                    pipe_prior_output = pl_module.flux_redux(Image.fromarray(masked_ref_image_resize))

                    tar_image_sq = pad_to_square(tar_image_crop, pad_value=255)
                    H2, W2 = tar_image_sq.shape[0], tar_image_sq.shape[1]
                    tar_image_resize = cv2.resize(tar_image_sq, size)

                    # Diptych 구성
                    diptych_ref_tar = np.concatenate([masked_ref_image_resize, tar_image_resize], axis=1)
                    tar_mask_3ch = np.stack([tar_mask_resize, tar_mask_resize, tar_mask_resize], -1)
                    mask_black = np.zeros_like(tar_image_resize)
                    mask_diptych = np.concatenate([mask_black, tar_mask_3ch], axis=1)

                    diptych_img = Image.fromarray(diptych_ref_tar)
                    mask_diptych[mask_diptych == 1] = 255
                    mask_diptych_img = Image.fromarray(mask_diptych)

                    # 모델 추론
                    generator = torch.Generator(pl_module.device).manual_seed(seed)
                    edited_image = pl_module.flux_fill_pipe(
                        image=diptych_img,
                        mask_image=mask_diptych_img,
                        height=mask_diptych_img.size[1],
                        width=mask_diptych_img.size[0],
                        max_sequence_length=512,
                        generator=generator,
                        **pipe_prior_output,
                    ).images[0]

                    # 후처리 및 저장
                    tw, th = edited_image.size
                    edited_image = edited_image.crop((tw // 2, 0, tw, th))
                    edited_image_np = np.array(edited_image)
                    
                    final_output = crop_back(edited_image_np, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop))
                    final_output = Image.fromarray(final_output)

                    save_filename = f"step{steps}_{index}_{source_image_filename}"
                    final_output.save(os.path.join(save_subdir, save_filename))
                else:
                    print(f"Skip {index}: Mask not found at {mask_image_path}")

        # 로직 실행
        if os.path.exists(csv_path):
            final_inference(csv_path, base_dir, num_samples)
        else:
            print(f"CSV file not found at {csv_path}")
        

       
           
