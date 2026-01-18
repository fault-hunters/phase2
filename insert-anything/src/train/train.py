from torch.utils.data import DataLoader
import torch
import lightning as L
import yaml
import os
import time
from ..data.all_data import AllDatasetCSV
from ..models.my_model import InsertAnything
from .callbacks import TrainingCallback

def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank

def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Initialize
    rank = get_rank()
    is_main_process = (rank == 0)
    torch.cuda.set_device(rank)

    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # ✅ YAML 설정에 'resume_lora_path'가 있는지 확인합니다.
    # 예: resume_lora_path: "./output/20250117/ckpt/15000/pytorch_lora_weights.safetensors"
    resume_path = training_config.get("resume_lora_path", None)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # ==========================================
    # ✅ CSV dataset (Train & Val) 설정
    # ==========================================
    csv_path = training_config.get("csv_path", None)
    val_csv_path = training_config.get("val_csv_path", None) # YAML에서 val_csv_path 읽어옴
    base_dir = training_config.get("base_dir", "")

    assert csv_path is not None, "train.csv_path must be set in config"

    # 1. Train Loader
    train_dataset = AllDatasetCSV(
        csv_path=csv_path,
        base_dir=base_dir,
        data_type=training_config.get("data_type", "noperson"),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # 2. Validation Loader (설정되어 있을 경우에만 생성)
    val_loader = None
    if val_csv_path:
        val_dataset = AllDatasetCSV(
            csv_path=val_csv_path,
            base_dir=base_dir,
            data_type=training_config.get("data_type", "noperson"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config["batch_size"],
            shuffle=False, # 검증 데이터는 섞지 않음
            num_workers=training_config["dataloader_workers"],
        )

    # Initialize model
    trainable_model = InsertAnything(
        flux_fill_id=config["flux_fill_path"],
        flux_redux_id=config["flux_redux_path"],
        lora_path=resume_path,  # ✅ 추가된 파라미터 - LoRA
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    # Callbacks
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
        # validation 로직이 있다면 얼마나 자주 체크할지 설정 (기본 1 epoch)
        check_val_every_n_epoch=1 
    )

    setattr(trainer, "training_config", training_config)

    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}", exist_ok=True)
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # ==========================================
    # ✅ Start training (Val Loader 포함)
    # ==========================================
    trainer.fit(trainable_model, train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()
