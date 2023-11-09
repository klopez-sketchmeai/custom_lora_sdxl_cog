from sdxl_train_network import SdxlNetworkTrainer

import argparse
import os
import shutil
import tarfile
import subprocess
import time

from cog import BaseModel, Input, Path

from library import sdxl_train_util
import train_network


SDXL_MODEL_CACHE = "./sdxl-cache"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"

REG_IMAGES_DIR = "man_reg_data_1024"
TRAINING_IMAGES_DIR = "kev_imgs_1024"
LOGS_DIR = "lora_logs"

"""
Wrapper around actual trainer.
"""
OUTPUT_DIR = "training_out"
OUTPUT_PATH = "trained_model.tar"


class TrainingOutput(BaseModel):
    weights: Path


def download_weights(url, dest):
    start = time.time()
    print("downloading from train.py url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

def train() -> TrainingOutput:

    if not os.path.exists(SDXL_MODEL_CACHE):
        download_weights(SDXL_URL, SDXL_MODEL_CACHE)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print('train.py about to call setup_parser')
    parser = setup_parser()
    print('train.py about to call parse_args')
    args = parser.parse_args()

    args.pretrained_model_name_or_path = SDXL_MODEL_CACHE

    args.train_data_dir = TRAINING_IMAGES_DIR
    args.reg_data_dir = REG_IMAGES_DIR
    args.output_dir = OUTPUT_DIR
    args.output_name = OUTPUT_PATH
    args.logging_dir = LOGS_DIR 

    args.resolution = "1024,1024"
    args.cache_latents = True
    args.cache_latents_to_disk = True
    args.enable_bucket = True
    args.max_bucket_reso = 2048
    args.bucket_no_upscale = True
    args.save_precision = "bf16"
    args.save_every_n_epochs = 1
    args.xformers = True # Test with this off for improved quality apparantely
    args.max_train_steps = 2000
    args.max_data_loader_n_workers = 0
    args.seed = 123456
    args.gradient_checkpointing = True
    args.mixed_precision = "bf16"
    args.optimizer_type = "Adafactor"
    args.learning_rate = 0.0004
    args.optimizer_args = ["scale_parameter=False", "relative_step=False", "warmup_init=False"]
    args.lr_scheduler_num_cycles = 10
    args.unet_lr = 0.0004
    args.text_encoder_lr = 0.0004
    args.network_module = "networks.lora"
    args.network_dim = 128
    args.no_half_vae = True

    print("the final lora input args: ", args)

    print("About to start LoRa training...")
    trainer = SdxlNetworkTrainer()
    trainer.train(args)

    directory = Path(OUTPUT_DIR)
    out_path = "trained_model.tar"

    with tarfile.open(out_path, "w") as tar:
        for file_path in directory.rglob("*"):
            print(file_path)
            arcname = file_path.relative_to(directory)
            tar.add(file_path, arcname=arcname)

    return TrainingOutput(weights=Path(out_path))

def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    return parser
