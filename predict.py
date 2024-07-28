# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path, Secret
import os
import subprocess

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    def predict(
        self,
        base_model: str = Input(description="Base huggingface model to use", default="runwayml/stable-diffusion-v1-5"),
        dataset: str = Input(description="Huggingface dataset to use", default="lambdalabs/naruto-blip-captions"),
        dataloader_num_workers: int = Input(description="Number of workers for dataloader", default=8, ge=1, le=16),
        resolution: int = Input(description="Resolution for training", default=512, ge=128, le=1024),
        train_batch_size: int = Input(description="Batch size for training", default=1, ge=1, le=4),
        gradient_accumulation_steps: int = Input(description="Gradient accumulation steps", default=4, ge=1, le=8),
        max_train_steps: int = Input(description="Maximum number of training steps", default=15000),
        learning_rate: float = Input(description="Learning rate for training", default=0.0001, ge=0.0001, le=0.01),
        max_grad_norm: float = Input(description="Maximum gradient norm", default=1, ge=0.001, le=10),
        validation_prompt: str = Input(description="Validation prompt", default="A naruto with blue eyes."),
        seed: int = Input(description="Seed for reproducibility", default=None),
        # Optional inputs:
        hf_token: Secret = Input(description="Huggingface token", default=None),
        hub_model_id: str = Input(description="Huggingface model id to upload to", default="naruto-lora")
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")

        output_dir = "/tmp/train-t2i-lora"
        run_params = [
            "accelerate", "launch", "train_text_to_image_lora.py",
            "--pretrained_model_name_or_path", base_model,
            "--dataset_name", dataset,
            "--dataloader_num_workers", str(dataloader_num_workers),
            "--resolution", str(resolution),
            "--center_crop",
            "--random_flip",
            "--train_batch_size", str(train_batch_size),
            "--gradient_accumulation_steps", str(gradient_accumulation_steps),
            "--max_train_steps", str(max_train_steps),
            "--learning_rate", str(learning_rate),
            "--max_grad_norm", str(max_grad_norm),
            "--lr_scheduler", "cosine",
            "--lr_warmup_steps", "0",
            "--output_dir", output_dir,
            "--validation_prompt", validation_prompt,
            "--seed", str(seed),
        ]
        if hf_token is not None:
            run_params.extend(["--push_to_hub"])
            run_params.extend(["--hf_token", hf_token.get_secret_value()])
            run_params.extend(["--hub_model_id", hub_model_id])
        subprocess.check_call(run_params)

        # tar up lora folder to create lora.tar:
        os.system(f"tar -cvf {output_dir}/lora.tar -C {output_dir} .")

        output_path = output_dir+"/lora.tar"
        return Path(output_path)
