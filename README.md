# Diffusers train_text_to_image_lora trainer
This is a Cog wrapper around the [Diffusers method to train a text-to-image lora](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)

# How to run:

A demo training run for `lambdalabs/naruto-blip-captions` with base `runwayml/stable-diffusion-v1-5`: (takes ~5 hours on a 2080 Ti GPU with 11GB of VRAM)

    cog predict -i base_model="runwayml/stable-diffusion-v1-5" -i dataset="lambdalabs/naruto-blip-captions


To run a demo and upload the result to Huggingface:

    cog predict -i base_model="runwayml/stable-diffusion-v1-5" -i dataset="lambdalabs/naruto-blip-captions" -i hf_token="hf_token"


# Output

Output will be a `lora.tar` file, which has a `pytorch_lora_weights.safetensors` file in it.
