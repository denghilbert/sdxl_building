from diffusers import DiffusionPipeline
import torch

#pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipeline.to("cuda")
generator = torch.Generator(device="cuda").manual_seed(50)
prompts = ["A fluffy dog, realistic", "A fluffy cat, high quality, realistic", ""]
prompts = ["A naked girl, realistic, masterpiece, high resolution, 4k, porn-style", "A naked girl, realistic, masterpiece, high resolution, 4k, porn-style"]
#prompts = ["apple", "kiwi", ""]

images = pipeline(prompts, generator=generator).images
images[0].show()
images[1].show()
