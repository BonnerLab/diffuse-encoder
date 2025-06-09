import json
import torch
import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from bonner.datasets.allen2021_natural_scenes._stimuli import (
    StimulusSet,
    load_nsd_metadata,
)
from diffusers import (
    ControlNetModel,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    StableDiffusionControlNetPipeline, 
    UniPCMultistepScheduler, 
    StableDiffusionControlNetInpaintPipeline
)

def rescale_image(image, target_scale):
    """
    Rescale a single PIL image to the specified target scale.
    
    Parameters:
    image (PIL.Image): The input image to be rescaled.
    target_scale (int): The target scale for resizing the image.
    
    Returns:
    PIL.Image: The rescaled image.
    """
    width, height = image.size
    new_width, new_height = target_scale
    return image.resize((new_width, new_height), Image.LANCZOS)

def rescale_images(images, target_scale=(512,512)):
    """
    Rescale a single PIL image or a list of PIL images to the specified target scale.
    
    Parameters:
    images (PIL.Image or list of PIL.Image): The input image(s) to be rescaled.
    target_scale (int): The target scale for resizing the image(s).
    
    Returns:
    PIL.Image or list of PIL.Image: The rescaled image(s).
    """
    if isinstance(images, list):
        return [rescale_image(img, target_scale) for img in images]
    else:
        return rescale_image(images, target_scale)

def load_model_and_pipeline(model_name, experiment, controlnet_name=None, mask_image= None,control_image= None):
    if experiment == 1:
        if mask_image == None:
            raise ValueError('The path for mask image is required for experiment 1')
        pipe = AutoPipelineForInpainting.from_pretrained(model_name, safety_checker=None, torch_dtype=torch.float16)
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
    
    elif experiment == 2:
        pipe = AutoPipelineForImage2Image.from_pretrained(model_name, safety_checker=None, torch_dtype=torch.float16)
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
    
    elif experiment == 3 and controlnet_name:
        if control_image ==None:
            raise ValueError('The path for control image is required for experiment 3')
        controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_name, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
    else:
        raise ValueError('Invalid experiment index or missing controlnet_name for experiment 3')
    return pipe

def get_prompt(stimulus_set, prompt_type, image_id_nsd=None, image_id_COCO=None):
    if prompt_type == 'caption':
        return stimulus_set.captions[image_id_nsd]
    elif prompt_type == 'category':
        labels = [f"{val} {category}" for val, category in zip(stimulus_set.annotations.values[image_id_nsd], stimulus_set.annotations['category'].values) if val != 0]
        if labels:
            return ', '.join(labels[:-1]) + ' and ' + labels[-1]
    elif prompt_type == 'BLIP':
        with open(f'BLIP_Caption/{image_id_COCO}.txt', 'r') as file:
            return file.read()
    elif prompt_type == 'guess':
        return ''
    else:
        raise ValueError('Please define a valid prompt type')

def process_images(pipe, experiment, prompt, init_image, mask_image, control_image, guidance_scale, strength, num_images_per_prompt, num_inference_steps):
    if guidance_scale < 0:
        guidance_scale = abs(guidance_scale)
        negative_prompt = prompt
        prompt = ['', '', '', '', '']
    else:
        negative_prompt = None

    if experiment == 1:
        mask = Image.open(mask_image)
        return pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask, num_images_per_prompt=num_images_per_prompt, guidance_scale=guidance_scale).images
    elif experiment == 2:
        return pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=strength, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt).images    
    elif experiment == 3:
        control = Image.open(control_image)
        return pipe(prompt=prompt, negative_prompt=negative_prompt, image=control, num_inference_steps=num_inference_steps, num_images_per_prompt=num_images_per_prompt).images

def save_images(images, output_dir, image_id_COCO = None, start_index=None):
    if image_id_COCO != None:
        output_dir = f'{output_dir}/{image_id_COCO}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, image in enumerate(images):
        start_index = start_index if start_index else 1
        filename = f"image_{idx+start_index}.jpg"
        image.save(os.path.join(output_dir, filename), "JPEG")

def main(model_name, controlnet_name, 
         prompt, guidance_scale, prompt_type, 
         image, strength, 
         mask_image, control_image, 
         num_inference_steps, num_images_per_prompt, 
         experiment, output, device, 
         NSD_shared,start_index):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    pipe = load_model_and_pipeline(model_name, experiment, controlnet_name, mask_image, control_image)
    pipe.to(device)
    
    if NSD_shared == 'Full':
        stimulus_set = StimulusSet()
        metadata = load_nsd_metadata()
        condition1 = metadata['shared1000'] == True
        condition2 = metadata['cocoSplit'] == 'train2017'
        image_ids_1000 = metadata[condition1 & condition2]['cocoId']
        five_permutations = np.array([np.random.permutation(1000) for _ in range(5)])

        for i in tqdm(range(1000)):
            index = i
            image_id_COCO = image_ids_1000.iloc[index]
            image_id_nsd = image_ids_1000.keys()[index]
            init_image = Image.fromarray(stimulus_set.stimuli[image_id_nsd].values)
            prompt = get_prompt(stimulus_set, prompt_type, image_id_nsd, image_id_COCO)

            if strength < 0:
                strength = abs(strength)
                random_image_id_nsd = image_ids_1000.keys()[five_permutations[:, index]]
                init_image = [Image.fromarray(stimulus_set.stimuli[random_image_id_nsd[count]].values) for count in range(5)]

            init_image = rescale_images(init_image)
            images = process_images(pipe, experiment, prompt, init_image, mask_image, control_image, guidance_scale, strength, num_images_per_prompt, num_inference_steps)
            save_images(images, output,image_id_COCO,start_index)

    elif NSD_shared == 'Sample':
        stimulus_set = StimulusSet()
        metadata = load_nsd_metadata()
        condition1 = metadata['shared1000'] == True
        condition2 = metadata['cocoSplit'] == 'train2017'
        image_ids_1000 = metadata[condition1 & condition2]['cocoId']
        five_permutations = np.array([np.random.permutation(1000) for _ in range(5)])

        for i in tqdm(range(5)):
            index = i
            image_id_COCO = image_ids_1000.iloc[index]
            image_id_nsd = image_ids_1000.keys()[index]
            init_image = Image.fromarray(stimulus_set.stimuli[image_id_nsd].values)
            prompt = get_prompt(stimulus_set, prompt_type, image_id_nsd, image_id_COCO)

            if strength < 0:
                strength = abs(strength)
                random_image_id_nsd = image_ids_1000.keys()[five_permutations[:, index]]
                init_image = [Image.fromarray(stimulus_set.stimuli[random_image_id_nsd[count]].values) for count in range(5)]

            init_image = rescale_images(init_image)
            images = process_images(pipe, experiment, prompt, init_image, mask_image, control_image, guidance_scale, strength, num_images_per_prompt, num_inference_steps)
            save_images(images, output,image_id_COCO,start_index)

    else:
        raise ValueError('NSD_shared must be "Full" or "Sample. Other case is not developed here."')
    return

if __name__ == "__main__":
    # 
    parser = argparse.ArgumentParser(description="Synthesize images using Hugging Face models")
    parser.add_argument('--model', type=str, default="runwayml/stable-diffusion-v1-5", help='Name of the model to use')
    parser.add_argument('--controlnet', type=str, default="lllyasviel/sd-controlnet-canny",help='Name of the ControlNet model to use')
    parser.add_argument('--prompt', type=str, help='Path to the prompt file')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale for text weight in the model')
    parser.add_argument('--prompt_type', type=str, default='caption', help='Type of prompt: caption, category, BLIP, guess')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--strength', type=float, default=0.8, help='Strength of the image influence')
    parser.add_argument('--mask_image', type=str, help='Path to the mask image')
    parser.add_argument('--control_image', type=str, help='Path to the control image')
    parser.add_argument('--num_inference_steps', type=int, default=20, help='Number of inference steps')
    parser.add_argument('--num_images_per_prompt', type=int, default=1, help='Number of images to generate per prompt')
    parser.add_argument('--experiment', type=int, required=True, help='1:inpaint, 2:img2img, 3:controlnet')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--NSD_shared', type=str, required=True, help='Use NSD shared1000 images: "Full" or "Sample"')
    parser.add_argument('--start_index', type=int, help='Default: the image we save would be 1')

    args = parser.parse_args()
    main(args.model, args.controlnet, 
         args.prompt, args.guidance_scale, args.prompt_type, 
         args.image, args.strength, 
         args.mask_image, args.control_image, 
         args.num_inference_steps, args.num_images_per_prompt, 
         args.experiment, args.output, args.device, 
         args.NSD_shared,args.start_index)

