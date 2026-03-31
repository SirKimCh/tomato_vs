import sys
import torch
if not torch.cuda.is_available():
    print("No GPU Found")
    sys.exit(1)

import os
import argparse
import shutil
import gc
import json
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline

try:
    from google import genai
    GENAI_NEW = True
except ImportError:
    GENAI_NEW = False

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--strength', type=float, default=0.35)
parser.add_argument('--guidance', type=float, default=7.5)
parser.add_argument('--output_log_dir', type=str, default='tomato_vs')
args = parser.parse_args()

base_dir = Path(__file__).parent.resolve()
baseline_train_dir = base_dir / 'datasets' / 'baseline' / 'train'
datasets_dir = base_dir / 'datasets'

env_path = base_dir / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv('GEMINI_API_KEY')
gemini_client = None

if api_key and GENAI_NEW:
    try:
        gemini_client = genai.Client(api_key=api_key)
        print("Gemini API initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        print("Stopping - Gemini prompt generation required.")
        sys.exit(1)
else:
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file")
    if not GENAI_NEW:
        print("Error: google-genai package not installed. Run: pip install google-genai")
    print("Stopping - Gemini prompt generation required.")
    sys.exit(1)

if not baseline_train_dir.exists():
    print(f"Error: {baseline_train_dir} does not exist")
    sys.exit(1)

classes = [d.name for d in baseline_train_dir.iterdir() if d.is_dir()]

if len(classes) == 0:
    print(f"Error: No class directories found in {baseline_train_dir}")
    sys.exit(1)

prompt_cache = {}

FALLBACK_PROMPTS = {
    "Tomato___Early_blight": "Tomato Early blight leaf, dark brown to black angular spots, distinct concentric rings, prominent yellow halo, dry necrotic lesions, slightly sunken texture, macro photography, 4k",
    "Tomato___Late_blight": "Tomato Late blight leaf, large irregular water-soaked gray-green lesions, white fuzzy mold on underside, rapid tissue death, dark brown margins, macro photography, 4k",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold disease, pale green to yellow patches on upper leaf surface, olive-brown velvety fungal growth on underside, curling leaf edges, macro photography, 4k",
    "Tomato___Septoria_leaf_spot": "Tomato Septoria leaf spot, small circular spots with dark brown borders, tan gray centers with tiny black dots, lower leaves affected first, macro photography, 4k",
    "Tomato___Spider_mites_Two_spotted_spider_mite": "Tomato spider mite damage, fine yellow stippling on leaves, bronze discoloration, tiny webbing visible, leaf curling and drying, macro photography, 4k",
    "Tomato___Target_Spot": "Tomato Target Spot disease, brown circular lesions with concentric rings, target-like pattern, yellow halo, leaf tissue death, macro photography, 4k",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus, severe upward leaf curling, yellowing margins, stunted growth, crumpled leaf texture, reduced leaf size, macro photography, 4k",
    "Tomato___Tomato_mosaic_virus": "Tomato mosaic virus leaf, mottled light and dark green patterns, leaf distortion, fernleaf appearance, blistered texture, stunted growth, macro photography, 4k",
    "Tomato___Bacterial_spot": "Tomato Bacterial spot leaf, small dark brown to black raised spots, water-soaked margins, yellow halos, scabby lesion texture, macro photography, 4k",
    "Tomato___healthy": "Healthy fresh tomato leaf, vibrant uniform green color, smooth surface, clearly visible veins, no spots or discoloration, macro photography, 4k"
}

def get_fallback_prompt(class_name):
    if class_name in FALLBACK_PROMPTS:
        return FALLBACK_PROMPTS[class_name]

    for key in FALLBACK_PROMPTS:
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            return FALLBACK_PROMPTS[key]

    class_clean = class_name.replace('_', ' ').replace('  ', ' ')
    return f"{class_clean} plant leaf disease symptoms, detailed texture, discoloration, spots, macro photography, 4k"

def generate_prompt_with_gemini(class_name):
    if class_name in prompt_cache:
        return prompt_cache[class_name]

    if 'healthy' in class_name.lower():
        prompt = "Healthy fresh tomato leaf, vibrant uniform green color, smooth surface, clearly visible veins, no spots or discoloration, highly detailed, photorealistic, macro photography, 4k"
        prompt_cache[class_name] = prompt
        return prompt

    try:
        class_clean = class_name.replace('_', ' ').replace('  ', ' ')
        system_prompt = f"You are a plant pathologist. Describe the visual symptoms of {class_clean} on a leaf for an image generation prompt. Start with the plant and disease name (e.g., 'Tomato Early blight leaf'). Keep it under 50 words, comma-separated, focus on colors, spots, texture, and lesion patterns. End with 'macro photography, 4k'. No intro/outro."

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=system_prompt
        )
        generated_prompt = response.text.strip()
        prompt_cache[class_name] = generated_prompt
        return generated_prompt
    except Exception as e:
        print(f"Warning: Gemini API error for {class_name}: {e}")
        fallback = get_fallback_prompt(class_name)
        print(f"Using fallback prompt for {class_name}")
        prompt_cache[class_name] = fallback
        return fallback

negative_prompt = "cartoon, drawing, painting, illustration, blur, low quality, distorted, watermark, text, artificial, tomato fruit, red fruit, stem, stalk, branch, insects, human, fingers"

print("Loading Stable Diffusion model...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)

print("Model loaded successfully.")

sd_name = 'sd_x5'
multiplier = 4

print(f"Generating {sd_name} (5x = original + {multiplier}x generated)...")

output_dir = datasets_dir / sd_name / 'train'
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

output_log_dir = Path(args.output_log_dir)
output_log_dir.mkdir(parents=True, exist_ok=True)

total_classes = len(classes)
first_class_dir = baseline_train_dir / classes[0]
num_original_per_class = len([f for f in os.listdir(first_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
total_images = total_classes * num_original_per_class * multiplier
global_gen_count = 0

print(f"\nTotal images to generate: {total_images}")
print("="*60)

pbar = tqdm(total=total_images, desc="SD Generation", unit="img")

for class_name in classes:
    class_output_dir = output_dir / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)

    class_input_dir = baseline_train_dir / class_name
    image_files = sorted([f for f in os.listdir(class_input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    num_original = len(image_files)
    gen_per_image = multiplier

    for img_file in image_files:
        img_path = class_input_dir / img_file
        shutil.copy(str(img_path), str(class_output_dir / img_file))

    prompt = generate_prompt_with_gemini(class_name)
    print(f"\n  {class_name}")
    print(f"    Prompt: {prompt}")

    gen_count = 0

    for img_file in image_files:
        img_path = class_input_dir / img_file
        init_image = Image.open(img_path).convert('RGB')
        init_image = init_image.resize((512, 512))

        img_basename = os.path.splitext(img_file)[0]
        img_ext = os.path.splitext(img_file)[1]

        for gen_idx in range(gen_per_image):
            with torch.no_grad():
                generated = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    strength=args.strength,
                    guidance_scale=args.guidance,
                    num_inference_steps=50
                ).images[0]

            output_path_gen = class_output_dir / f"{img_basename}_sd{gen_idx}{img_ext}"
            generated.save(output_path_gen)
            gen_count += 1
            global_gen_count += 1
            pbar.update(1)

            if gen_count % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    print(f"    Result: {num_original} original + {gen_count} generated = {num_original + gen_count} total")
    torch.cuda.empty_cache()
    gc.collect()

pbar.close()

print(f"\n{'='*60}")
print(f"Generation complete: {global_gen_count}/{total_images} images")
print(f"{'='*60}")

output_log_path = output_log_dir / 'used_prompts.json'
with open(output_log_path, 'w', encoding='utf-8') as f:
    json.dump(prompt_cache, f, indent=2, ensure_ascii=False)

generation_log_path = output_log_dir / 'generation_log.txt'
with open(generation_log_path, 'w', encoding='utf-8') as f:
    f.write(f"SD Generation Log\n")
    f.write(f"Strength: {args.strength}, Guidance: {args.guidance}\n")
    f.write(f"Total Generated: {global_gen_count}/{total_images}\n")
    f.write(f"{'='*60}\n\n")
    for class_name, prompt in prompt_cache.items():
        f.write(f"Class: {class_name}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"{'-'*40}\n\n")

print(f"Prompts saved to {output_log_path}")
print(f"Log saved to {generation_log_path}")
print("SD generation complete.")
