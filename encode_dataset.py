#!pip install diffusers==0.2.4

import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
from torchvision import transforms as tfms
import sys
import os
from tqdm import tqdm
from pathlib import Path
import csv 
from PIL.PngImagePlugin import PngInfo


torch_device = None
vae = None
to_tensor_tfm = None

csv_header = ['imageFile', 'minLatentVal', 'maxLatentVal']

def setup():
    global torch_device
    global vae 
    global to_tensor_tfm 

    # Set device
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True)

    # To the GPU we go!
    vae = vae.to(torch_device)

    # Using torchvision.transforms.ToTensor
    to_tensor_tfm = tfms.ToTensor()

def pil_to_latent(input_im):
  # Single image -> single latent in a batch (so size 1, 4, 64, 64)
  with torch.no_grad():
    latent = vae.encode(to_tensor_tfm(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
  return 0.18215 * latent.sample() # or .mean or .sample

def latents_to_pil(latents):
  # bath of latents -> list of images
  latents = (1 / 0.18215) * latents
  latents = latents.to(torch_device)
  with torch.no_grad():
    image = vae.decode(latents)
  image = (image.detach().cpu() / 2 + 0.5).clamp(0, 1)
  image = image.permute(0, 2, 3, 1).numpy()
  images = (image * 255).round().astype("uint8")
  pil_images = [Image.fromarray(image) for image in images]
  return pil_images

def latents_as_images(latents):
  minValue = latents.min()
  maxValue = latents.max()
  latents = (latents-minValue)/(maxValue-minValue)  
  image = latents.detach().cpu().permute(0, 2, 3, 1).numpy()        
  images = (image * 255).round().astype("uint8")
  pil_images = [Image.fromarray(image) for image in images]
  minValue = minValue.detach().cpu().numpy().astype("float32")
  maxValue = maxValue.detach().cpu().numpy().astype("float32")
  return pil_images, str(minValue), str(maxValue)

def encode(input_image_file):
# Load the image with PIL
    input_image = Image.open(input_image_file).resize((512, 512))
    encoded = pil_to_latent(input_image)    
    return encoded

def decode(encoded_latents):
    decoded = latents_to_pil(encoded_latents)[0]    
    return decoded

def is_float(value):
  try:
    float(value)
    return True
  except:
    return False

def reduced_latents_from_png(png_image_name):
  image_in = Image.open(png_image_name)
  image_text_data = image_in.text         #Check the PNG metadata for minValue, maxValue keys and load their value as float32 if possible
  minValue = image_text_data["minValue"]
  if is_float(minValue):
    minValue = float(minValue)
  else:
    minValue = float(-5.0)
  maxValue = image_text_data["maxValue"]
  if is_float(maxValue):
    maxValue = float(maxValue)
  else:
    maxValue = float(5.0)
  image_in = np.array(image_in, np.float32)  
  print(image_in.shape)
  image_in = image_in/255.0
  image_in = image_in.transpose((2, 0, 1))
  image_out = np.expand_dims(image_in, 0)  
  reduced_latents = torch.tensor(image_out)
  reduced_latents = (reduced_latents*(maxValue-minValue))+minValue
  print(reduced_latents.shape)
  return reduced_latents

def load_png_decode(input_file, output_file):
  print("Load reduced latents")
  reduced_latents = reduced_latents_from_png(input_file)  
  print("Decode reduced latents")
  image = decode(reduced_latents)
  image.save(output_file)

def encode_folder(input_folder, output_folder):
  files = list(Path(input_folder).rglob("*.jpg"))
  csv_data = []
  csv_data.append(csv_header)
  for file in tqdm(files):
    input_file = str(file)
    output_file_name = str(file.stem) + ".png"
    output_file = output_folder + output_file_name
    encoded_latents = encode(input_file)
    encoded_latents_as_image, minValue, maxValue = latents_as_images(encoded_latents)
    metadata = PngInfo()  #Write MinVale, MaxValue as part of the PNG metadata
    metadata.add_itxt("minValue", minValue)
    metadata.add_itxt("maxValue", maxValue)
    encoded_latents_as_image[0].save(output_file, pnginfo=metadata)  #Note: The alpha channel also contains information
    csv_data.append([output_file_name, minValue, maxValue]) #The generated CSV is not used in this sample, it is for reference only
  
  csv_file = output_folder + 'latentMinMaxValues.csv'
  with open(csv_file, 'w', encoding='UTF8', newline="\n") as f:
    writer = csv.writer(f)  
    writer.writerows(csv_data)

def explore_minmax(png_image_name):
    image_in = Image.open(png_image_name)
    image_in = np.array(image_in, np.float32)  
    print(image_in.shape)
    image_in = image_in/255.0
    image_in = image_in.transpose((2, 0, 1))
    image_out = np.expand_dims(image_in, 0)  
    reduced_latents = torch.tensor(image_out)
    for minValue in range(-12,-2,1):
      for maxValue in range(3,13,1):
        updated_reduced_latents = (reduced_latents*float(maxValue-minValue))+float(minValue)
        image = decode(updated_reduced_latents)
        output_file = "explore/minValue" + str(minValue) + "_maxValue"+ str(maxValue) + ".jpg"
        image.save(output_file)


def main():
    print("Load VAE")
    setup()
    encode_folder("input/test_data/A/", "output/test_data/A/")
    encode_folder("input/test_data/B/", "output/test_data/B/")
    
    load_png_decode("output/test_data/A/seed40022.png", "out_seed40022_A_Val.png")
    load_png_decode("output/test_data/B/seed40022.png", "out_seed40022_B_Val.png")
    
    explore_minmax("output/test_data/B/seed40020.png")

if __name__ == '__main__':
    main()