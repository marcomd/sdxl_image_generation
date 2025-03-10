# SDXL Image generation

![image](./assets/logo.jpg)

## Introduction

This simple python script mounts a webserver to process a prompt and return an image generated by using a text-to-image SDXL model.

It leans on:

- CUDA if an Nvidia card is available
- Diffusers in order to use pretrained diffusion models
- PyTorch as deep learning frameworks
- Uvicorn is an ASGI web server implementation

## Install

`pip install -r requirements.txt `

## Set authentication key

Set an authentication key as env variable on the server that you will use to authenticate the client.

On unix:

```
export GENERATION_AUTH_KEY=wiv83hveivd83vk83
```


On windows:

```
set GENERATION_AUTH_KEY=wiv83hveivd83vk83
```


## Start the server

```bash
python main.py

Loading pipeline components...: 100%|████████████████████████████████████████████████████| 7/7 [00:16<00:00,  2.40s/it]
Starting SDXL Image Generation Server...
INFO:     Started server process [15920]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```


## Submit a prompt

On the client, make the call by passing:

- the authentication key as header
- params:
  - `prompt` the description used by the model to generate the image
  - `model_type` the model used to generate the image, among supported ones (see the section)
  - `height` and `width` of the generated image (default 512x512). Please note that each model supports specific resolutions, so check before changing these values ​​otherwise you may experience incorrect results.


```bash
curl -X POST http://localhost:8000/generate -H "Authorization: wiv83hveivd83vk83" -H "Content-Type: application/json" -d '{"prompt": "A serene mountain landscape at sunset", "model_type": "sdxl-anime", "height": 512, "width": 512}' -o landscape.jpg
```

localhost is the address where you run the webserver.


### Model types

Supported values are:

- `sdxl-anime-legacy` the model used is cagliostrolab/animagine-xl-3.1
- `sdxl-anime` the model used is cagliostrolab/animagine-xl-4.0
- `sdxl-turbo` the model used is stabilityai/sdxl-turbo


## Pipeline cache system

A pipeline cache is present. This cache significantly speeds up the image generation after the first request ...if the model does not change. When instead the request must generate an image of another model, the pipeline is recreated and this requires additional time. In theory this cache flushing could be avoided with systems with enough VRAM but this is beyond the scope of this project, in which I wanted to create something simple and quick.

