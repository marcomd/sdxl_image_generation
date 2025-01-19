# Image generation using a text-to-image model

## Introduction

This simple python script mounts a webserver to process a prompt and return an image

## Install

`pip install -r requirements.txt `

## Execution

```bash
python main.py

Loading pipeline components...: 100%|████████████████████████████████████████████████████| 7/7 [00:16<00:00,  2.40s/it]
Starting SDXL Image Generation Server...
INFO:     Started server process [15920]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```


## Usage

On the client, make the call by setting the authentication key


```bash
curl -X POST http://localhost:8000/generate -H "Authorization: wiv83hveivd83vk83" -H "Content-Type: application/json" -d '{"prompt": "A serene mountain landscape at sunset"}'
```

where localhost is the address on which you run the webserver