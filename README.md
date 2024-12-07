
set GENERATION_AUTH_KEY=wiv83hveivd83vk83


curl -X POST http://localhost:8000/generate -H "Authorization: wiv83hveivd83vk83" -H "Content-Type: application/json" -d '{"prompt": "A serene mountain landscape at sunset"}'