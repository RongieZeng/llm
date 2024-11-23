# Use a pipeline as a high-level helper 
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-7B-Instruct", device=0)
print(pipe(messages, max_length=100, do_sample=True, temperature=0.7))
