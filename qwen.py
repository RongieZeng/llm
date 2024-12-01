# Use a pipeline as a high-level helper 
from transformers import pipeline,AutoTokenizer,AutoModelForSequenceClassification
from torch import nn

messages = [
    {"role": "user", "content": "你是谁?"},
]
# pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device=0)
# print(pipe(messages, max_length=100, do_sample=True, temperature=0.7))


# pipe = pipeline("sentiment-analysis",model="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment", device=0)

model = AutoModelForSequenceClassification.from_pretrained("IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment")
tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment")
# classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)
# print(classifier("Qwen 是一个很差劲的人工智能助手"))
encoding = tokenizer("Qwen 是一个很差劲的人工智能助手")
print(encoding)

pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
pt_outputs = model(**pt_batch)
pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)

pt_save_directory = "./pt_save_pretrained"
tokenizer.save_pretrained(pt_save_directory)
model.save_pretrained(pt_save_directory)

tokenizer = AutoTokenizer.from_pretrained(pt_save_directory)
pt_model = AutoModelForSequenceClassification.from_pretrained(pt_save_directory, from_pt=True)