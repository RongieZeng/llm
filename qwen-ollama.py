import requests
import json

def qwen_ollama(prompt):
    # url = "http://localhost:11434/v1/chat/completions"
    url = "http://localhost:11434/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer ollama" # 替换为你的API密钥
    }
    data = {
        "model": "qwen2.5:14b",
        "prompt": prompt,
        # "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()
    # return result["choices"][0]["message"]["content"]
    return result["choices"][0]["text"]

# 情感分析
prompt = "好好好，您说的都对###请分析一下上面这句话的情绪，并给出一个情感分析结果。使用json格式返回（只返回json结果）,不要输出其他内容。,注意,json的key用英文，不要用中文。json的value可以用中文，格式如下：{\"sentiment\":\"不满\",\"confidence\":0.95}"
response = qwen_ollama(prompt)
print(response)

# 实体识别
prompt = "我在公园上城这里呢，华标说他在公司###请识别上面这句话中的实体，并给出一个实体识别结果。使用json格式返回（只返回json结果）,不要输出其他内容。,注意,json的key用英文，不要用中文。json的value可以用中文，格式严格按下面：{\"entity\":\"实体1\",\"value\":\"实体的值\"}"
response = qwen_ollama(prompt)
print(response)