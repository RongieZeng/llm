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
    response = requests.post(url, headers=headers, json = data)
    result = response.json()
    # return result["choices"][0]["message"]["content"]
    return result["choices"][0]["text"]

### 提示模板
# 情感分析
sentimentPromptTemplate = "%s###请分析一下上面这句话的情绪，并给出一个情感分析结果。输出结果必须被json.loads接受。,注意,json的key用英文，不要用中文。json的value可以用中文，格式如下：{\"sentiment\":\"不满\",\"confidence\":0.95}"

# 实体识别
entityPromptTemplate = "%s###请识别上面这句话中的实体，并给出一个实体识别结果。使用json格式返回（只返回json结果）,不要输出其他内容。,注意,json的key用英文.不要用中文.json的value可以用中文，格式严格按下面：{\"entity\":\"实体1\",\"value\":\"实体的值\"}"

# 摘要
summaryPromptTemplate = "%s###请对上面这句话进行摘要，并给出一个摘要结果。输出结果必须被json.loads接受。,注意,json的key用英文，不要用中文。json的value可以用中文，格式如下：{\"summary\":\"摘要结果\"}"

### 执行示例
sentimentPrompt = sentimentPromptTemplate % "好好好，您说的都对"
response = qwen_ollama(sentimentPrompt)
print(response)

entityPrompt = entityPromptTemplate % "先生，你好，您现在在哪里啊呀？ 哦，我们现在在五楼啊，您呢？在哪里的五楼？在增城，增城新塘华润公园上城这里哦"
response = qwen_ollama(entityPrompt)
print(response)

summaryPrompt = summaryPromptTemplate % "先生，你好，您现在在哪里啊呀？ 哦，我们现在在五楼啊，您呢？在哪里的五楼？在增城，增城新塘华润公园上城这里哦"
response = qwen_ollama(summaryPrompt)

print(response)