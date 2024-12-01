import ollama
import requests
import json

# 计算器
def add_two_numbers(a: int, b: int) -> int:
  """
  Add two numbers

  Args:
    a: The first integer number
    b: The second integer number

  Returns:
    int: The sum of the two numbers
  """
  a = int(a)
  b = int(b)
  print(f"Adding {a} and {b}")
  return a + b

def sub_two_numbers(a: int, b: int) -> int:
  """
  Add two numbers

  Args:
    a: The first integer number
    b: The second integer number

  Returns:
    int: The subtraction of the two numbers
  """
  a = int(a)
  b = int(b)
  print(f"Subing {a} and {b}")
  return a - b

response = ollama.chat(
  'llama3.2:latest',
  messages=[{'role': 'user', 'content': 'What is 20 - 1?'}],
  tools=[add_two_numbers, sub_two_numbers], # Actual function reference
)

available_functions = {
  'add_two_numbers': add_two_numbers,
  "sub_two_numbers": sub_two_numbers
}

print('Response:',  json.dumps(response, default=lambda o: o.__dict__, indent=4))
for tool in response.message.tool_calls or []:
  function_to_call = available_functions.get(tool.function.name)
  if function_to_call:
    print('Function output:', function_to_call(**tool.function.arguments))
  else:
    print('Function not found:', tool.function.name)

available_functions = {
  'request': requests.request,
}

# 抓网页
response = ollama.chat(
  'llama3.2:latest',
  messages=[{
    'role': 'user',
    'content': 'get the http://www.baidu.com webpage?',
  }],
  tools=[requests.request], 
)

print('Response:',  json.dumps(response, default=lambda o: o.__dict__, indent=4))

for tool in response.message.tool_calls or []:
  function_to_call = available_functions.get(tool.function.name)
  if function_to_call == requests.request:
    # Make an HTTP request to the URL specified in the tool call
    resp = function_to_call(
      method=tool.function.arguments.get('method'),
      url=tool.function.arguments.get('url'),
    )
    # print(resp.text)
  else:
    print('Function not found:', tool.function.name)