import os
from IPython.display import Markdown, display
from openai import OpenAI
import openai

import getpass, os
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter key: ")

messages = [
    {"role": "system", "content": "you are a helpful assistant"},   #The system message (system prompt)
    {"role": "user", "content": "What is 3 + 2?"}                   #The user message (user prompt)
]

response = openai.chat.completions.create(model='gpt-5-mini', messages=messages)
response.choices[0].message.content
