# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

text = """
You are a helpful code assistant. Your task is to generate a valid JSON object based on the given information:
name: John
lastname: Smith
address: #1 Samuel St.
Just generate the JSON object without explanations:
"""

completion = client.chat.completions.create(
  model="local-model", # this field is currently unused
  messages=[    
    {"role": "user", "content": text}
  ],
  temperature=0.7,
)

print(completion.choices[0].message.content)