import requests

params = {"model": "llama3", "prompt": "Test embedding for Giana observer"}

response = requests.post("http://localhost:11434/api/embeddings", json=params)
response.raise_for_status()
data = response.json()
print("len", len(data["embedding"]))
