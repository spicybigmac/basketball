import json
classes = 8
with open("training/history.json", "r") as f:
    history = json.loads(f)

print(len(history))

