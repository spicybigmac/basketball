import json
classes = 8
with open("training/history.json", "r") as f:
    # print(f)
    history = json.load(f)

print(len(history))

