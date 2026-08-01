import json
classes = 8
with open("training/history.json", "r") as f:
    # print(f)
    history = json.load(f)

print(len(history))

# visualizations

from matplotlib import pyplot as plt

# train and test loss over epochs

plt.plot([i for i in range(1,len(history))], [history[i][0] for i in range(len(history))], label="training", color="blue")
plt.plot([i for i in range(1,len(history))], [history[i][4] for i in range(len(history))], label="validation", color="red")

plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.title("Average Loss per Epoch")

plt.legend()
plt.savefig("loss")
plt.show()


# train and test accuracy over epochs

plt.plot([i for i in range(1,len(history)+1)], [history[i][1] for i in range(len(history))], label="training", color="blue", ls="--")
plt.plot([i for i in range(1,len(history)+1)], [history[i][5] for i in range(len(history))], label="validation", color="red", ls="--")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")

plt.legend()
plt.savefig("acc")
plt.show()


# loss of each class over epochs

colors = ["red","orange","yellow","green","blue","cyan","purple","black"]

# training
for j in range(classes):
    plt.plot([i for i in range(1,len(history)+1)], [history[i][2][j] for i in range(len(history))], label=f"{traindata.classes[j]}", color=colors[j], ls="-")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Average Loss per Class per Epoch (training)")

plt.legend()
plt.savefig("classlosstrain")
plt.show()

# validation
for j in range(classes):
    plt.plot([i for i in range(1,len(history)+1)], [history[i][6][j] for i in range(len(history))], label=f"{testdata.classes[j]}", color=colors[j], ls="--")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Average Loss per Class per Epoch (validation)")

plt.legend()
plt.savefig("classlossval")
plt.show()


# accuracy of each class over epochs

# training
for j in range(classes):
    plt.plot([i for i in range(1,len(history)+1)], [history[i][3][j] for i in range(len(history))], label=f"{traindata.classes[j]}", color=colors[j])

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy per Class per Epoch (training)")

plt.legend()
plt.savefig("classacctrain")
plt.show()

# validation
for j in range(classes):
    plt.plot([i for i in range(1,len(history)+1)], [history[i][7][j] for i in range(len(history))], label=f"{testdata.classes[j]}", color=colors[j])

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy per Class per Epoch (validation)")

plt.legend()
plt.savefig("classaccval")
plt.show()