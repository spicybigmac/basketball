import json
classes = 8
with open("training/history.json", "r") as f:
    # print(f)
    history = json.load(f)

print(len(history))

# we need to subtract the mean color and divide by the standard deviation of the Kinetics-400 dataset
# this makes it easier to train the model which is already used to these colors



traintrans = v2.Compose([
    v2.Resize([112,112], antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

testtrans = v2.Compose([
    v2.Resize([112,112], antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

class BasketballDataset(Dataset):
    def __init__(self, training, datadir, frames=16, samplesperclass=5):
        self.datadir = Path(datadir)
        self.goalframes = frames
        self.training = training
        
        # use sorted here to ensure determinism so the labels arent cooked if the drive is shuffled
        self.classes = sorted([d.name for d in self.datadir.iterdir() if d.is_dir() and not d.name[0] == "."])

        videopaths, labels = [], []

        for i,name in enumerate(self.classes):
            classdir = self.datadir / name

            # use sorted here to ensure determinism so i can debug a specific video index
            vids = sorted([f for f in classdir.iterdir() if f.is_file() and f.suffix.lower() == ".mp4"])

            for path in vids[:samplesperclass]:
                # this only takes the first samplesperclass videos from each class
                videopaths.append(path)
                labels.append(i)

        # self._cache = defaultdict() # cache for less cpu work
        
        # im js a 42-using sheep
        trainpaths, testpaths, trainlabels, testlabels = train_test_split(videopaths, labels, test_size=0.2, stratify=labels, random_state=42)

        if self.training:
            self.videopaths = trainpaths
            self.labels = trainlabels
        else:
            self.videopaths = testpaths
            self.labels = testlabels

        print(f"loaded {len(self.videopaths)} video paths for {'training' if training else 'testing'} :3")

    def __len__(self):
        # dataloader uses this function for like progress bars and stuff
        return len(self.videopaths)

    def __getitem__(self, idx):
        # dataloader always calls this function with idx as the only parameter
        path = self.videopaths[idx]
        label = self.labels[idx]

        frames = self._load_video(path)

        vidtensor = torch.from_numpy(frames)
        # the v2 transform function looks at the last 2 dimensions, so we need to make them height and width respectively
        vidtensor = vidtensor.permute(0,3,1,2)
        if self.training:
            vidtensor = traintrans(vidtensor)
        else:
            vidtensor = testtrans(vidtensor)
        vidtensor = vidtensor.permute(1,0,2,3)

        return vidtensor, label

    def _load_video(self, path):
        cap = cv2.VideoCapture(str(path))
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if framecount < self.goalframes:
            # note: this will NEVER HAPPEN unless a video file gets corrupted
            idxs = np.pad(np.arange(framecount), (0, self.goalframes - framecount), mode='edge')
        else:
            idxs = np.linspace(0, framecount - 1, self.goalframes, dtype=int)

        idxs = set(idxs)
        frames = []
        frameidx = 0

        while len(frames) < self.goalframes:
            ret, frame = cap.read()

            if ret:
                if frameidx in idxs:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert color
                    frames.append(frame)
            else:
                # videos broken so extend the last working frame otherwise black screen
                while len(frames) < self.goalframes:
                    frames.append(frames[-1] if len(frames) > 0 else np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8))

            frameidx += 1

        cap.release()
        return np.stack(frames)
traindata = BasketballDataset(True, datasetpath, frames, samplesperclass)

# visualizations

from matplotlib import pyplot as plt

# train and test loss over epochs

plt.plot([i for i in range(1,len(history)+1)], [history[i][0] for i in range(len(history))], label="training", color="blue")
plt.plot([i for i in range(1,len(history)+1)], [history[i][4] for i in range(len(history))], label="validation", color="red")

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