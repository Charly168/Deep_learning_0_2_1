from PIL import Image
from torch.utils.data import Dataset
class Mydataset(Dataset):
    def __init__(self,root,transforms):
        self.root = root
        self.images,self.labels = self.readData()
        self.transforms = transforms


    def __getitem__(self, item):
        image = Image.open(self.images[item])
        label = self.labels[item]
        if self.transforms:
            image = self.transforms(image)

        return image,label

    def __len__(self):
        return len(self.images)

    def readData(self):
        images = []
        labels = []
        for data in self.root:
            images.append(data["image"])
            labels.append(data["label"])

        return images,labels






