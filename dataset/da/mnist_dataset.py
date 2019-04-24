import numpy  as np
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import transforms

from dataset.utils import get_raw_data_dir


class MNISTDataset(data.Dataset):

  def __init__(self, train=True, labels=[], transform=None, img_size=28):
    self.raw_data_dir = get_raw_data_dir()

    self.transform = transform

    self.dataset = datasets.MNIST(root=self.raw_data_dir,
                                  train=train,
                                  download=True,
                                  transform=transform)

    self.idx = []
    if len(labels) > 0:
      class_labels = self.dataset.targets.numpy()
      for label in labels:
        self.idx.extend(np.where(class_labels == label)[0])
    else:
      self.idx = list(range(self.dataset.targets.size()[0]))

    super(MNISTDataset, self).__init__()


  def __len__(self):
    return len(self.idx)


  def __getitem__(self, item):
    return self.dataset[self.idx[item]]
