import numpy as np
from torch.utils import data
from torchvision import datasets

from dataset.utils import get_raw_data_dir


class SVHNDataset(data.Dataset):
  def __init__(self, train=True, labels=[], transform=None):
    self.raw_data_dir = get_raw_data_dir() / "svhn"

    self.transform = transform

    split = "train" if train else "test"

    self.dataset = datasets.SVHN(
      root=self.raw_data_dir, split=split, download=True, transform=transform
    )

    self.idx = []
    class_labels = self.dataset.labels
    if len(labels) > 0:
      for label in labels:
        self.idx.extend(np.where(class_labels == label)[0])
    else:
      self.idx = list(range(len(class_labels)))

    super(SVHNDataset, self).__init__()

  def __len__(self):
    return len(self.idx)

  def __getitem__(self, item):
    return self.dataset[self.idx[item]]


if __name__ == "__main__":
  dataset = SVHNDataset()
  from dataset.img_utils import plot_image

  plot_image(dataset[0][0])
