from torch.utils import data
from dataset.utils import get_raw_data_dir
from pathlib import Path
from torchvision import datasets
from PIL import Image
from scipy import io


class OfficeCaltech10(data.Dataset):

  def __init__(self, domain, transform=None, target_transform=None):
    self.raw_data_dir = get_raw_data_dir() / Path('OfficeCaltechDomainAdaptation/images') / Path(domain)
    self.transform = transform
    self.target_transform = target_transform
    self.dataset = datasets.ImageFolder(root=self.raw_data_dir)
    self.imgs = self.dataset.imgs

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, item):
    img_path, label = self.imgs[item]

    im = Image.open(img_path)
    im = im.convert('RGB')

    if self.transform:
      im = self.transform(im)

    if self.target_transform:
      label = self.target_transform(label)


    return im, label


if __name__ == '__main__':
  a = OfficeCaltech10('amazon')
  from dataset.img_utils import plot_image
  import random
  for i in random.sample(range(1, len(a)), 3):
    img = a[i][0]
    plot_image(img)