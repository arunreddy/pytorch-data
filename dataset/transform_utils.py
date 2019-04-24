from torchvision import transforms

def transform_mnist(img_size=28):
  return transforms.Compose(
    [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
  )

def transform_usps(img_size=28):
  return transforms.Compose(
    [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
  )