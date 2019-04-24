import os
from pathlib import Path


def get_data_dir():
  return Path(os.getenv('DATA_DIR', '/data'))

def get_raw_data_dir():
  return get_data_dir() / 'raw'
