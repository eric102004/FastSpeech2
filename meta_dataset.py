import torch
from torch.utils.data import Dataset, Dataloader

import numpy as np
import math

import os

import hparams
import audio as Audio
from utils import pad_1D, pad_2D, process_meta
from text import text_to_sequence, sequence_to_text

device = torch.device('cuda' if torch.cuda.is_available(), else 'cpu')

class meta_Dataset(Dataset):
	def __init__(self, filename='meta_train.txt', sort=True):
		self.basename, self.text = process_meta(os.path.join(hparams.preprocessed_path, filename))
		self.sort = sort

	def __len__(self):
		return len(self.text)

	def __getitem__(self, idx):
		basename = self.basename[idx]
		phone = np.array(text_to_sequence(self.text[idx], []))
		mel_path = os.path.join(hparmas.preprocesed_path, "mel", "{}")
