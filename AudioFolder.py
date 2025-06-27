
import torchaudio
import torch
from torch.utils.data import Dataset

class audioFileTransformer(Dataset):
  def __init__(self, filePath, transform, targ_sr, num_samples, device):
    super().__init__()
    self.filePath = filePath
    self.transform=transform
    self.targ_sr = targ_sr
    self.num_samples = num_samples
    self.device = device

  def _resample_if_necessary(self, signal, sr):
    if sr != self.targ_sr:
      resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.targ_sr)
      signal = resampler(signal)
    return signal

  def _mixDown_if_necessary(self, signal):
    if signal.shape[0] > 1:
      signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

  def _cut_if_necessary(self, signal):
    if signal.shape[1] > self.num_samples:
      signal = signal[:, :self.num_samples]
    return signal

  def _rightPad_if_necessary(self, signal):
    length_signal = signal.shape[1]
    if length_signal < self.num_samples:
      num_missing_samples = self.num_samples - length_signal
      last_dim_padding = (0, num_missing_samples)
      signal = torch.nn.functional.pad(signal, pad=last_dim_padding)
    return signal

  def __len__(self):
    return 1

  def __getitem__(self, index):
    # audioPath, label = self.getPathWithClass(index)
    # self.transform = self.transform.to(device)
    signal, sr = torchaudio.load(self.filePath) # sr = sample rate
    signal = signal.to(self.device)
    signal = self._cut_if_necessary(signal)
    signal = self._rightPad_if_necessary(signal)
    if self.transform:
      signal = self._resample_if_necessary(signal, sr)
      signal = self._mixDown_if_necessary(signal)
      signal = self.transform(signal)
    return signal#, label