import pathlib
import random

import numpy as np
import torch
import torchaudio
# import torch.utils.data.dataset as dataset
from torch.utils.data import Dataset

from torch.multiprocessing import current_process
import re

from lib.RMS import db2rms, get_rms
from lib.wav2spec import PitchAdjustableMelSpectrogram

is_main_process = not bool(re.match(r'((.*Process)|(SyncManager)|(.*PoolWorker))-\d+', current_process().name))


def wav_aug(wav, hop_size, speed=1.):
    orig_freq = int(np.round(hop_size * speed))
    new_freq = hop_size
    return torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)(wav)


class collater:
    def __init__(self, config, infer=False):
        super().__init__()
        self.config = config
        self.infer = infer

    def get_base_len(self, minibatch):
        if self.config['pad_strategy'] == 'min':
            base_len = None
            for i in minibatch:
                if i['mel'] is None:
                    continue
                lens = len(i['mel'])
                if base_len is None:
                    base_len = lens
                if base_len > lens:
                    base_len = lens
        elif self.config['pad_strategy'] == 'max':
            base_len = None
            for i in minibatch:
                if i['mel'] is None:
                    continue
                lens = len(i['mel'])
                if base_len is None:
                    base_len = lens
                if base_len < lens:
                    base_len = lens
        elif self.config['pad_strategy'] == 'middle':
            t_list = []

            for i in minibatch:
                if i['mel'] is None:
                    continue
                lens = len(i['mel'])
                t_list.append(lens)
            base_len = np.median(t_list)
        else:
            raise RuntimeError('')
        return int(base_len)

    @torch.no_grad()
    def collater_fn(self, minibatch):
        base_len = self.get_base_len(minibatch=minibatch)
        if not self.infer:
            for i in minibatch:
                if i['mel'] is None:
                    # i['mask'] = None
                    del i['mel']
                    continue
                L = len(i['mel'])
                if L > base_len:
                    start = np.random.randint(0, L - base_len)
                    i['mel'] = i['mel'][start:start + base_len]
                    i['mask'] = torch.zeros(base_len)
                elif L == base_len:
                    i['mask'] = torch.zeros(base_len)
                else:
                    p = base_len - L
                    i['mel'] = torch.nn.functional.pad(i['mel'], (0, 0, 0, p))
                    i['mask'] = torch.cat([torch.zeros(L), torch.ones(p)], dim=0)
        spectrogram = torch.stack([record['mel'] for record in minibatch if 'mel' in record])
        if self.infer:
            return {'mel': spectrogram.detach(), 'mask': None}
        if self.config['pad_strategy'] == 'min':
            return {'mel': spectrogram.detach(), 'mask': None}
        else:
            masks = torch.stack([record['mask'] for record in minibatch if 'mask' in record])
            return {'mel': spectrogram.detach(), 'mask': masks}


class VAD_DataSet_norm(Dataset):
    def __init__(self, config, infer=False):
        super().__init__()
        if not is_main_process:
            torch.set_num_threads(2)
        self.max_mel_farms = config['max_mel_farms']
        self.wav2mel = PitchAdjustableMelSpectrogram(
            sample_rate=config['audio_sample_rate'],
            n_fft=config['fft_size'],
            win_length=config['win_size'],
            hop_length=config['hop_size'],
            f_min=config['fmin'],
            f_max=config['fmax'],
            n_mels=config['audio_num_mel_bins'],
        )
        self.hop_size = config['hop_size']
        self.sr = config['audio_sample_rate']
        self.infer = infer
        self.rmsa = db2rms(config['DB_rms'])
        self.config = config

        data_root_path = pathlib.Path(config['binary_index_dir'])
        data_name = config['binary_index_name']
        if not infer:
            datapc = data_root_path / f'{data_name}_train.tsv'
        else:
            datapc = data_root_path / f'{data_name}_val.tsv'

        with open(datapc, 'r', encoding='utf8') as f:
            self.datapath = f.read().strip().split('\n')
        with open(data_root_path / f'{data_name}_misc.tsv', 'r', encoding='utf8') as f:
            self.music_datapath = f.read().strip().split('\n')
        self.music_len = len(self.music_datapath)

        self.add_music_prob = config['add_music_prob']
        self.add_music = config['add_music']

        self.music_mix_min = config['music_mix_min']
        self.music_mix_max = config['music_mix_max']

        self.volume_aug_prob = config['volume_aug_prob']
        self.volume_aug = config['volume_aug']

        self.key_shift_aug_prob = config['key_shift_aug_prob']
        self.key_shift_aug = config['key_shift_aug']
        self.max_key_shift = config['max_key_shift']
        self.min_key_shift = config['min_key_shift']
        self.noise_aug_prob = config['noise_aug_prob']
        self.noise_aug = config['noise_aug']

    def audio_mix(self, voice, music, rate=0.5):
        voice = voice * rate
        music = music * (1 - rate)
        return voice + music

    def volume_augmentation(self, voice, ):

        max_amp = float(torch.max(torch.abs(voice))) + 1e-5
        max_shift = min(3, np.log(1 / max_amp))
        log_mel_shift = random.uniform(-3, max_shift)

        voice *= np.exp(log_mel_shift)

        return voice

    def __getitem__(self, index):
        data_path = self.datapath[index]
        voice, sr_voice = torchaudio.load(data_path)
        if sr_voice != self.sr:
            voice = torchaudio.transforms.Resample(orig_freq=sr_voice, new_freq=self.sr)(voice)

        if self.key_shift_aug and random.random() < self.key_shift_aug_prob:
            key_s = random.uniform(self.min_key_shift, self.max_key_shift)
            voice = wav_aug(voice, self.hop_size, speed=key_s)

        RMSX = get_rms(voice[0])

        target = (RMSX < self.rmsa).long()
        if self.add_music and random.random() < self.add_music_prob:
            music_path = self.music_datapath[random.randint(0, self.music_len - 1)]
            rate = random.uniform(self.music_mix_min, self.music_mix_max)
            music, sr_music = torchaudio.load(music_path)
            if sr_music != self.sr:
                music = torchaudio.transforms.Resample(orig_freq=sr_music, new_freq=self.sr)(music)
            voice = self.audio_mix(voice, music, rate)


