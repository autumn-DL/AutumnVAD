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
from lib.data_noise_aug import DataAugNoise
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
                    i['target'] = i['target'][start:start + base_len]
                    i['mask'] = torch.zeros(base_len)
                elif L == base_len:
                    i['mask'] = torch.zeros(base_len)
                else:
                    p = base_len - L
                    i['mel'] = torch.nn.functional.pad(i['mel'], (0, 0, 0, p))
                    i['target'] = torch.nn.functional.pad(i['target'], ( 0, p))
                    i['mask'] = torch.cat([torch.zeros(L), torch.ones(p)], dim=0)
        spectrogram = torch.stack([record['mel'] for record in minibatch if 'mel' in record])
        target = torch.stack([record['target'] for record in minibatch if 'target' in record])
        if self.infer:
            return {'mel': spectrogram.detach(), 'mask': None, 'target': target}
        if self.config['pad_strategy'] == 'min':
            return {'mel': spectrogram.detach(), 'mask': None, 'target': target}
        else:
            masks = torch.stack([record['mask'] for record in minibatch if 'mask' in record])
            return {'mel': spectrogram.detach(), 'mask': masks, 'target': target}


class VAD_DataSet_norm(Dataset):
    def __init__(self, config, infer=False):
        super().__init__()
        if not is_main_process:
            torch.set_num_threads(2)
        self.max_mel_farms = config['max_mel_farms']
        self.melhop=config['hop_size']
        self.wav2mel = PitchAdjustableMelSpectrogram(
            sample_rate=config['audio_sample_rate'],
            n_fft=config['fft_size'],
            win_length=config['win_size'],
            hop_length=config['hop_size'],
            f_min=config['fmin'],
            f_max=config['fmax'],
            n_mels=config['audio_num_mel_bins'],
        )
        self.rms_hop = config['hop_size']
        self.rms_win = config['rms_win_size']
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
        # self.noise_aug_prob = config['noise_aug_prob']
        self.use_noise_aug = config['use_noise_aug']
        self.noise_aug_obj = DataAugNoise(config)

        self.use_mel_speed_shift = config['use_mel_speed_shift']
        self.mel_speed_shift_prob = config['mel_speed_shift_prob']
        self.mel_speed_shift_max = config['mel_speed_shift_max']
        self.mel_speed_shift_min = config['mel_speed_shift_min']
        self.use_mel_key_shift = config['use_mel_key_shift']
        self.mel_key_shift_prob = config['mel_key_shift_prob']
        self.mel_key_shift_min = config['mel_key_shift_min']
        self.mel_key_shift_max = config['mel_key_shift_max']

        self.use_empty_voice = config['use_empty_voice']
        self.empty_voice_prob = config['empty_voice_prob']

        self.voice_volume_aug_prob = config['voice_volume_aug_prob']
        self.voice_volume_aug = config['voice_volume_aug']

    def pre_music(self,lens,music):
        ml=len(music)
        if lens>ml:
            music=torch.nn.functional.pad(music,(0,lens-ml))
        elif lens<ml:
            start = np.random.randint(0, ml - lens)
            music = music[start: start + lens]
        return music
    def audio_mix(self, voice, music, rate=0.5):
        music=self.pre_music(music=music,lens=len(voice))
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
        voice=torch.unsqueeze(voice[0],0)
        if sr_voice != self.sr:
            voice = torchaudio.transforms.Resample(orig_freq=sr_voice, new_freq=self.sr)(voice)
        if self.max_mel_farms is not  None and not self.infer:
            Lvocie=len(voice[0])
            start = np.random.randint(0, Lvocie - self.max_mel_farms*self.melhop)
            voice=voice[:,start:start+self.max_mel_farms*self.melhop]
            # voice=voice[:self.max_mel_farms*self.melhop]

        if self.key_shift_aug and random.random() < self.key_shift_aug_prob and not self.infer:
            key_s = random.uniform(self.min_key_shift, self.max_key_shift)
            voice = wav_aug(voice, self.hop_size, speed=key_s)

        RMSX = get_rms(voice[0], frame_length=self.rms_win,
                       hop_length=self.rms_hop, )
        if self.voice_volume_aug and random.random() < self.voice_volume_aug_prob:
            voice = self.volume_augmentation(voice)
        if self.use_empty_voice and random.random() < self.empty_voice_prob:
            voice = torch.randn_like(voice)*1e-5
            target = torch.ones_like(RMSX).long()
        else:
            target = (RMSX < self.rmsa).long()
        if self.add_music and random.random() < self.add_music_prob:
            music_path = self.music_datapath[random.randint(0, self.music_len - 1)]
            rate = random.uniform(self.music_mix_min, self.music_mix_max)
            music, sr_music = torchaudio.load(music_path)
            music = torch.unsqueeze(music[0], 0)
            if sr_music != self.sr:
                music = torchaudio.transforms.Resample(orig_freq=sr_music, new_freq=self.sr)(music)
            voice[0] = self.audio_mix(voice[0], music[0], rate)
        if self.volume_aug and random.random() < self.volume_aug_prob:
            voice = self.volume_augmentation(voice)
        if self.use_noise_aug:
            voice = self.noise_aug_obj.add_noise(voice)

        speed = 1
        keys = 0
        # if self.use_mel_speed_shift and random.random() < self.mel_speed_shift_prob:
        #     speed = random.uniform(self.mel_speed_shift_min, self.mel_speed_shift_max)
        if self.use_mel_key_shift and random.random() < self.mel_key_shift_prob:
            keys = random.uniform(self.mel_key_shift_min, self.mel_key_shift_max)

        mels = self.wav2mel.dynamic_range_compression_torch(self.wav2mel(voice, speed=speed, key_shift=keys),
                                                            clip_val=1e-6)

        return {'mel': mels[0].T, 'target': target}

    def __len__(self):
        return len(self.datapath)
    def collater(self):
        co = collater(config=self.config, infer=self.infer)
        return co.collater_fn
