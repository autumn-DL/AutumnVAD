import pathlib

import click

from lib.wav2spec import PitchAdjustableMelSpectrogram
from model_trainer.basic_lib.config_util import get_config
from model_trainer.basic_lib.find_last_checkpoint import get_latest_checkpoint_path

import torch

import torchaudio

import numpy as np
import torch.nn.functional as F

from trainCLS.VADCLS_V2_BIGRU import VADCLS

@torch.no_grad()
@click.command(help='')
@click.option('--exp_name', required=False, metavar='EXP', help='Name of the experiment')
@click.option('--ckpt_path', required=False, metavar='FILE', help='Path to the checkpoint file')
@click.option('--save_path', required=True, metavar='FILE', help='Path to save the exported checkpoint')
@click.option('--work_dir', required=False, metavar='DIR', help='Working directory containing the experiments')
@click.option('--wav_path', required=True, metavar='DIR', help='Working directory containing the experiments')
def export(exp_name, ckpt_path, save_path, work_dir,wav_path):
    # print_config(config)
    if exp_name is None and ckpt_path is None:
        raise RuntimeError('Either --exp_name or --ckpt_path should be specified.')
    if ckpt_path is None:
        if work_dir is None:
            work_dir = pathlib.Path(__file__).parent / 'experiments'
        else:
            work_dir = pathlib.Path(work_dir)
        work_dir = work_dir / exp_name
        assert not work_dir.exists() or work_dir.is_dir(), f'Path \'{work_dir}\' is not a directory.'
        ckpt_path = get_latest_checkpoint_path(work_dir)

    config_file = pathlib.Path(ckpt_path).with_name('config.yaml')
    config = get_config(config_file)


    mel_spec_transform = PitchAdjustableMelSpectrogram(sample_rate=config['audio_sample_rate'],
                                                       n_fft=config['fft_size'],
                                                       win_length=config['win_size'],
                                                       hop_length=config['hop_size'],
                                                       f_min=config['fmin'],
                                                       f_max=config['fmax'],
                                                       n_mels=config['audio_num_mel_bins'], )

    model=VADCLS(config)
    sc=torch.load(ckpt_path,map_location='cpu')
    model.load_state_dict(torch.load(ckpt_path,map_location='cpu')['model'])
    model.eval()
    model=model.cuda()

    audio,sr=torchaudio.load(wav_path)
    audio=audio[0]
    audio=audio[None,:]
    if sr!=config['audio_sample_rate']:
        audio=torchaudio.transforms.Resample(sr,config['audio_sample_rate'])(audio)
    mel = mel_spec_transform.dynamic_range_compression_torch(mel_spec_transform(audio),clip_val=1e-6)
    mel=torch.transpose(mel,1,2)
    P = model(mel.cuda())
    P=torch.sigmoid(P)
    P1=F.interpolate(P,scale_factor=config['hop_size'],mode='linear')[0]
    P1=(P1>0.5).long()
    PR=len(P1[0])
    AR=len(audio[0])
    if PR>AR:
        P2=P1[:,:AR]
    else:
        P2=F.pad(P1,(0,AR-PR),'constant',0)
    P2=P2.cpu()
    P3=(P2==0).long()
    audx=audio*P2
    torchaudio.save(save_path+'_1.wav',audx,config['audio_sample_rate'])
    torchaudio.save(save_path+'_2.wav', audio*P3, config['audio_sample_rate'])
    pass

if __name__ == '__main__':
    export()