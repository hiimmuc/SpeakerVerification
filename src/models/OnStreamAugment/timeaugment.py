import torch
import torch.nn as nn
import torch_audiomentations as t_aug
import audiomentations as aug
import glob
from random import random
import os

# torchaudio
class TimeAugment(nn.Module):
    def __init__(self, augment_options, audio_spec, mode='np', **kwargs):
        super(TimeAugment, self).__init__()
        self.augment_chain = augment_options['augment_chain']
        self.musan_path = augment_options['augment_paths']['musan']
        self.noise_vad_path = augment_options['augment_paths']['noise_vad']
        self.rirs_path = augment_options['augment_paths']['rirs']

        self.audio_spec = audio_spec
        self.sr = int(audio_spec['sample_rate'])
        
        self.mode = mode

        # print("Augment set information...")
        # dataset/augment_data/musan_split/ + noise/free-sound/noise-free-sound-0000.wav
        # dataset/augment_data/musan_split/ + music/fma/music-fma-0003.wav
        musan_noise_files = glob.glob(
            os.path.join(self.musan_path, '*/*/*/*.wav'))
        # print(f"Using {len(musan_noise_files)} files of MUSAN noise")

        # custom noise use additive noise latter
        noise_vad_files = glob.glob(
            os.path.join(self.noise_vad_path, '*/*.wav'))
        # print(f"Using {len(noise_vad_files)} files of VAD noise")
        
        # noise from rirs noise
        rir_noise_files = glob.glob(os.path.join(f'{self.rirs_path}/pointsource_noises/', '*.wav')) + glob.glob(
            os.path.join(f'{self.rirs_path}/real_rirs_isotropic_noises/', '*.wav'))
        # print(f"Using {len(rir_noise_files)} files of rirs noise")

        # RIRS_NOISES/simulated_rirs/ + smallroom/Room001/Room001-00001.wav
        reverberation_files = glob.glob(
            os.path.join(f'{self.rirs_path}/simulated_rirs/', '*/*/*.wav'))
        # print(f"Using {len(reverberation_files)} files of rirs reverb")
        
        background_noise_lst = musan_noise_files + noise_vad_files + rir_noise_files
        
        # Initialize augmentation callable
        # usable: shift, PolarityInversion, PitchShift, PeakNormalization(apply_to="only_too_loud_sounds"), Gain, ApplyImpulseResponse, AddColoredNoise, AddBackgroundNoise
        if mode == 'torch':
            self.apply_augmentation = t_aug.Compose(
                transforms=[
                    t_aug.Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, p=0.5),
                    t_aug.Shift(min_shift=-0.25, max_shift=0.25, p=0.25),
                    t_aug.PolarityInversion(p=random()),
                    t_aug.PeakNormalization(apply_to="only_too_loud_sounds", p=1.0),
                    t_aug.AddColoredNoise(min_snr_in_db = 3.0,
                                    max_snr_in_db = 30.0,
                                    min_f_decay = -2.0,
                                    max_f_decay = 2.0, p=0.5),
                    t_aug.AddBackgroundNoise(background_noise_lst, 
                                       min_snr_in_db=3.0, 
                                       max_snr_in_db=30.0, p=0.5),
                    t_aug.ApplyImpulseResponse(reverberation_files, 
                                         compensate_for_propagation_delay=True, p=0.25),
                ]
            )
        else:
            self.apply_augmentation = aug.SomeOf(
                num_transforms=(4, None),
                transforms=[
                    aug.AddBackgroundNoise(background_noise_lst, 
                                       min_snr_in_db=3,
                                       max_snr_in_db=30,
                                       noise_rms="relative",
                                       min_absolute_rms_in_db=-45,
                                       max_absolute_rms_in_db=-15,
                                       p = 0.5),
                    aug.AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=40.0, p=0.5),
                    aug.ApplyImpulseResponse(ir_path=reverberation_files,
                                         p=0.5,
                                         lru_cache_size=128,
                                         leave_length_unchanged=False,),
                    aug.AirAbsorption(min_temperature= 10.0,
                                  max_temperature = 20.0,
                                  min_humidity = 60.0,
                                  max_humidity = 80.0,
                                  min_distance = 0.1,
                                  max_distance = 1.0,
                                  p=0.5),
                    aug.Gain(min_gain_in_db=-6, max_gain_in_db=12, p=0.5),
                    aug.GainTransition(min_gain_in_db= -6.0,
                                   max_gain_in_db = 6.0,
                                   min_duration = 0.1,
                                   max_duration = 2.0,
                                   duration_unit = "seconds",
                                   p = 0.5,),
                    aug.PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.5),
                    aug.PolarityInversion(p=0.5),
                    # RoomSimulator(**room_args),
                    aug.Shift(min_fraction=-0.2,
                          max_fraction=0.2,
                          rollover=True,
                          fade=True,
                          fade_duration=0.01,
                          p=0.5,),
                    aug.TanhDistortion(min_distortion = 0.01, max_distortion = 0.2, p = 0.5),
                    aug.TimeMask(min_band_part=0.0, max_band_part=0.2, fade=True, p=0.5),
                    aug.TimeStretch(min_rate=0.95, max_rate=1.05, leave_length_unchanged=True, p=0.5),
                    ])
            
        
    def forward(self, x):
        if self.mode == 'torch' and len(x.shape) < 3: # batch, channel, amplitude 
            x = x.unsqueeze()
        else:
            pass

        return self.apply_augmentation(x, sample_rate = self.sr)

    
    