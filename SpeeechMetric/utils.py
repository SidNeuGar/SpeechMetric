import torch
import torchaudio
import random

def get_random_segment(file_path, segment_length=15, target_sr=16000):
    audio, sr = torchaudio.load(file_path)
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(audio)

    # Convert stereo to mono if needed
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    num_samples = audio.shape[1]
    segment_samples = segment_length * target_sr

    if num_samples <= segment_samples:
        return audio  # Return the whole audio if it's shorter than 15 seconds

    # Select a random starting position
    start_sample = random.randint(0, num_samples - segment_samples)
    return audio[:, start_sample: start_sample + segment_samples]