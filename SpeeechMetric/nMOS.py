import torch
import librosa

def predict_nMOS(file_path):
    wave, sr = librosa.load(file_path, sr=None, mono=True)
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    score = predictor(torch.from_numpy(wave).unsqueeze(0), sr)
    return score.item()