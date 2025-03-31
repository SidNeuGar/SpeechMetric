import torch
import torchaudio
from torchaudio.pipelines import WAVLM_BASE_PLUS
from SpeeechMetric.utils import get_random_segment

# Load WavLM model
bundle = WAVLM_BASE_PLUS
wavlm_model = bundle.get_model()
wavlm_model = wavlm_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def extract_embedding(file_path):
    waveform = get_random_segment(file_path)  # Load audio segment

    if waveform.shape[0] > 1:  # Convert stereo to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform = waveform.to(device)

    # Extract features
    with torch.no_grad():
        features = wavlm_model.extract_features(waveform)[0]  # Extract the first tensor
        features = features[0]


    return features.mean(dim=1)  # Mean pooling over time


def predict_SECS(file_path1, file_path2):
    embedding_1 = extract_embedding(file_path1)
    embedding_2 = extract_embedding(file_path2)

    SECS = torch.nn.functional.cosine_similarity(embedding_1, embedding_2)
    return SECS.item()
