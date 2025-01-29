from huggingface_hub import hf_hub_download

# automatically checks for cached file, optionally set `cache_dir` location
model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)

import torch
import torchaudio

# Load model and move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ecapa2 = torch.jit.load(model_file, map_location=device)
ecapa2 = ecapa2.to(device)
ecapa2.half()  # Optional for faster inference

def predict_SECS(file_path1, file_path2):
    # Process first audio file
    audio1, sr1 = torchaudio.load(file_path1)
    if audio1.shape[0] > 1:  # If channels > 1, it's stereo
        audio1 = torch.mean(audio1, dim=0, keepdim=True)
    audio1 = audio1.to(device)  # Move audio to GPU

    # Generate embedding for first audio
    embedding_1 = ecapa2(audio1)

    # Process second audio file
    audio2, sr2 = torchaudio.load(file_path2)
    if audio2.shape[0] > 1:  # If channels > 1, it's stereo
        audio2 = torch.mean(audio2, dim=0, keepdim=True)
    audio2 = audio2.to(device)  # Move audio to GPU


    # Generate embedding for second audio
    embedding_2 = ecapa2(audio2)

    # Compute cosine similarity
    SECS = torch.dot(embedding_1.squeeze(), embedding_2.squeeze()) / (
        torch.linalg.norm(embedding_1) * torch.linalg.norm(embedding_2)
    )
    # print(f"Cosine Similarity (SECS): {SECS.item()}")
    return SECS.item()