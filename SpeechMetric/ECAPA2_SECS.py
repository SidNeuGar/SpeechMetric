from huggingface_hub import hf_hub_download
from SpeechMetric.utils import get_random_segment

# automatically checks for cached file, optionally set `cache_dir` location
model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir='./cache_models')

import torch

# Load model and move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("SECS Evaluation using ", device)
ecapa2 = torch.jit.load(model_file, map_location=device)
ecapa2 = ecapa2.to(device)
ecapa2.half()  # Optional for faster inference

def predict_SECS(file_path1, file_path2):
    # Process first audio file
    audio1 = get_random_segment(file_path1)

    if audio1.shape[0] > 1:  # If channels > 1, it's stereo
        audio1 = torch.mean(audio1, dim=0, keepdim=True)
    audio1 = audio1.to(device)  # Move audio to GPU

    # Generate embedding for first audio
    embedding_1 = ecapa2(audio1)
    print(embedding_1.shape)

    # Process second audio file
    audio2 = get_random_segment(file_path2)

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


def predict_SECS_batch(file_paths1, file_paths2, batch_size=8):
    assert len(file_paths1) == len(file_paths2), "Both lists must have the same number of file paths."

    SECS_results = []

    for i in range(0, len(file_paths1), batch_size):
        batch_files1 = file_paths1[i:i + batch_size]
        batch_files2 = file_paths2[i:i + batch_size]

        audio1_list, audio2_list = [], []

        for file1, file2 in zip(batch_files1, batch_files2):
            audio1 = get_random_segment(file1, segment_length=10)
            audio2 = get_random_segment(file2, segment_length=10)

            # Convert stereo to mono if necessary
            if audio1.shape[0] > 1:
                audio1 = torch.mean(audio1, dim=0, keepdim=True)
            if audio2.shape[0] > 1:
                audio2 = torch.mean(audio2, dim=0, keepdim=True)

            audio1_list.append(audio1)
            audio2_list.append(audio2)

        # Stack and move to device
        audio1_batch = torch.stack(audio1_list).to(device)
        audio2_batch = torch.stack(audio2_list).to(device)

        # Compute embeddings
        embedding_1_batch = ecapa2(audio1_batch)
        embedding_2_batch = ecapa2(audio2_batch)

        # Compute cosine similarity
        batch_SECS = torch.nn.functional.cosine_similarity(embedding_1_batch, embedding_2_batch, dim=1)
        SECS_results.extend(batch_SECS.cpu().tolist())

    return SECS_results