import torch
from SpeeechMetric.utils import get_random_segment

def predict_nMOS(file_path):

    wave = get_random_segment(file_path, target_sr=16000)
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    score = predictor(wave, 16000)
    return score.item()


def predict_nMOS_batch(file_paths, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)

    scores = []
    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i: i + batch_size]

        # Process batch
        batch_audio = [get_random_segment(file, target_sr=16000, segment_length=10).squeeze() for file in batch_files]
        batch_audio = torch.stack(batch_audio).to(device)  # Move to GPU if available

        # print(batch_audio.shape)
        # Predict MOS scores
        batch_scores = predictor(batch_audio, 16000)
        scores.extend(batch_scores.cpu().tolist())  # Move results to CPU for processing

    return scores