from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import soundfile as sf
import librosa
import jiwer

_model_cache = {}
_processor_cache = {}


def load_model_and_processor(model_name):
    """
    Load the model and processor only once and cache them.
    """
    if model_name not in _model_cache:
        _processor_cache[model_name] = WhisperProcessor.from_pretrained(model_name)
        _model_cache[model_name] = WhisperForConditionalGeneration.from_pretrained(model_name)

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model_cache[model_name].to(device)
        print(f"Model loaded on {device}")
    return _processor_cache[model_name], _model_cache[model_name]

def predict_CER(file_path, ground_truth, whisper_model=None):

    processor, model = load_model_and_processor(whisper_model)

    device = next(model.parameters()).device

    # Load the audio file using soundfile
    audio_file = file_path
    audio, sr = sf.read(audio_file)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)

    with torch.no_grad():
        # Handle suppress_tokens issue
        predicted_ids = model.generate(input_features, suppress_tokens=[50256, 50362])

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    cer = jiwer.cer(transcription.strip(), ground_truth.strip())
    return cer

