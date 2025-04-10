import os
import uuid
import csv
import subprocess
import ffmpeg
import librosa
import torch
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pydub import AudioSegment, silence
from transformers import pipeline as hf_pipeline

# --- CONFIG ---
# Change this to your root input directory (will search recursively)
INPUT_DIR = "/home/sid/Audio_Data/Voices_Hollywood_Bollywood/Hollywood"
# Base output directory: each processed file gets a unique subfolder here.
OUTPUT_ROOT = "/home/sid/Code/diarization/processed_Hollywood"
MASTER_TRANSCRIPT_CSV = "/home/sid/Code/diarization/master_transcript.csv"

STANDARDIZED_AUDIO = "standardized.wav"
CLEANED_AUDIO = "cleaned.wav"
DIARIZATION_RTTM = "diarization.rttm"
SPEAKER_SEGMENTS_DIR = "speaker_segments"
TRANSCRIPT_FILE = "speaker_transcript.txt"
MAX_SEGMENT_DURATION = 15  # seconds
SILENCE_THRESH = -40  # dBFS
MIN_SILENCE_LEN = 200  # milliseconds


# --- STEP 1: STANDARDIZE AUDIO FORMAT ---
def standardize_audio(input_path, output_path):
    ffmpeg.input(input_path).output(output_path, format='wav').run(overwrite_output=True)
    print(f"[✓] Standardized audio saved to {output_path}")


# --- STEP 2: DENOISE WITH DEEPFILTERNET ---
def denoise_audio(input_path, output_path):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    try:
        subprocess.run([
            "deepFilter",
            os.path.abspath(input_path),
            "--output-dir", str(output_dir)
        ], check=True)
        expected_output = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(input_path))[0] + "_DeepFilterNet3.wav"
        )
        os.rename(expected_output, output_path)
        print(f"[✓] Denoised audio saved to {output_path}")
    except Exception as e:
        print(f"[!] DeepFilterNet failed: {e}")


# --- STEP 3: DIARIZATION ---
def diarize_audio(audio_path, output_rttm):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    diarization = pipeline(audio_path)
    with open(output_rttm, "w") as f:
        diarization.write_rttm(f)
    print(f"[✓] Diarization RTTM saved to {output_rttm}")
    return diarization


# --- STEP 4: EXTRACT SPEAKER AUDIO SEGMENTS ---
def extract_speaker_segments(audio_path, diarization, segments_dir):
    os.makedirs(segments_dir, exist_ok=True)
    audio = AudioSegment.from_wav(audio_path)
    speaker_segments = {}

    for i, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        clip = audio[start_ms:end_ms]

        if clip.duration_seconds > MAX_SEGMENT_DURATION:
            chunks = silence.split_on_silence(
                clip,
                min_silence_len=MIN_SILENCE_LEN,
                silence_thresh=SILENCE_THRESH,
                keep_silence=250
            )
            for j, chunk in enumerate(chunks):
                if chunk.duration_seconds > 0:
                    filename = os.path.join(segments_dir, f"{speaker}/_seg_{i}_{j}.wav")
                    os.makedirs(os.path.join(segments_dir, f"{speaker}"), exist_ok=True)
                    chunk.export(filename, format="wav")
                    speaker_segments.setdefault(speaker, []).append(filename)
        else:
            filename = os.path.join(segments_dir, f"{speaker}/_seg_{i}.wav")
            os.makedirs(os.path.join(segments_dir, f"{speaker}"), exist_ok=True)
            clip.export(filename, format="wav")
            speaker_segments.setdefault(speaker, []).append(filename)

    print(f"[✓] Extracted segments saved in '{segments_dir}'")
    return speaker_segments


# --- STEP 5: TRANSCRIBE USING WHISPER ---
# def transcribe_segments(speaker_segments, transcript_file):
#     # Initialize Whisper pipeline (using the "whisper-small" model in this example)
#     whisper_pipe = hf_pipeline("automatic-speech-recognition", model="openai/whisper-small")
#     full_transcript = ""
#     with open(transcript_file, "w") as out:
#         for speaker, files in speaker_segments.items():
#             for file in files:
#                 result = whisper_pipe(file)
#                 line = f"[{speaker}] {os.path.basename(file)}: {result['text']}\n"
#                 out.write(line)
#                 full_transcript += line
#                 print(f"[✓] Transcribed {file}")
#     print(f"[✓] Full transcript saved to {transcript_file}")
#     return full_transcript
#
# import csv
#
# MASTER_TRANSCRIPT_CSV = "master_transcript.csv"
#
# def transcribe_segments(speaker_segments, base_dir):
#     whisper_pipe = hf_pipeline("automatic-speech-recognition", model="openai/whisper-small")
#     local_transcript_path = os.path.join(base_dir, "speaker_transcript.txt")
#
#     with open(local_transcript_path, "w") as txt_out, open(MASTER_TRANSCRIPT_CSV, "a", newline='') as csv_out:
#         csv_writer = csv.writer(csv_out)
#         for speaker, files in speaker_segments.items():
#             for file in files:
#                 result = whisper_pipe(file)
#                 transcript_text = result['text'].strip()
#
#                 # Write to local .txt
#                 txt_out.write(f"[{speaker}] {os.path.basename(file)}: {transcript_text}\n")
#
#                 # Write to global CSV
#                 csv_writer.writerow([os.path.abspath(file), transcript_text])
#                 print(f"[✓] Transcribed {file}")
#     print(f"[✓] Full transcript saved to {local_transcript_path}")
#     print(f"[✓] Appended transcripts to {MASTER_TRANSCRIPT_CSV}")


def transcribe_segments(speaker_segments, base_dir):
    whisper_pipe = hf_pipeline("automatic-speech-recognition", model="openai/whisper-small")
    local_transcript_path = os.path.join(base_dir, "speaker_transcript.txt")
    segment_results = []  # List to hold tuples of (segment_path, transcript)

    with open(local_transcript_path, "w") as txt_out:
        for speaker, files in speaker_segments.items():
            for file in files:

                result = whisper_pipe(file)
                transcript_text = result['text'].strip()
                # Write to per-file transcript text
                txt_out.write(f"[{speaker}] {os.path.basename(file)}: {transcript_text}\n")
                # Append tuple (using absolute path) to our list
                segment_results.append((os.path.abspath(file), transcript_text))
                print(f"[✓] Transcribed {file}")

    print(f"[✓] Full transcript saved to {local_transcript_path}")
    return segment_results


# --- PROCESS A SINGLE AUDIO FILE ---
# def process_audio_file(input_file, output_root):
#     # Create a unique directory for the processed file
#     base_filename = os.path.splitext(os.path.basename(input_file))[0]
#     unique_id = uuid.uuid4().hex
#     unique_dir = os.path.join(output_root, f"{base_filename}_{unique_id}")
#     os.makedirs(unique_dir, exist_ok=True)
#
#     # Define file paths inside the unique directory
#     standardized_path = os.path.join(unique_dir, STANDARDIZED_AUDIO)
#     cleaned_path = os.path.join(unique_dir, CLEANED_AUDIO)
#     rttm_path = os.path.join(unique_dir, DIARIZATION_RTTM)
#     segments_dir = os.path.join(unique_dir, SPEAKER_SEGMENTS_DIR)
#     transcript_path = os.path.join(unique_dir, TRANSCRIPT_FILE)
#
#     print(f"\n[INFO] Processing file: {input_file}")
#     # Step 1: Standardize the audio file
#     standardize_audio(input_file, standardized_path)
#
#     # Step 2: Denoise the audio file
#     denoise_audio(standardized_path, cleaned_path)
#
#     # Step 3: Diarize the cleaned audio and save RTTM
#     diarization = diarize_audio(cleaned_path, rttm_path)
#
#     # Step 4: Extract speaker segments from the cleaned audio
#     speaker_segments = extract_speaker_segments(cleaned_path, diarization, segments_dir)
#
#     # Step 5: Transcribe segments using Whisper and return the full transcript text
#     full_transcript = transcribe_segments(speaker_segments, transcript_path)
#
#     return unique_dir, full_transcript


def process_audio_file(input_file, output_root):
    # Create a unique directory for the processed file
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    unique_id = uuid.uuid4().hex
    unique_dir = os.path.join(output_root, f"{base_filename}_{unique_id}")
    os.makedirs(unique_dir, exist_ok=True)

    # Define file paths inside the unique directory
    standardized_path = os.path.join(unique_dir, STANDARDIZED_AUDIO)
    cleaned_path = os.path.join(unique_dir, CLEANED_AUDIO)
    rttm_path = os.path.join(unique_dir, DIARIZATION_RTTM)
    segments_dir = os.path.join(unique_dir, SPEAKER_SEGMENTS_DIR)

    print(f"\n[INFO] Processing file: {input_file}")
    # Step 1: Standardize the audio file
    standardize_audio(input_file, standardized_path)

    # Step 2: Denoise the audio file
    denoise_audio(standardized_path, cleaned_path)

    # Step 3: Diarize the cleaned audio and save RTTM
    diarization = diarize_audio(cleaned_path, rttm_path)

    # Step 4: Extract speaker segments from the cleaned audio
    speaker_segments = extract_speaker_segments(cleaned_path, diarization, segments_dir)

    # Step 5: Transcribe segments using Whisper; returns list of tuples (segment_path, transcript)
    segment_results = transcribe_segments(speaker_segments, unique_dir)

    return unique_dir, segment_results

# --- MAIN PROCESSOR ---
# def process_all_audio(input_root, output_root, master_csv):
#     # Create output root if it doesn't exist
#     os.makedirs(output_root, exist_ok=True)
#
#     # Prepare (or create) master CSV: if it doesn't exist, write header.
#     if not os.path.exists(master_csv):
#         with open(master_csv, "w", newline="") as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(["Processed_Folder", "Transcript"])
#
#     # Walk through the input directory tree
#     supported_extensions = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
#     for root, dirs, files in os.walk(input_root):
#         for file in files:
#             if file.lower().endswith(supported_extensions):
#                 input_file = os.path.join(root, file)
#                 try:
#                     processed_folder, transcript = process_audio_file(input_file, output_root)
#                     # Append result to the master CSV; use absolute path for processed_folder.
#                     with open(master_csv, "a", newline="") as csvfile:
#                         writer = csv.writer(csvfile)
#                         writer.writerow([os.path.abspath(processed_folder), transcript.strip()])
#                 except Exception as e:
#                     print(f"[!] Error processing {input_file}: {e}")

def process_all_audio(input_root, output_root, master_csv):
    # Create output root if it doesn't exist
    os.makedirs(output_root, exist_ok=True)

    # Prepare (or create) master CSV: if it doesn't exist, write header.
    if not os.path.exists(master_csv):
        with open(master_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Processed_Segment_Path", "Transcript"])

    # Walk through the input directory tree
    supported_extensions = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
    for root_dir, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(supported_extensions):
                input_file = os.path.join(root_dir, file)
                try:
                    processed_folder, segment_results = process_audio_file(input_file, output_root)
                    # Append each segment's result to the master CSV; use absolute path for each segment
                    with open(master_csv, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        for seg_path, transcript in segment_results:
                            writer.writerow([seg_path, transcript])
                except Exception as e:
                    print(f"[!] Error processing {input_file}: {e}")


if __name__ == "__main__":
    process_all_audio(INPUT_DIR, OUTPUT_ROOT, MASTER_TRANSCRIPT_CSV)
