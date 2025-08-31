# EchoMorph.ai → EN Voice: Transcribe, Diarize, Translate, TTS, and Mux Pipeline

## Overview

This project transforms Japanese-language videos into English-voiced outputs through an automated pipeline. The system demuxes source media, transcribes speech using Whisper, performs speaker diarization, translates content to English, generates per-speaker English text-to-speech via Murf API, and produces a final video with synchronized English audio.

## 🎯 Key Features

- *Media Processing*: Intelligent demuxing and audio format conversion using FFmpeg
- *Advanced ASR*: Whisper-powered transcription with GPU acceleration and language detection
- *Speaker Diarization*: AI-powered speaker identification using Resemblyzer embeddings and spectral clustering
- *Bilingual Translation*: Automated Japanese-to-English translation with deep-translator
- *Natural TTS*: Per-speaker voice synthesis using Murf API for realistic conversations
- *Video Reconstruction*: Seamless audio-video synchronization in final output

## 🛠 Technical Stack

### Environment Requirements
- *Platform*: Google Colab (recommended) or Linux with Python 3.9+
- *System Dependencies*: FFmpeg (must be available on PATH)
- *Hardware*: CUDA-compatible GPU recommended for faster processing

### Python Dependencies

torch, torchaudio, torchvision
openai-whisper (or faster-whisper for performance)
demucs, ffmpeg-python, soundfile, pydub, librosa
resemblyzer, spectralcluster, scikit-learn
pandas, deep-translator
murf, requests


### API Configuration
- *Murf API*: Set MURF_API_KEY in Google Colab Secrets or environment variables

## ⚙ Configuration Parameters

### File Paths
- AUDIO_IN: /content/output/Video_a1_und.wav (extracted audio)
- TMP_WAV: /content/_tmp_16k_mono.wav (standardized for processing)
- CSV_OUT: /content/transcript_with_speakers.csv (bilingual transcript)

### Processing Settings
- WHISPER_MODEL: "large-v3" (use "small"/"medium" for limited RAM)
- LANG: "ja" for forced Japanese detection (set None for auto-detect)
- MIN_SEG_DUR: 0.35s minimum segment duration for stable embeddings
- K_MIN, K_MAX: Speaker count range (2-6); set equal values to force specific count
- GAP_MS: 700ms silence padding between synthesized speech segments

### Voice Mapping
Configure speaker-to-voice assignments in voice_map:
python
voice_map = {
    "Speaker 1": {"voice_id": "murf_voice_id", "style": "conversational"},
    "Speaker 2": {"voice_id": "another_voice_id", "style": "formal"}
}


## 🔄 Processing Workflow

### 1. Media Preparation
- Demux MKV into separate video and audio streams
- Convert audio to standardized WAV format (mono, 16kHz)
- Preserve video stream for final reconstruction

### 2. Speech Recognition & Analysis
- *Transcription*: Whisper processes audio with timestamp precision
- *Standardization*: Audio normalization for consistent processing
- *Segmentation*: Intelligent splitting for optimal diarization

### 3. Speaker Diarization
- *Embedding Generation*: Resemblyzer creates speaker embeddings
- *Clustering*: Spectral clustering with automated K-selection via silhouette analysis
- *Segment Merging*: Combines short segments for embedding stability
- *Label Assignment*: Maps clusters back to timestamped segments

### 4. Translation & Synthesis
- *Bilingual Export*: CSV generation with Japanese text and English translations
- *Voice Synthesis*: Per-speaker TTS using Murf API with custom voice mapping
- *Audio Stitching*: Combines individual speech segments with appropriate gaps

### 5. Final Assembly
- *Audio Replacement*: Merges synthesized English audio with original video
- *Format Preservation*: Maintains video quality and metadata
- *Output Delivery*: Produces final MKV with English voiceover

## 🚀 Quick Start (Google Colab)

1. *Setup Environment*:
   python
   # Upload Video.mkv and demux.py to /content/
   # Set MURF_API_KEY in Colab Secrets
   

2. *Execute Pipeline*:
   - Run cells sequentially in the provided notebook
   - Monitor progress through each processing stage
   - Check intermediate outputs for quality validation

3. *Key Outputs*:
   - transcript_with_speakers.csv: Bilingual transcript with speaker labels
   - generated_tts/run_TIMESTAMP/: Individual TTS files and stitched audio
   - final_with_audio.mkv: Complete processed video

## 🔧 Important Technical Notes

### Recent Fixes & Improvements
- *Whisper Configuration*: Proper dictionary construction for transcription parameters
- *Clustering*: Fixed SpectralClusterer initialization with correct parameters
- *Voice IDs*: Updated placeholder Murf voice IDs with valid API endpoints
- *Audio Sync*: Implemented -shortest flag for stream length matching

### Performance Optimization
- Use faster-whisper for improved transcription speed
- GPU auto-detection for optimal hardware utilization
- Configurable model sizes for memory-constrained environments
- Batch processing for TTS generation efficiency

### Quality Considerations
- Minimum segment duration prevents clustering instability
- Silhouette scoring ensures optimal speaker count detection
- Gap timing maintains natural conversation flow
- Audio standardization improves processing consistency

## 🎯 Use Cases

- *Content Localization*: Transform Japanese educational content for English audiences
- *Media Translation*: Convert interviews, documentaries, and presentations
- *Accessibility*: Create English versions of Japanese media content
- *Research Applications*: Analyze multilingual conversation patterns

## 📊 Expected Output Quality

The pipeline produces broadcast-quality results suitable for:
- Educational content distribution
- Professional media localization
- Research and analysis workflows
- Accessibility compliance requirements

## 🔍 Troubleshooting

- *Memory Issues*: Reduce Whisper model size or process shorter segments
- *Audio Sync*: Adjust gap timing or use manual alignment for critical applications
- *Voice Quality*: Experiment with different Murf voice mappings and styles
- *Speaker Count*: Fine-tune K_MIN/K_MAX based on known participant count

---

This pipeline represents a complete solution for automated multilingual video processing, combining state-of-the-art AI technologies for practical media transformation workflows.
