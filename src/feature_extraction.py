import librosa
import numpy as np

def extract_features(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, duration=30)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    
    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma = np.mean(chroma.T, axis=0)
    
    # Extract Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = np.mean(contrast.T, axis=0)
    
    # Combine features into a single feature vector
    features = np.hstack([mfccs, chroma, contrast])
    
    return features
