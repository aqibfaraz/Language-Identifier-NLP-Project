import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================== CONFIG ==================
PROJECT_ROOT = r"C:\Users\sk\OneDrive\Desktop\NLP Project"
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset.csv")
LSTM_MODEL_PATH = os.path.join(PROJECT_ROOT, "language-identification-system", "models", "lstm_model.h5")

# Test samples in different languages
TEST_SAMPLES = {
    'English': 'Hello, how are you today?',
    'Spanish': 'Hola, ¿cómo estás hoy?',
    'French': 'Bonjour, comment allez-vous?',
    'German': 'Hallo, wie geht es dir heute?',
    'Chinese': '你好，今天怎么样？',
    'Japanese': 'こんにちは、今日はどうですか？',
    'Arabic': 'مرحبا، كيف حالك اليوم؟',
    'Russian': 'Привет, как дела?',
    'Thai': 'สวัสดี วันนี้เป็นอย่างไร',
    'Hindi': 'नमस्ते, आप आज कैसे हैं?'
}

# ============================================

def load_model_and_tokenizer():
    """Load LSTM model"""
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    
    print("Loading LSTM model...")
    model = load_model(LSTM_MODEL_PATH)
    
    # Load dataset to fit tokenizer on training data
    df = pd.read_csv(DATASET_PATH)
    
    # Create and fit tokenizer on full dataset
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['Text'])
    
    # Get language labels
    label_encoder = LabelEncoder()
    label_encoder.fit(df['language'].unique())
    
    return model, tokenizer, label_encoder


def detect_language_lstm(text, model, tokenizer, label_encoder):
    """Detect language using LSTM model"""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100, padding='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)
    predicted_index = np.argmax(prediction[0])
    predicted_language = label_encoder.inverse_transform([predicted_index])[0]
    confidence = prediction[0][predicted_index] * 100
    
    return predicted_language, confidence


def main():
    print("\n" + "=" * 70)
    print("LSTM LANGUAGE DETECTION - TEST SAMPLES")
    print("=" * 70 + "\n")
    
    try:
        # Load model
        model, tokenizer, label_encoder = load_model_and_tokenizer()
        print("✓ Model loaded successfully!\n")
        print("=" * 70)
        
        # Test with samples
        correct = 0
        for expected_lang, text in TEST_SAMPLES.items():
            detected_lang, confidence = detect_language_lstm(text, model, tokenizer, label_encoder)
            
            is_correct = "✓" if detected_lang == expected_lang else "✗"
            if detected_lang == expected_lang:
                correct += 1
            
            print(f"\n{is_correct} Expected: {expected_lang:12} | Detected: {detected_lang:12} | Confidence: {confidence:6.2f}%")
            print(f"  Text: {text}")
        
        print("\n" + "=" * 70)
        print(f"Accuracy on test samples: {correct}/{len(TEST_SAMPLES)} ({100*correct/len(TEST_SAMPLES):.1f}%)")
        print("=" * 70 + "\n")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
