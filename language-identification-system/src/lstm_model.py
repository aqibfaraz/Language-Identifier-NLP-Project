import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# ================== CONFIG ==================

MODEL_PATH = r"C:\Users\sk\OneDrive\Desktop\NLP Project\language-identification-system\models\lstm_model.h5"
DATASET_PATH = r"C:\Users\sk\OneDrive\Desktop\NLP Project\dataset.csv"
TOKENIZER_PATH = r"C:\Users\sk\OneDrive\Desktop\NLP Project\language-identification-system\models\tokenizer.pkl"

# ============================================

LANG_CODE_TO_NAME = {
    'en': 'English', 'fr': 'French', 'de': 'German', 'es': 'Spanish', 'it': 'Italian',
    'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese',
    'ar': 'Arabic', 'hi': 'Hindi', 'ur': 'Urdu', 'bn': 'Bengali', 'pa': 'Punjabi',
    'nl': 'Dutch', 'sv': 'Swedish', 'et': 'Estonian', 'tr': 'Turkish', 'fa': 'Persian',
    'ro': 'Romanian', 'ko': 'Korean', 'th': 'Thai', 'ta': 'Tamil', 'la': 'Latin',
    'id': 'Indonesian', 'ps': 'Pushto'
}


def get_language_name(code: str) -> str:
    """Convert language code to full name"""
    if not code:
        return code
    return LANG_CODE_TO_NAME.get(code, code)


class LSTMLanguageIdentifier:
    """LSTM-based Language Identification Model"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.languages = None

    def build_model(self, input_shape, num_classes):
        """Build LSTM model architecture"""
        from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
        from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout # pyright: ignore[reportMissingImports]

        self.model = Sequential()
        self.model.add(Embedding(input_dim=5000, output_dim=128, input_length=input_shape))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(64))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def load_model(self):
        """Load trained LSTM model"""
        from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
        from tensorflow.keras.preprocessing.text import Tokenizer # pyright: ignore[reportMissingImports]
        import pickle
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = load_model(self.model_path)
        
        # Load dataset to recreate tokenizer and label encoder
        df = pd.read_csv(DATASET_PATH)
        
        # Recreate tokenizer
        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(df['Text'])
        
        # Recreate label encoder
        self.label_encoder = LabelEncoder()
        self.languages = df['language'].unique()
        self.label_encoder.fit(self.languages)
        
        print(f"✅ Model loaded: {self.model_path}\n")

    def detect_language(self, text):
        """Detect language from input text"""
        from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]
        
        if not self.model:
            raise Exception("Model not loaded. Call load_model() first.")
        
        if not self.tokenizer:
            raise Exception("Tokenizer not initialized.")
        
        # Clean text
        text = " ".join(text.split())
        
        # Convert to sequences and pad
        sequences = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=100, padding='post')
        
        # Predict
        predictions = self.model.predict(padded, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index]
        
        predicted_language = self.label_encoder.inverse_transform([predicted_index])[0]
        
        return {
            "language": predicted_language,
            "confidence": float(confidence),
            "all_probabilities": {
                self.label_encoder.inverse_transform([i])[0]: float(pred)
                for i, pred in enumerate(predictions[0])
            }
        }

    def prepare_model(self):
        """Load model if it exists"""
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please train the model first.")


# ================== MAIN ==================

if __name__ == "__main__":

    detector = LSTMLanguageIdentifier(MODEL_PATH)
    
    try:
        # Load model
        detector.prepare_model()
        
        print("=" * 60)
        print("LSTM LANGUAGE DETECTION")
        print("=" * 60 + "\n")
        print("Enter text to detect language (type 'exit' to quit)\n")

        while True:
            text = input("Text: ")
            if text.lower().strip() == "exit":
                print("Goodbye 👋")
                break

            if not text.strip():
                print("Please enter some text.\n")
                continue

            try:
                result = detector.detect_language(text)
                print(
                    f"Detected Language: {result['language']} "
                    f"(Confidence: {result['confidence']:.4f})\n"
                )
            except Exception as e:
                print(f"Error: {e}\n")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure the model is trained and saved at {MODEL_PATH}")
