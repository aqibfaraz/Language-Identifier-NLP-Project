import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# ================== CONFIG ==================
PROJECT_ROOT = r"C:\Users\sk\OneDrive\Desktop\NLP Project"
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset.csv")
LSTM_MODEL_PATH = os.path.join(PROJECT_ROOT, "language-identification-system", "models", "lstm_model.h5")

# ============================================

def load_and_prepare_data(test_size=0.2):
    """Load dataset and split into train and test sets"""
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['Text'], df['language'], test_size=test_size, random_state=42, stratify=df['language']
    )
    
    print(f"Dataset loaded with {len(df)} samples")
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    print(f"Languages: {df['language'].unique().tolist()}\n")
    
    return X_train, X_test, y_train, y_test, df['language'].unique()


def train_lstm_model():
    """Train and save LSTM model"""
    print("=" * 60)
    print("LSTM MODEL TRAINING")
    print("=" * 60 + "\n")
    
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import EarlyStopping
        
        # Load and prepare data
        X_train, X_test, y_train, y_test, languages = load_and_prepare_data()
        
        # Create tokenizer
        print("Creating tokenizer...")
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_train)
        
        # Convert text to sequences
        print("Converting text to sequences...")
        X_train_sequences = tokenizer.texts_to_sequences(X_train)
        X_train_padded = pad_sequences(X_train_sequences, maxlen=100, padding='post')
        
        X_test_sequences = tokenizer.texts_to_sequences(X_test)
        X_test_padded = pad_sequences(X_test_sequences, maxlen=100, padding='post')
        
        # Encode labels
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        label_encoder.fit(languages)
        y_train_encoded = label_encoder.transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        y_train_categorical = to_categorical(y_train_encoded, num_classes=len(languages))
        y_test_categorical = to_categorical(y_test_encoded, num_classes=len(languages))
        
        # Build model
        print("Building LSTM model...")
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(64))
        model.add(Dropout(0.5))
        model.add(Dense(len(languages), activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        print("\nModel Summary:")
        model.summary()
        
        # Train model
        print("\nTraining model...")
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        history = model.fit(
            X_train_padded, y_train_categorical,
            batch_size=32,
            epochs=15,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        loss, accuracy = model.evaluate(X_test_padded, y_test_categorical, verbose=0)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test Loss: {loss:.4f}")
        
        # Save model
        print(f"\nSaving model to {LSTM_MODEL_PATH}...")
        model.save(LSTM_MODEL_PATH)
        print("✅ LSTM model trained and saved successfully!\n")
        
        return True
        
    except ImportError as e:
        print(f"ERROR: TensorFlow/Keras not installed. Please install it first.")
        print(f"Details: {e}\n")
        return False
    except Exception as e:
        print(f"Error training LSTM model: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = train_lstm_model()
    sys.exit(0 if success else 1)
