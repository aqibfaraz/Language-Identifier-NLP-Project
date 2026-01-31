import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import fasttext
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# ================== CONFIG ==================
PROJECT_ROOT = r"C:\Users\sk\OneDrive\Desktop\NLP Project"
FASTTEXT_MODEL_PATH = os.path.join(PROJECT_ROOT, "language-identification-system", "models", "fasttext_language_id.bin")
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset.csv")
LSTM_MODEL_PATH = os.path.join(PROJECT_ROOT, "language-identification-system", "models", "lstm_model.h5")

# Language mappings
LANG_NAME_TO_CODE = {
    'English': 'en', 'French': 'fr', 'German': 'de', 'Spanish': 'es', 'Italian': 'it',
    'Portuguese': 'pt', 'Portugese': 'pt', 'Russian': 'ru', 'Chinese': 'zh', 'Japanese': 'ja',
    'Arabic': 'ar', 'Hindi': 'hi', 'Urdu': 'ur', 'Bengali': 'bn', 'Punjabi': 'pa',
    'Dutch': 'nl', 'Swedish': 'sv', 'Estonian': 'et', 'Turkish': 'tr', 'Persian': 'fa',
    'Romanian': 'ro', 'Korean': 'ko', 'Thai': 'th', 'Tamil': 'ta', 'Latin': 'la',
    'Indonesian': 'id', 'Pushto': 'ps'
}

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


def evaluate_fasttext_model(X_test, y_test):
    """Evaluate FastText model accuracy"""
    print("=" * 60)
    print("FASTTEXT MODEL EVALUATION")
    print("=" * 60)
    
    if not os.path.exists(FASTTEXT_MODEL_PATH):
        print(f"WARNING: FastText model not found at {FASTTEXT_MODEL_PATH}")
        print("Please train the FastText model first.\n")
        return None
    
    try:
        # Load the fasttext model
        model = fasttext.load_model(FASTTEXT_MODEL_PATH)
        
        # Convert test labels from language names to codes
        y_test_codes = [LANG_NAME_TO_CODE.get(lang, lang) for lang in y_test]
        
        # Predict on test set
        predictions = []
        for text in X_test:
            # fasttext predict returns (labels, probabilities)
            prediction = model.predict(text.replace('\n', ' '))[0][0]
            # Remove '__label__' prefix if present
            lang_code = prediction.replace('__label__', '')
            predictions.append(lang_code)
        
        # Convert to array for comparison
        predictions = np.array(predictions)
        y_test_codes = np.array(y_test_codes)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_codes, predictions)
        precision = precision_score(y_test_codes, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test_codes, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test_codes, predictions, average='weighted', zero_division=0)
        
        # Calculate correct predictions
        correct = np.sum(y_test_codes == predictions)
        total = len(y_test_codes)
        percentage = accuracy * 100
        
        print(f"Accuracy:  {correct}/{total} ({percentage:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}\n")
        
        print("Classification Report:")
        print(classification_report(y_test_codes, predictions, zero_division=0))
        print()
        
        return accuracy, precision, recall, f1
        
    except Exception as e:
        print(f"Error evaluating FastText model: {e}\n")
        return None


def evaluate_lstm_model(X_test, y_test, languages):
    """Evaluate LSTM model accuracy"""
    print("=" * 60)
    print("LSTM MODEL EVALUATION")
    print("=" * 60)
    print("Loading and evaluating LSTM model (this may take a moment)...\n")
    
    try:
        from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
        from tensorflow.keras.preprocessing.text import Tokenizer # pyright: ignore[reportMissingImports]
        from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]
        from tensorflow.keras.utils import to_categorical # pyright: ignore[reportMissingImports]
        
        # LSTM model path
        lstm_model_path = LSTM_MODEL_PATH
        
        if not os.path.exists(lstm_model_path):
            print(f"WARNING: LSTM model not found at {lstm_model_path}")
            print("Please train the LSTM model first.\n")
            return None
        
        # Load the model
        model = load_model(lstm_model_path)
        
        # Prepare test data (same preprocessing as training)
        # Create tokenizer
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_test)
        
        # Convert text to sequences
        X_test_sequences = tokenizer.texts_to_sequences(X_test)
        X_test_padded = pad_sequences(X_test_sequences, maxlen=100, padding='post')
        
        # Encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(languages)
        y_test_encoded = label_encoder.transform(y_test)
        y_test_categorical = to_categorical(y_test_encoded, num_classes=len(languages))
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test_padded, y_test_categorical, verbose=0)
        
        # Get predictions
        predictions_probs = model.predict(X_test_padded, verbose=0)
        predictions_indices = np.argmax(predictions_probs, axis=1)
        predictions = label_encoder.inverse_transform(predictions_indices)
        
        # Calculate metrics
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        # Calculate correct predictions
        correct = np.sum(y_test.values == predictions) if hasattr(y_test, 'values') else np.sum(y_test == predictions)
        total = len(y_test)
        percentage = accuracy * 100
        
        print(f"Accuracy:  {correct}/{total} ({percentage:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}\n")
        
        print("Classification Report:")
        print(classification_report(y_test, predictions, zero_division=0))
        print()
        
        return accuracy, precision, recall, f1
        
    except ImportError as e:
        print(f"ERROR: TensorFlow/Keras not installed. Please install it first.")
        print(f"Details: {e}\n")
        return None
    except Exception as e:
        print(f"Error evaluating LSTM model: {e}\n")
        import traceback
        traceback.print_exc()
        return None
        print(f"F1-Score:  {f1:.4f}\n")
        
        print("Classification Report:")
        print(classification_report(y_test, predictions, zero_division=0))
        print()
        
        return accuracy, precision, recall, f1
        
    except ImportError as e:
        print(f"ERROR: TensorFlow/Keras not installed. Please install it first.")
        print(f"Details: {e}\n")
        return None
    except Exception as e:
        print(f"Error evaluating LSTM model: {e}\n")
        return None


def main():
    """Main evaluation function"""
    print("\n" + "=" * 60)
    print("MODEL ACCURACY EVALUATION")
    print("=" * 60 + "\n")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, languages = load_and_prepare_data()
    
    # Evaluate FastText model
    fasttext_results = evaluate_fasttext_model(X_test, y_test)
    
    # Evaluate LSTM model
    lstm_results = evaluate_lstm_model(X_test, y_test, languages)
    
    # Comparison
    print("=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    
    if fasttext_results and lstm_results:
        fasttext_acc, fasttext_prec, fasttext_rec, fasttext_f1 = fasttext_results
        lstm_acc, lstm_prec, lstm_rec, lstm_f1 = lstm_results
        
        print(f"\n{'Model':<15} {'Accuracy':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 66)
        print(f"{'FastText':<15} {fasttext_acc:.4f} ({int(fasttext_acc*len(y_test))}/{len(y_test)}) {fasttext_prec:<12.4f} {fasttext_rec:<12.4f} {fasttext_f1:<12.4f}")
        print(f"{'LSTM':<15} {lstm_acc:.4f} ({int(lstm_acc*len(y_test))}/{len(y_test)}) {lstm_prec:<12.4f} {lstm_rec:<12.4f} {lstm_f1:<12.4f}")
        print("-" * 66)
        
        if fasttext_acc > lstm_acc:
            diff = (fasttext_acc - lstm_acc) * 100
            print(f"\n✓ FastText performs better by {diff:.2f}%")
        elif lstm_acc > fasttext_acc:
            diff = (lstm_acc - fasttext_acc) * 100
            print(f"\n✓ LSTM performs better by {diff:.2f}%")
        else:
            print(f"\n✓ Both models have similar accuracy")
    elif fasttext_results:
        fasttext_acc, _, _, _ = fasttext_results
        print(f"\n✓ FastText model evaluated successfully.")
        print(f"  Accuracy: {int(fasttext_acc*len(y_test))}/{len(y_test)} ({fasttext_acc*100:.2f}%)")
        print("✗ LSTM model evaluation skipped or failed.")
    elif lstm_results:
        lstm_acc, _, _, _ = lstm_results
        print(f"\n✓ LSTM model evaluated successfully.")
        print(f"  Accuracy: {int(lstm_acc*len(y_test))}/{len(y_test)} ({lstm_acc*100:.2f}%)")
        print("✗ FastText model evaluation skipped or failed.")
    else:
        print("\n✗ Both models failed to evaluate. Please check the model paths and ensure models are trained.")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
