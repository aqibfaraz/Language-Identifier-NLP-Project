import fasttext
import os

try:
    import pycountry
except Exception:
    pycountry = None


# ================== CONFIG ==================

MODEL_PATH = r"C:\Users\sk\OneDrive\Desktop\NLP Project\language-identification-system\src\fasttext_language_id.bin"
TRAIN_DATA_PATH = r"C:\Users\sk\OneDrive\Desktop\NLP Project\language-identification-system\data\train.csv"

# ============================================


LANG_CODE_TO_NAME = {
    'en': 'English', 'fr': 'French', 'de': 'German', 'es': 'Spanish', 'it': 'Italian',
    'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese',
    'ar': 'Arabic', 'hi': 'Hindi', 'ur': 'Urdu', 'bn': 'Bengali', 'pa': 'Punjabi'
}


def get_language_name(code: str) -> str:
    if not code:
        return code

    base = code.split('-')[0].lower()

    if base in LANG_CODE_TO_NAME:
        return LANG_CODE_TO_NAME[base]

    if pycountry:
        try:
            lang = pycountry.languages.get(alpha_2=base)
            if lang and getattr(lang, "name", None):
                return lang.name
        except Exception:
            pass

    return base


# ================== MODEL CLASS ==================

class FastTextLanguageIdentifier:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def train_and_save(self, data_path):
        print("⚠️ Model not found. Training started...\n")

        self.model = fasttext.train_supervised(
            input=data_path,
            lr=0.5,
            dim=200,
            epoch=50,
            wordNgrams=3,
            minn=3,
            maxn=6,
            loss="softmax"
        )

        self.model.save_model(self.model_path)
        print(f"✅ Model trained & saved: {self.model_path}\n")

    def load_model(self):
        self.model = fasttext.load_model(self.model_path)
        print(f"✅ Model loaded: {self.model_path}\n")

    def prepare_model(self, data_path):
        """
        CORE LOGIC:
        - Model exists → LOAD
        - Else → TRAIN + SAVE
        """
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self.train_and_save(data_path)

    def detect_language(self, text):
        if not self.model:
            raise Exception("Model not loaded")

        text = " ".join(text.split())

        labels, confidences = self.model.predict(text, k=1)
        lang_code = labels[0].replace("__label__", "")

        return {
            "language_code": lang_code,
            "language": get_language_name(lang_code),
            "confidence": confidences[0]
        }


# ================== MAIN ==================

if __name__ == "__main__":

    detector = FastTextLanguageIdentifier(MODEL_PATH)

    # 🔥 ONE LINE DECISION (TRAIN or LOAD)
    detector.prepare_model(TRAIN_DATA_PATH)

    print("Enter text to detect language (type 'exit' to quit)\n")

    while True:
        text = input("Text: ")
        if text.lower().strip() == "exit":
            print("Goodbye 👋")
            break

        try:
            result = detector.detect_language(text)
            print(
                f"Detected Language: {result['language']} "
                f"(Code: {result['language_code']}, "
                f"Confidence: {result['confidence']:.4f})\n"
            )
        except Exception as e:
            print(f"Error: {e}\n")
