def create_n_grams(text, n):
    """Create character n-grams from the input text."""
    return [text[i:i+n] for i in range(len(text)-n+1)]

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy of predictions."""
    return sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred)) / len(y_true)

def calculate_precision(y_true, y_pred):
    """Calculate precision of predictions."""
    true_positive = sum((y_t == y_p == 1) for y_t, y_p in zip(y_true, y_pred))
    predicted_positive = sum(y_pred)
    return true_positive / predicted_positive if predicted_positive > 0 else 0

def calculate_recall(y_true, y_pred):
    """Calculate recall of predictions."""
    true_positive = sum((y_t == y_p == 1) for y_t, y_p in zip(y_true, y_pred))
    actual_positive = sum(y_true)
    return true_positive / actual_positive if actual_positive > 0 else 0

def calculate_f1_score(y_true, y_pred):
    """Calculate F1 score of predictions."""
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0