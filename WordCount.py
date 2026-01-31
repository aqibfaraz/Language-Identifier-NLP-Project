import pandas as pd
from collections import Counter

# Load the dataset
df = pd.read_csv('dataset.csv')

# Count unique languages
unique_languages = df['language'].nunique()
print(f"\nNumber of unique languages: {unique_languages}")


# Count total number of words across all texts
total_words = 0
for text in df['Text']:
    if isinstance(text, str):
        words = text.split()
        total_words += len(words)

print(f"\nTotal number of words in dataset: {total_words}")

# Count words per language
print(f"\nTotal words per language:")
language_word_counts = {}
for idx, row in df.iterrows():
    lang = row['language']
    text = row['Text']
    if isinstance(text, str):
        word_count = len(text.split())
        if lang in language_word_counts:
            language_word_counts[lang] += word_count
        else:
            language_word_counts[lang] = word_count

# Sort by word count
sorted_language_words = sorted(language_word_counts.items(), key=lambda x: x[1], reverse=True)
for lang, word_count in sorted_language_words:
    print(f"{lang}: {word_count} words")
