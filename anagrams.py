import csv
import nltk
from nltk.corpus import gutenberg
from nltk.util import ngrams
from collections import Counter
import string

nltk.download('gutenberg', quiet=True)
nltk.download('punkt', quiet=True)

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return nltk.word_tokenize(text)

def get_common_elements(text):
    words = clean_text(text)
    word_freq = Counter(words).most_common(5)
    bigrams = Counter(ngrams(words, 2)).most_common(5)
    trigrams = Counter(ngrams(words, 3)).most_common(5)
    return word_freq, bigrams, trigrams

def find_palindromes(text):
    words = clean_text(text)
    palindromes = []
    for word in words:
        if len(word) > 1 and word == word[::-1] and word not in palindromes:
            palindromes.append(word)
    return palindromes

def find_anagrams(text):
    words = clean_text(text)
    anagram_pairs = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            w1 = words[i]
            w2 = words[j]
            if w1 != w2 and len(w1) == len(w2) and sorted(w1) == sorted(w2):
                pair = (w1, w2)
                if pair not in anagram_pairs and (w2, w1) not in anagram_pairs:
                    anagram_pairs.append(pair)
    return anagram_pairs

def find_rhymes(text):
    words = clean_text(text)
    rhymes = {}
    for word in words:
        if len(word) > 2:
            ending = word[-3:]
            if ending not in rhymes:
                rhymes[ending] = [word]
            else:
                if word not in rhymes[ending]:
                    rhymes[ending].append(word)
    final_rhymes = {}
    for ending in rhymes:
        if len(rhymes[ending]) > 1:
            final_rhymes[ending] = rhymes[ending]
    return final_rhymes

csv_path = r'C:\Users\sneha\Desktop\LAB(MV)\text.csv'

with open(csv_path, 'r') as file:
    reader = csv.reader(file)
    csv_text = ' '.join(row[0] for row in reader if len(row) > 0 and row[0].strip() != '')


print("\n========== Analysis for CSV File ==========")
words, bigrams, trigrams = get_common_elements(csv_text)
palindromes = find_palindromes(csv_text)
anagrams = find_anagrams(csv_text)
rhymes = find_rhymes(csv_text)

print("\n--- Most Common Words ---")
print(words)

print("\n--- Most Common Bigrams ---")
print(bigrams)

print("\n--- Most Common Trigrams ---")
print(trigrams)

print("\n--- Palindrome Words ---")
print(palindromes)

print("\n--- Anagram Pairs ---")
print(anagrams)

print("\n--- Rhyming Words ---")
print(rhymes)

austen_text = gutenberg.raw('austen-emma.txt')[:5000]
shakespeare_text = gutenberg.raw('shakespeare-hamlet.txt')[:5000]

print("\n========== Analysis for Austen's Emma ==========")
words, bigrams, trigrams = get_common_elements(austen_text)
palindromes = find_palindromes(austen_text)
anagrams = find_anagrams(austen_text)
rhymes = find_rhymes(austen_text)

print("\n--- Most Common Words ---")
print(words)

print("\n--- Most Common Bigrams ---")
print(bigrams)

print("\n--- Most Common Trigrams ---")
print(trigrams)

print("\n--- Palindrome Words ---")
print(palindromes)

print("\n--- Anagram Pairs ---")
print(anagrams)

print("\n--- Rhyming Words ---")
print(rhymes)



print("\n========== Analysis for Shakespeare's Hamlet ==========")
words, bigrams, trigrams = get_common_elements(shakespeare_text)
palindromes = find_palindromes(shakespeare_text)
anagrams = find_anagrams(shakespeare_text)
rhymes = find_rhymes(shakespeare_text)

print("\n--- Most Common Words ---")
print(words)

print("\n--- Most Common Bigrams ---")
print(bigrams)

print("\n--- Most Common Trigrams ---")
print(trigrams)

print("\n--- Palindrome Words ---")
print(palindromes)

print("\n--- Anagram Pairs ---")
print(anagrams)

print("\n--- Rhyming Words ---")
print(rhymes)

