import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from functools import lru_cache
import faiss
import joblib  # For parallel processing

nltk.download('stopwords')

# Load the set of stopwords and define negation words
stop_words = set(stopwords.words('english'))
negation_words = {"not", "n't", "no", "never"}

# Load the SBERT model (fine-tuned on engineering texts if available)
model = SentenceTransformer('allenai/scibert_scivocab_uncased')

def preprocess_text(text):
    # Convert text to lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    words = text.split()
    processed_words = []
    negate = False

    for word in words:
        # Keep negation words, remove other stopwords
        if word in stop_words and word not in negation_words:
            continue

        # Handle negation trigger words
        if word in negation_words:
            negate = True
            continue

        # If negation is true, append NOT_ to the word and reset negation
        if negate:
            processed_words.append("NOT_" + word)
            negate = False
        else:
            processed_words.append(word)

    return processed_words

def remove_redundant_words(words):
    # Remove redundant words
    return list(dict.fromkeys(words))

def remove_question_keywords(answer_words, question_words):
    # Remove question keywords from the answer
    filtered_words = [word for word in answer_words if word not in question_words]
    return filtered_words

def compute_word_frequency_vector(words, unique_words):
    # Count frequency of each word in the list of words
    word_freq = Counter(words)
    # Create a vector based on the unique words from both lists
    vector = [word_freq[word] for word in unique_words]
    return vector

def classify_question(question):
    # Enhanced classification logic, possibly using a trained classifier
    explanation_keywords = {"explain", "describe", "why", "how"}
    
    # Check if the question contains any keywords that indicate it's explanation-based
    if any(keyword in question.lower() for keyword in explanation_keywords):
        return "explanation"
    return "fact"

@lru_cache(maxsize=10000)
def get_embedding(text):
    return model.encode(text)

def evaluate_with_keywords(reference_answer, candidate_answer, question):
    # Preprocess the question
    question_processed = preprocess_text(question)

    # Preprocess the reference and candidate answers
    ref_processed = preprocess_text(reference_answer)
    cand_processed = preprocess_text(candidate_answer)

    # Remove question keywords from the reference and candidate answers
    ref_filtered = remove_question_keywords(ref_processed, question_processed)
    cand_filtered = remove_question_keywords(cand_processed, question_processed)

    # Remove redundant words
    ref_filtered = remove_redundant_words(ref_filtered)
    cand_filtered = remove_redundant_words(cand_filtered)

    print(f"Filtered Reference Keywords: {ref_filtered}")
    print(f"Filtered Candidate Keywords: {cand_filtered}")

    # Combine unique words from both filtered answers
    unique_words = list(set(ref_filtered + cand_filtered))

    # Compute word frequency vectors for both answers
    ref_vector = compute_word_frequency_vector(ref_filtered, unique_words)
    cand_vector = compute_word_frequency_vector(cand_filtered, unique_words)

    # Convert vectors to numpy arrays for cosine similarity calculation
    ref_vector = np.array(ref_vector).reshape(1, -1)
    cand_vector = np.array(cand_vector).reshape(1, -1)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(ref_vector, cand_vector)[0][0]

    # Convert similarity to a percentage
    match_percentage = cosine_sim * 100

    return match_percentage, ref_filtered, cand_filtered, question_processed

def evaluate_with_sbert(reference_answer, candidate_answer, question):
    # Preprocess the question if needed (not used in SBERT evaluation here)
    
    # Preprocess the reference and candidate answers
    ref_processed = preprocess_text(reference_answer)
    cand_processed = preprocess_text(candidate_answer)

    # Generate embeddings using SBERT (cached)
    ref_embedding = get_embedding(" ".join(ref_processed))
    cand_embedding = get_embedding(" ".join(cand_processed))

    # Calculate cosine similarity
    cosine_sim = cosine_similarity([ref_embedding], [cand_embedding])[0][0]

    # Convert similarity to a percentage
    match_percentage = cosine_sim * 100

    return match_percentage, ref_processed, cand_processed, preprocess_text(question)

def evaluate_answer(reference_answer, candidate_answer, question):
    # Classify the question type
    question_type = classify_question(question)
    
    if question_type == "explanation":
        # Use SBERT for explanation-based questions
        return evaluate_with_sbert(reference_answer, candidate_answer, question)
    else:
        # Use keyword matching for fact-based questions
        return evaluate_with_keywords(reference_answer, candidate_answer, question)

# Example usage tailored for EEE/ECE interview questions
reference_answer = "In electrical engineering, the primary objective is to design systems that efficiently transmit and convert electrical energy, ensuring minimal loss and optimal performance."
question = "what the primary objective of electrical engineering?"
candidate_answer = "The main goal in electrical engineering is to design systems that efficiently transmit and convert electrical energy with minimal loss."

match_percentage, ref_keywords, cand_keywords, question_keywords = evaluate_answer(reference_answer, candidate_answer, question)

print(f"Match Percentage: {match_percentage:.2f}%")
print(f"Reference Keywords: {ref_keywords}")
print(f"Candidate Keywords: {cand_keywords}")
print(f"Question Keywords: {question_keywords}")
