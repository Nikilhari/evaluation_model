from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from collections import Counter
from nltk.corpus import stopwords
import numpy as np
from functools import lru_cache
import string
# Initialize Flask app
app = Flask(__name__)

nltk.download('stopwords')

# Load the set of stopwords and define negation words
stop_words = set(stopwords.words('english'))
negation_words = {"not", "n't", "no", "never"}

# Load the SciBERT model for scientific texts
model = SentenceTransformer('allenai/scibert_scivocab_uncased')

def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    processed_words = []
    negate = False
    for word in words:
        if word in stop_words and word not in negation_words:
            continue
        if word in negation_words:
            negate = True
            continue
        if negate:
            processed_words.append("NOT_" + word)
            negate = False
        else:
            processed_words.append(word)
    return processed_words

def remove_redundant_words(words):
    return list(dict.fromkeys(words))

def remove_question_keywords(answer_words, question_words):
    return [word for word in answer_words if word not in question_words]

@lru_cache(maxsize=10000)
def get_embedding(text):
    return model.encode(text)

def classify_question(question):
    explanation_keywords = {"explain", "describe", "why", "how"}
    if any(keyword in question.lower() for keyword in explanation_keywords):
        return "explanation"
    return "fact"

def evaluate_with_keywords(reference_answer, candidate_answer, question):
    question_processed = preprocess_text(question)
    ref_processed = preprocess_text(reference_answer)
    cand_processed = preprocess_text(candidate_answer)
    ref_filtered = remove_question_keywords(ref_processed, question_processed)
    cand_filtered = remove_question_keywords(cand_processed, question_processed)
    ref_filtered = remove_redundant_words(ref_filtered)
    cand_filtered = remove_redundant_words(cand_filtered)
    unique_words = list(set(ref_filtered + cand_filtered))
    ref_vector = [Counter(ref_filtered)[word] for word in unique_words]
    cand_vector = [Counter(cand_filtered)[word] for word in unique_words]
    cosine_sim = cosine_similarity([ref_vector], [cand_vector])[0][0]
    return cosine_sim * 100, ref_filtered, cand_filtered, question_processed

def evaluate_with_sbert(reference_answer, candidate_answer, question):
    ref_processed = preprocess_text(reference_answer)
    cand_processed = preprocess_text(candidate_answer)
    ref_embedding = get_embedding(" ".join(ref_processed))
    cand_embedding = get_embedding(" ".join(cand_processed))
    cosine_sim = cosine_similarity([ref_embedding], [cand_embedding])[0][0]
    return cosine_sim * 100, ref_processed, cand_processed, preprocess_text(question)

def evaluate_answer(reference_answer, candidate_answer, question):
    question_type = classify_question(question)
    if question_type == "explanation":
        return evaluate_with_sbert(reference_answer, candidate_answer, question)
    else:
        return evaluate_with_keywords(reference_answer, candidate_answer, question)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    question = data['question']
    reference_answer = data['reference_answer']
    candidate_answer = data['candidate_answer']

    match_percentage, ref_keywords, cand_keywords, question_keywords = evaluate_answer(
        reference_answer, candidate_answer, question)

    return jsonify({
        'match_percentage': match_percentage,
        'reference_keywords': ref_keywords,
        'candidate_keywords': cand_keywords,
        'question_keywords': question_keywords
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
