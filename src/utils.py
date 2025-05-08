import json
from json import JSONDecodeError
import requests

# Importing NLTK for text processing (if we don't use lemmatization or word2vec)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Importation for word2vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors


def load_bioasq_questions(file_path, num_questions=None, test=True):
    """
    Load and process BioASQ dataset questions.

    Args:
        file_path (str): Path to the BioASQ JSON file
        num_questions (int, optional): Number of questions to return. If None, returns all questions.

    Returns:
        list: List of processed question dictionaries
    """
    try:
        # Load the full dataset
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract relevant fields from each question
        if test:
            processed_questions = [
                {
                    'body': question['body'],
                    'type': question['type'],
                    'id': question['id'],
                    'target_documents': question['documents'],
                }
                for question in data['questions']
                if question['type'] in ['yesno', 'factoid', 'summary', 'list']
            ]
        else:
            processed_questions = [
                {
                    'body': question['body'],
                    'type': question['type'],
                    'id': question['id']
                }
                for question in data['questions']
                if question['type'] in ['yesno', 'factoid', 'summary', 'list']
            ]

        # Return requested number of questions or all if num_questions is None
        if num_questions is not None:
            return processed_questions[:num_questions]
        else:
            return processed_questions

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return []
    except KeyError as e:
        print(f"Error: Missing expected key in JSON structure: {e}")
        return []


def get_session():
    """
    This function retrieves a session ID from the BioASQ server as a URL.
    These session IDs can be used for multiple requests but expire after 10 minutes,
    so they must be renewed periodically.

    Returns:
        str: The session ID as a string (e.g., http://bioasq.org:8000/2?-3a641fde%3A19687315e96%3A-7fe2) if the request is successful, None otherwise.
    Raises:
        requests.RequestException: If the GET request fails due to network issues or server errors.
    """
    try:
        GET_SESSION_URL = "http://bioasq.org:8000/pubmed"
        # Sending a GET request to the server
        response = requests.get(GET_SESSION_URL)

        # Checking if the request was successful
        if response.status_code == 200:
            # Extracting the session ID from the response
            return str(response.text)
        else:
            print(f"Error: Received status code {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        raise e


def get_most_relevant_documents(keywords, page=0, documents_per_page=25):
    """
    This function retrieves the most relevant documents from the BioASQ server based on the provided keywords.

    Args:
        keywords (str): The keywords to search for in the documents.
        page (int): The page number for pagination. Default is 0.
        documents_per_page (int): The number of documents to retrieve per page. Default is 10.

    Returns:
        list: A list of objects containing the most relevant documents.
            Content of the objects:
                year (string): The year of publication.
                documentAbstract (string): Abstract of the document.
                meshAnnotations (unclear - Null): MESH annotations of the document. (No idea what this is, usually Null)
                pmid (string): The PubMed ID of the document. Useful in case you want to look for the entire document in PubMed.
                        E.g. pmid = 38939119; https://pubmed.ncbi.nlm.nih.gov/38939119/
                title (string): Title of the document.
                sections (unclear - Null: section of the document? (No idea what this is, usually Null)
                fulltextAvailable (Boolean): Indicates if the full text of the document is available.
                journal (string): Journal in which the document was published?
                meshHeading (list of strings): MESH entities of the document, related to knowledge graphs?



    """
    session_url = get_session()
    request_data = f'json={{"findPubMedCitations": ["{keywords}", {page}, {documents_per_page}]}}'

    response = requests.post(session_url, data=request_data)

    if response.status_code == 200:
        return response.json()['result']['documents']
    else:
        print(f"Error: Received status code {response.status_code}")
        return None

def extract_keywords(text):
    tokens = word_tokenize(text.lower())

    # Filter stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    keywords = [
        word for word in tokens
        if word.isalnum() and word not in stop_words
    ]
    # Return keywords as a string
    # return ' '.join(keywords)

    # Return keywords as a list of strings
    return keywords


def load_vectors_gensim(types_path, vectors_path, vector_size=200):
    word_vectors = KeyedVectors(vector_size=vector_size)
    words = []
    vectors = []

    with open(types_path, 'r', encoding='utf-8') as f_types, open(vectors_path, 'r', encoding='utf-8') as f_vecs:
        for word_line, vec_line in zip(f_types, f_vecs):
            word = word_line.strip()
            vector = np.array([float(num) for num in vec_line.strip().split()], dtype=np.float32)
            words.append(word)
            vectors.append(vector)

    word_vectors.add_vectors(words, vectors)
    return word_vectors


def get_similar_words(word, model, top_k=3):
    if word in model:
        return [w for w, _ in model.most_similar(word, topn=top_k)]
    else:
        return []


def expand_question_with_w2v(question, model):
    tokens = extract_keywords(question)
    expansion = {}
    for token in tokens:
        similar = get_similar_words(token, model)
        if similar:
            expansion[token] = similar
        else:
            expansion[token] = [] 
    return expansion


def build_boolean_query(expansion_dict):
    query_parts = []
    for keyword, similars in expansion_dict.items():
        terms = [keyword] + similars
        group = " OR ".join(terms)
        query_parts.append(f"({group})")
    return " AND ".join(query_parts)


def save_results_to_json(ranked_questions, filename = '../output/output_questions.json'):
    """
    Save the results to a JSON file.
    """

    output_data = {
        "questions": ranked_questions
    }

    with open(filename, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)


    return f"Results saved to {filename}"
