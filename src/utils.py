import json
from json import JSONDecodeError
import requests
import xml.etree.ElementTree as ET

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

def load_bioasq_test_questions(file_path):
    """
    Load and process BioASQ test dataset questions.

    Args:
        file_path (str): Path to the BioASQ test JSON file

    Returns:
        list: List of processed question dictionaries
    """
    try:
        # Load the full dataset
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract questions (test data has a simpler structure)
        processed_questions = [
            {
                'body': question['body'],
                'type': question['type'],
                'id': question['id'],
            }
            for question in data['questions']
        ]

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
    except UnicodeDecodeError as e:
        print(f"Error: Unicode decoding error: {e}. Try different encoding.")
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
    This function retrieves the most relevant documents from PubMed using the E-utilities API.

    Args:
        keywords (str): The keywords to search for in the documents.
        page (int): The page number for pagination. Default is 0. (Unused)
        documents_per_page (int): The number of documents to retrieve per page. Default is 25.

    Returns:
        list: A list of objects containing the most relevant documents.
            Content of the objects:
                pmid (string): The PubMed ID of the document.
                title (string): Title of the document.
                documentAbstract (string): Abstract of the document.
                year (string): Year of publication (if available).
                journal (string): Journal name (if available).
    """
    # Step 1: Search for PubMed IDs
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": keywords,
        "retmode": "json",
        "retmax": documents_per_page
    }
    search_response = requests.get(search_url, params=search_params)

    if search_response.status_code == 200:
        id_list = search_response.json()["esearchresult"]["idlist"]
        if not id_list:
            return []

        # Step 2: Fetch details for those IDs
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml"
        }
        fetch_response = requests.get(fetch_url, params=fetch_params)

        if fetch_response.status_code == 200:
            articles = []
            root = ET.fromstring(fetch_response.content)
            for article in root.findall(".//PubmedArticle"):
                pmid = article.findtext(".//PMID")
                title = article.findtext(".//ArticleTitle") or ""
                abstract = " ".join([abst.text or "" for abst in article.findall(".//AbstractText")])
                # Try to get year and journal if available
                year = article.findtext(".//PubDate/Year") or ""
                journal = article.findtext(".//Journal/Title") or ""
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "documentAbstract": abstract,
                    "year": year,
                    "journal": journal
                })
            return articles
        else:
            print(f"Error: Received status code {fetch_response.status_code} from efetch")
            return []
    else:
        print(f"Error: Received status code {search_response.status_code} from esearch")
        return []

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
