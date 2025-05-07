from utils import get_most_relevant_documents
from utils import extract_keywords
import torch
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

def get_embeddings(texts, model, tokenizer, max_length=512):
    """
    Generate embeddings for a list of texts using the provided model and tokenizer.

    Args:
        texts (list): List of text strings to embed
        model: Transformer model
        tokenizer: Tokenizer for the model
        max_length (int): Maximum sequence length

    Returns:
        numpy.ndarray: Array of embeddings
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tokenize texts
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    ).to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Mean pooling - take average of all token embeddings
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.last_hidden_state.size()).float()
    sum_embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask

    # Convert from PyTorch tensor to numpy array
    return embeddings.cpu().numpy()
def retrieve_and_rank_documents_neural(question, model, tokenizer, max_docs=50):
    """
    Retrieve documents using the PubMed API and rank them using neural embeddings.

    Args:
        question (dict): Question dictionary with 'body' field
        model: Transformer model for embeddings
        tokenizer: Tokenizer for the model
        max_docs (int): Maximum number of documents to retrieve initially

    Returns:
        list: Top 10 ranked documents
    """
    # Extract keywords for API search (same as baseline)
    keywords = ' '.join(extract_keywords(question['body']))

    # Get documents from PubMed API
    documents = get_most_relevant_documents(keywords, documents_per_page=max_docs)

    if not documents:
        return []

    # Create text representations for documents (title + abstract)
    doc_texts = [f"{doc['title']} {doc['documentAbstract']}" for doc in documents]

    # Generate embeddings for question and documents
    question_embedding = get_embeddings([question['body']], model, tokenizer)[0]
    document_embeddings = get_embeddings(doc_texts, model, tokenizer)

    # Calculate cosine similarity between question and each document
    similarities = cosine_similarity([question_embedding], document_embeddings)[0]

    # Combine documents with their similarity scores and sort
    ranked_docs = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)

    # Return top 10 documents
    return [doc for doc, _ in ranked_docs[:10]]
def extract_and_rank_snippets_neural(question, documents, model, tokenizer, max_snippets=10):
    """
    Extract snippets from documents and rank them using neural embeddings.

    Args:
        question (dict): Question dictionary with 'body' field
        documents (list): List of document dictionaries
        model: Transformer model for embeddings
        tokenizer: Tokenizer for the model
        max_snippets (int): Maximum number of snippets to return

    Returns:
        list: Top ranked snippets with metadata
    """
    # Generate question embedding
    question_embedding = get_embeddings([question['body']], model, tokenizer)[0]

    all_snippets = []

    # Process each document
    for doc in documents:
        # Combine title and abstract
        full_text = f"{doc['title']} {doc['documentAbstract']}"

        # Split into sentences
        sentences = sent_tokenize(full_text)

        # Track offsets for each sentence
        current_offset = 0
        sentence_offsets = []

        for sentence in sentences:
            start_offset = full_text.find(sentence, current_offset)
            end_offset = start_offset + len(sentence) - 1
            sentence_offsets.append((start_offset, end_offset))
            current_offset = end_offset + 1

        # Generate embeddings for all sentences
        if sentences:
            sentence_embeddings = get_embeddings(sentences, model, tokenizer)

            # Calculate similarity scores
            similarities = cosine_similarity([question_embedding], sentence_embeddings)[0]

            # Create snippet objects with metadata
            for i, (sentence, score) in enumerate(zip(sentences, similarities)):
                start_offset, end_offset = sentence_offsets[i]

                # Determine if this is from title or abstract
                if start_offset < len(doc['title']):
                    section = "title"
                else:
                    section = "abstract"
                    # Adjust offset for abstract
                    if start_offset >= len(doc['title']):
                        start_offset = start_offset - len(doc['title']) - 1
                        end_offset = end_offset - len(doc['title']) - 1

                snippet = {
                    'document': f"http://www.ncbi.nlm.nih.gov/pubmed/{doc['pmid']}",
                    'text': sentence,
                    'offsetInBeginSection': start_offset,
                    'offsetInEndSection': end_offset,
                    'beginSection': section,
                    'endSection': section,
                    'score': float(score)
                }
                all_snippets.append(snippet)

    # Sort all snippets by score and select top ones
    ranked_snippets = sorted(all_snippets, key=lambda x: x['score'], reverse=True)

    return ranked_snippets[:max_snippets]