
def format_for_metric_evaluation(questions, documents_key='documents'):
    """
    Format question in a common structure for metric evaluation. 
    The common strutcture is the following:
    {
        [question_id]: [document1, document2, ...],
        [question_2_id]: [document1, document2, ...],
        ...
        [question_n_id]: [document1, document2, ...]
    }

    This must be performed for both the gold standard and the retrieved documents in order to use the metric function in this file.

    Args:
        questions (list): List of question dictionaries
        documents_key (str): Key to access the documents in the question dictionary
    Returns:
        dict: Dictionary with question IDs as keys and their corresponding documents as values
    """

    formatted_questions = {}

    for question in questions:
        formatted_questions[question['id']] = question[documents_key]
 
    return formatted_questions


###### METRICS ######

def evaluate_model(retrieved_docs, gold_standard, metric='map') -> float:

    """
    Evaluate the model using the specified metric.

    Args:
        retrieved_docs (dict): Retrieved documents for each question
        gold_standard (dict): Gold standard documents for each question
        metric (str): Metric to use for evaluation ('MAP', 'mean_precision', 'mean_recall', 'f1_score')

    Returns:
        float: Evaluation score
    """

    scores = []
    metric_function = {
        'map': average_precision,
        'mean_precision': precision,
        'mean_recall': recall,
        'f1_score': f1_score
    }.get(metric)

    if not metric_function:
        raise ValueError(f"Unknown metric: {metric}. Use 'MAP', 'mean_precision', 'mean_recall', or 'f1_score'.")


    for question in gold_standard:

        retrieved = retrieved_docs.get(question, [])
    
        gold = gold_standard[question]

        scores.append(metric_function(retrieved, gold))
    


    return sum(scores) / len(scores) if scores else 0.0

def average_precision(retrieved, gold) -> float:

    acumulated_precision = 0.0
    hits = 0
    
    for i, doc in enumerate(retrieved):

        if doc in gold:
            hits += 1

            acumulated_precision += hits / (i + 1)

    return acumulated_precision / len(gold) if gold else 0.0


def precision(retrieved, gold) -> float:

    if not retrieved:
        return 0.0

    true_positives = len([doc for doc in retrieved if doc in gold])

    return true_positives / len(retrieved)


def recall(retrieved, gold) -> float:

    if not gold:
    
        return 0.0

    true_positives = len([doc for doc in retrieved if doc in gold])

    return true_positives / len(gold)


def f1_score(retrieved, gold) -> float:

    p = precision(retrieved, gold)

    r = recall(retrieved, gold)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)
