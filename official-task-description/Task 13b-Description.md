# BioASQ Task 13b: Biomedical Semantic QA (Involves IR, QA, Summarization)

Link to dataset: [BioASK13](https://participants-area.bioasq.org/Tasks/13b/trainingDataset/)

Task 13b will use benchmark datasets containing training and test biomedical questions, in English, along with gold standard (reference) answers. 

The participants will have to respond to each test question with relevant articles (in English, from designated article repositories), relevant snippets (from the relevant articles), exact answers (e.g., named entities in the case of factoid questions) and 'ideal' answers (English paragraph-sized summaries). 

The test dataset of Task 13b will be released in batches, each containing approximately 100 questions. The task will start on March, 2025. Separate winners will be announced for each batch. Participation in the task can be partial; for example, it is acceptable to participate in only some of the batches, to return only relevant articles (and no article snippets), or to return only exact answers (or only `ideal' answers). System responses will be evaluated both automatically and manually.

# Notes about the task:
- What Phase do we have to do?:
    - Just Phase A is enough for this task.
    - We can also do Phase A+ and/or B for extra points.
- Number documents per question:
    - At most 10 documents per question.

# Training data - JSON explanation:

The JSON file contains an array of "questions" objects. Each question object contains the following fields:

- "body" (string): The question text.

- "documents" (Array of strings): Url of the documents.

- "ideal_answer" (string): Hand-written ideal answer (this answer is not explicitly in any snippet of the documents).

- "triplets" (Array of objects): Each object contains the following fields:
    - "p" (string): The predicate of the triplet.
    - "s" (string): The subject of the triplet.
    - "o" (string): The object of the triplet.

- "exact_answer": 
    - Array of arrays of strings: Array of multiple exact answers. Inside each each array, there can be the same exact answer in different formats.
    - Array of strings: Same as above, but only one array of strings.
    - String: Simple answer "yes" or "no".

- "type" (string): Type of the question. It can be one of the following:
    - "yesno": Yes/No question.
    - "factoid": Factoid question.
    - "list": List question.
    - "summary": Summary question.

- "concepts" (Array of strings): List of URIs of relevant medical concepts for the question.

- "id" (string): id of the question.

- "snippets" (Array of objects): Each object contains the following fields:
    - "offsetInBeginSection" (int): Number of characters from the beginning of the section (of the document) to the beginning of the snippet.
    - "offsetInEndSection" (int): Number of characters from the beginning of the section (of the document) to the end of the snippet.
    - "text" (string): The snippet text (possible answer to the question).
    - "beginSection" (string): The section of the document where the snippet is found. Common values are "title", "abstract", and "sections.0".
    - "endSection" (string): The section of the document where the snippet is found.


## Notes about the dataset:

### Type of questions:

The benchmark datasets contain four types of questions:
- Yes/no questions: 

    These are questions that, strictly speaking, require "yes" or "no" answers, though of course in practice longer answers will often be desirable. For example, "Do CpG islands colocalise with transcription start sites?" is a yes/no question.

- Factoid questions: 

    These are questions that, strictly speaking, require a particular entity name (e.g., of a disease, drug, or gene), a number, or a similar short expression as an answer, though again a longer answer may be desirable in practice. For example, "Which virus is best known as the cause of infectious mononucleosis?" is a factoid question.

- List questions: 

    These are questions that, strictly speaking, require a list of entity names (e.g., a list of gene names), numbers, or similar short expressions as an answer; again, in practice additional information may be desirable. For example, "Which are the Raf kinase inhibitors?" is a list question.

- Summary questions: 
    
    These are questions that do not belong in any of the previous categories and can only be answered by producing a short text summarizing the most prominent relevant information. For example, "What is the treatment of infectious mononucleosis?" is a summary question.


### Phase A

In Phase A of Task 13b, the participants will be provided with English questions q1, q2,...,qn. For each question qi, each participating system will be required to return any (ideally all) of the following lists:

- A list of at most **10 relevant articles** (documents) di,1, di,2, di,3,... from the designated article repositories. 
    - The list should be ordered by decreasing confidence, i.e., di,1, should be the article that the system considers to be the most relevant to the question q1,, di,2, should be the article that the system considers to be the second most relevant etc. 
    - A single article list will be returned per question and participating system, and the list may contain articles from multiple designated repositories. The returned article list will actually contain unique article identifiers (obtained from the repositories).

- A list of at most **10 relevant text snippets** si,1, si,2, si,3,... from the returned articles. 
    - The list should be ordered by decreasing confidence. A single snippet list will be returned per question and participating system, and the list may contain any number (or no) snippets from any of the returned articles di,1, di,2, di,3,... Each snippet will be represented by the unique identifier of the article it comes from, the identifier of the section the snippet starts in, the offset of the first character of the snippet in the section the snippet starts in, the identifier of the section the snippet ends in, and the offset of the last character of the snippet in the section the snippet ends in. The snippets themselves will also have to be returned (as strings).