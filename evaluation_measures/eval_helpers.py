import requests
from bs4 import BeautifulSoup
import time
import sys
import json

def load_documents_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    question_to_pmids = dict()
    for q in data["questions"]:
        doc_urls = q.get("documents", [])
        qid = q.get("id")
        pmids = [url.split("/")[-1] for url in doc_urls if "pubmed" in url]
        question_to_pmids[qid] = pmids
    # print(f"question_to_pmids: {question_to_pmids}")
    return question_to_pmids


def mesh_terms_for_pmid_list(pmid_list):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmid_list),
        "retmode": "xml"
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Warning: Failed to fetch PMID {pmid_list} with status code {response.status_code}", file=sys.stderr)
            return set()

        soup = BeautifulSoup(response.content, "xml")
        mesh_terms = set()

        for mesh_heading in soup.find_all("MeshHeading"):
            descriptor = mesh_heading.find("DescriptorName") # type: ignore
            if descriptor and descriptor.get("UI"): # type: ignore
                mesh_terms.add(descriptor.get("UI")) # type: ignore

        return mesh_terms
    except Exception as e:
        print(f"Error fetching PMID {pmid_list}: {e}", file=sys.stderr)
        return set()

def write_mesh_file(question_to_pmids, output_file):
    with open(output_file, "w") as out_f:
        for _, pmid_list in question_to_pmids.items():
            out_f.write(" ".join(sorted(pmid_list)) + "\n")

    print(f"Saved results to: {output_file}")

# Run the script
if __name__ == "__main__":
    # result_name = "BioASQ-task13b-phaseA-testset4-neural-results2"
    # result_file = "evaluation_measures/results/json/" + result_name + ".json" # "src/BioASQ-task13b-phaseA-testset4-neural-results.json"
    # golden_file = "data/BioASQ-task13bPhaseB-testset4"
    if len(sys.argv) != 4:
        print("Usage: python eval_helpers.py <golden_file> <result_file> <output_name>")
        sys.exit(1)
    golden_file = sys.argv[1]
    result_file = sys.argv[2]
    output_name = sys.argv[3]

    result_pmids = load_documents_json(result_file)
    golden_pmids = load_documents_json(golden_file)

    print(f"Number of questions in result file: {len(result_pmids)}")
    print(f"Number of questions in golden file: {len(golden_pmids)}")
    print(f"Number of shared questions: {len(result_pmids.keys() & golden_pmids.keys())}")

    shared_question_ids = result_pmids.keys() & golden_pmids.keys()
    # As a for loop
    result_meshes = {}
    golden_meshes = {}
    for i,q in enumerate(shared_question_ids):
        print(f"Processing question {i+1}/{len(shared_question_ids)}")
        result_meshes[q] = mesh_terms_for_pmid_list(result_pmids[q])
        golden_meshes[q] = mesh_terms_for_pmid_list(golden_pmids[q])
        print(f"  Result question {q} has {len(result_meshes[q])} mesh terms")
        print(f"  Golden question {q} has {len(golden_meshes[q])} mesh terms")

    # Filter so that we only keep questions that have mesh terms in both result and golden
    filtered_result_meshes = {q: result_meshes[q] for q in shared_question_ids if result_meshes[q] and golden_meshes[q]}
    filtered_golden_meshes = {q: golden_meshes[q] for q in shared_question_ids if result_meshes[q] and golden_meshes[q]}
    print(f"Number of questions with mesh terms in result file: {len(filtered_result_meshes)}")
    print(f"Number of questions with mesh terms in golden file: {len(filtered_golden_meshes)}") # Should match
    # Write the results to files
    write_mesh_file(filtered_result_meshes, "evaluation_measures/results/mesh/" + output_name + "_pred.txt")
    write_mesh_file(filtered_golden_meshes, "evaluation_measures/results/mesh/" + output_name + "_gold.txt")

