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

def get_mesh_terms(pmid):
    """Fetches MeSH descriptors (UI) for a given PubMed ID using E-utilities."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Warning: Failed to fetch PMID {pmid}", file=sys.stderr)
            return set()

        soup = BeautifulSoup(response.content, "xml")
        mesh_terms = set()

        for mesh_heading in soup.find_all("MeshHeading"):
            descriptor = mesh_heading.find("DescriptorName")
            if descriptor and descriptor.get("UI"):
                mesh_terms.add(descriptor.get("UI"))

        return mesh_terms
    except Exception as e:
        print(f"Error fetching PMID {pmid}: {e}", file=sys.stderr)
        return set()

def mesh_terms_for_pmid_list(pmid_list, delay=0.3):
    all_mesh = set()
    for pmid in pmid_list:
        mesh_terms = get_mesh_terms(pmid)
        all_mesh.update(mesh_terms)
        time.sleep(delay)  # To avoid overloading the NCBI servers
    print(f"Processed query: {len(all_mesh)} MeSH terms")
    return all_mesh

def write_mesh_file(question_to_pmids, output_file):
    with open(output_file, "w") as out_f:
        for qid, pmid_list in question_to_pmids.items():
            out_f.write(" ".join(sorted(pmid_list)) + "\n")

    print(f"\nSaved results to: {output_file}")

# Run the script
if __name__ == "__main__":
    result_file = "src/BioASQ-task13b-phaseA-testset4-neural-results.json"
    golden_file = "data/BioASQ-task13bPhaseB-testset4"

    result_pmids = load_documents_json(result_file)
    golden_pmids = load_documents_json(golden_file)

    print(f"Number of questions in result file: {len(result_pmids)}")
    print(f"Number of questions in golden file: {len(golden_pmids)}")
    print(f"Number of shared questions: {len(result_pmids.keys() & golden_pmids.keys())}")

    shared_question_ids = result_pmids.keys() & golden_pmids.keys()

    result_meshes = {q: mesh_terms_for_pmid_list(result_pmids[q]) for q in shared_question_ids}
    golden_meshes = {q: mesh_terms_for_pmid_list(golden_pmids[q]) for q in shared_question_ids}

    write_mesh_file(result_meshes, "system_A_results.txt")
    write_mesh_file(golden_meshes, "true_labels.txt")
