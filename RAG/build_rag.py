import json
from data_store import Chroma
from doc_loader import DocLoader
from embedding import get_embedding
import os


def load_qa_pairs(jsonl_file):
    qa_pairs = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            qa_text = f"Question: {data['question']}\nAnswer: {data['option' + chr(65 + data['answer'])]}\nExplanation: {data['answer_explained']}"
            qa_pairs.append({
                'text': qa_text,
                'metadata': {
                    'question_id': data['question_id'],
                    'type': data['type']
                }
            })
    return qa_pairs

def build_rag_database(jsonl_file, db_path):
    # Load QA pairs
    qa_pairs = load_qa_pairs(jsonl_file)
    
    # Process each QA pair
    for idx, qa_pair in enumerate(qa_pairs):
        # Generate embedding for the QA pair
        embedding = get_embedding(qa_pair['text'])
        
        # Prepare data for Chroma
        chroma_data = {
            "embeddings": [embedding],
            "contents": [qa_pair['text']],
            "metadatas": [qa_pair['metadata']],
            "ids": [str(idx)]
        }
        
        # Add to Chroma database
        chroma_instance.add_data_to(chroma_data)
    
    print(f"Successfully built RAG database with {len(qa_pairs)} QA pairs")
    return chroma_instance

def query_rag_database(chroma_instance, query_text):
    query_embedding = get_embedding(query_text)
    results = chroma_instance.query(query_embedding )
    return results

if __name__ == "__main__":
    jsonl_file = "TextQA_all.jsonl"
    db_path = "rag_database"
    
    if not os.path.exists(db_path):
        chroma_instance = build_rag_database(jsonl_file, db_path)
    else:   
        chroma_instance = Chroma(
            default_database_path=db_path,
            first_collection_name="qa_collection",
            top_k=5
        )

    query = "How to proceed at an uncontrolled intersection?"
    results = query_rag_database(chroma_instance, query)
    
    # Add error handling and better result formatting
    if results and 'documents' in results and results['documents']:
        for doc in results['documents'][0]:
            print("\nRelevant QA pair:")
            print(doc)
    else:
        print("No results found for the query.")
