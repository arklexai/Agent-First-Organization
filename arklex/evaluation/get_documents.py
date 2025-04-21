import os
import sys
import json
import pickle
from pathlib import Path
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from arklex.utils.loader import Loader, CrawledObject, SourceType

def get_domain_info(documents):
    summary = None
    for doc in documents:
        if doc['URL'] == 'summary':
            summary = doc['content']
            break
    return summary

def load_docs(document_dir, doc_config, limit=10):
    if document_dir is not None:
        try:
            if "rag_docs" not in doc_config:
                if "task_docs" not in doc_config:
                    raise ValueError("The config json file must have a key 'rag_docs' or 'task_docs' with a list of documents to load.")
            else:
                rag_docs = doc_config['task_docs']
                filename = "task_documents.pkl"
            filepath = os.path.join(document_dir, filename)
            total_num_docs = sum([doc.get("num") if doc.get("num") else 1 for doc in rag_docs])
            loader = Loader()
            if Path(filepath).exists():
                docs = pickle.load(open(os.path.join(document_dir, filename), "rb"))
            else:
                docs = []
                for doc in rag_docs:
                    source = doc.get("source")
                    if doc.get('type') != 'local':
                        num_docs = doc.get("num") if doc.get("num") else 1
                        urls = loader.get_all_urls(source, num_docs)
                        crawled_urls = loader.to_crawled_url_objs(urls)
                        docs.extend(crawled_urls)
                    elif doc.get('type') == 'local':
                        file_list = [os.path.join(source, f) for f in os.listdir(source)]
                        docs.extend(loader.to_crawled_local_objs(file_list))
                Loader.save(filepath, docs)
            if total_num_docs > 50:
                limit = total_num_docs // 5
            else:
                limit = 10
            if isinstance(docs[0], CrawledObject):
                documents = []
                # Get candidate websites for only web urls
                web_docs = list(filter(lambda x: x.source_type == SourceType.WEB, docs))
                local_docs = list(filter(lambda x: x.source_type == SourceType.LOCAL, docs))
                documents.extend(loader.get_candidates_websites(web_docs, limit))
                documents.extend(local_docs)
                documents = [doc.to_dict() for doc in documents]
            else:
                raise ValueError("The documents must be a list of CrawledObject objects.")
        except Exception as e:
            print(f"Error loading documents: {e}")
            documents = []
    else:
        documents = []
    return documents

if __name__ == "__main__":
    doc_config = json.load(open('./temp_files/richtech_config.json'))
    docs = load_docs('./temp_files', doc_config, 10)