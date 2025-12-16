import json
import uuid
from pathlib import Path

DOC_FILE = Path("data/documents.json")

def save_document_name(file_name: str):
    DOC_FILE.parent.mkdir(exist_ok=True)

    if not DOC_FILE.exists():
        DOC_FILE.write_text("[]")

    with open(DOC_FILE, "r+") as f:
        data = json.load(f)

        doc_id = str(uuid.uuid4())
        data.append({
            "doc_id": doc_id,
            "file_name": file_name
        })

        f.seek(0)
        json.dump(data, f, indent=2)

    return doc_id
