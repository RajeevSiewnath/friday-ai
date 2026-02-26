import json
from pathlib import Path
from invocation.JsonDocument import JsonDocument, JsonDocumentCollection


def document_loader() -> JsonDocumentCollection:
    files = []
    folder_path = Path("data")
    for file_path in folder_path.rglob("*"):
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            document = json.loads(content)
            files.append(
                {
                    "source": str(file_path.relative_to(folder_path)),
                    "type": str(file_path.parent.relative_to(folder_path)),
                    "document": document["document"],
                    "metadata": document["metadata"],
                    "id": document["id"],
                }
            )
    return JsonDocumentCollection.from_docs([JsonDocument(**file) for file in files])


if __name__ == "__main__":
    print(document_loader())
