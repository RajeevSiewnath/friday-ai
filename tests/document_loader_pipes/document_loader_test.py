import json
from pathlib import Path
from tqdm import tqdm
from pipelines.abstract_pipeline import AbstractPipe
from pipelines.document_loader_pipeline import Document, DocumentCollection


class DocumentLoader(AbstractPipe[DocumentCollection]):
    path: str

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def pipe(self, _):
        files = []
        folder_path = Path(self.path)
        for file_path in tqdm(folder_path.rglob("*")):
            if file_path.is_file():
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                document = json.loads(content)
                files.append(
                    {
                        "path": str(file_path.relative_to(folder_path)),
                        "type": str(file_path.parent.relative_to(folder_path)),
                        "content": document["document"],
                        "metadata": document["metadata"],
                        "id": document["id"],
                    }
                )
        return DocumentCollection.from_docs([Document(**file) for file in files])


if __name__ == "__main__":
    print(DocumentLoader("data").pipe(DocumentCollection.from_docs([])))
