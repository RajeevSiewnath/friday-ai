from itertools import islice
import json
from pathlib import Path
from tqdm import tqdm
from pipelines.abstract_pipeline import AbstractPipe
from pipelines.document_loader_pipeline import Document, DocumentCollection


class DocumentLoader(AbstractPipe[DocumentCollection]):
    path: str
    max: int

    def __init__(self, path: str, max=None):
        super().__init__()
        self.path = path
        self.max = max

    def pipe(self, _):
        files = []
        folder_path = Path(self.path)
        all_files = (f for f in folder_path.rglob("*") if f.is_file())
        for file_path in tqdm(islice(all_files, self.max), total=self.max):
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
