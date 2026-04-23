import json
from itertools import islice
from pathlib import Path
from tqdm import tqdm
from friday.core.document import Document
from friday.graph.document.states.document_state import DocumentState
from friday.loggers.logger import Logger


def document_loader_factory(path: str, max=None):
    def document_loader(state: DocumentState):
        logger = Logger.get_logger("node.document_loader")
        logger.info("loading documents from path: %s", path)

        files = []
        folder_path = Path(path)
        all_files = (f for f in folder_path.rglob("*") if f.is_file())
        for file_path in tqdm(islice(all_files, max), total=max):
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

        logger.debug("documents: %s", lambda: [Document(**file) for file in files])

        return [Document(**file) for file in files]

    return document_loader
