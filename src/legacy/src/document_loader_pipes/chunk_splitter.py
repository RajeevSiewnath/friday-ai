from copy import deepcopy
from pydantic import BaseModel, Field
from tqdm import tqdm
from pipelines.pipeline import Pipe
from models.document import Document, DocumentCollection


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query"
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    @property
    def get_chunked_content(self):
        return self.headline + "\n\n" + self.summary + "\n\n" + self.original_text


class Chunks(BaseModel):
    chunks: list[Chunk] = Field(description="Collection of chunks")


class ChunkSplitter(Pipe[DocumentCollection]):
    average_chunk_size: int

    def __init__(self, average_chunk_size: int = 100):
        super().__init__()
        self.average_chunk_size = average_chunk_size

    def make_prompt(self, document: Document):
        how_many = (len(document.content) // self.average_chunk_size) + 1
        return f"""
    You take a document and you split the document into overlapping chunks for a KnowledgeBase.

    The document is from the collected files that built Rajeev Siewnath's CV.
    The document is of type: {document.type}
    The document has been retrieved from: {document.path}

    A chatbot will use these chunks to answer questions about Rajeev.
    You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
    This document should probably be split into {how_many} chunks, but you can have more or less as appropriate.
    There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

    For each chunk, you should provide a headline, a summary, and the original text of the chunk.
    Together your chunks should represent the entire document with overlap.

    Here is the document:

    {document.content}

    Respond with the chunks.
    """

    def process_document(self, document: Document) -> DocumentCollection:
        messages = [{"role": "user", "content": self.make_prompt(document)}]
        chunks: Chunks = self.llm.invoke(input=messages, response_format=Chunks)
        doc_copy = deepcopy(document)
        return DocumentCollection.from_docs(
            [
                Document(
                    id=doc_copy.id + "_chunk_" + str(index + 1),
                    content=chunk.get_chunked_content,
                    **doc_copy.model_dump(exclude={"content", "id"}),
                )
                for index, chunk in enumerate(chunks.chunks)
            ]
        )

    def run(self, input):
        chunks: DocumentCollection = DocumentCollection.from_docs([])
        for doc in tqdm(input.documents):
            chunks += self.process_document(doc)
        return chunks
