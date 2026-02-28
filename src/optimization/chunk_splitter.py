from tqdm import tqdm
from invocation.document_loader import document_loader
from invocation.invoke_llm import invoke_llm
from invocation.JsonDocument import JsonDocument, JsonDocumentCollection
from optimization.Chunk import ChunkResult, Chunks

AVERAGE_CHUNK_SIZE = 100


def make_prompt(document: JsonDocument):
    how_many = (len(document.document) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the collected files that built Rajeev Siewnath's CV.
The document is of type: {document.type}
The document has been retrieved from: {document.source}

A chatbot will use these chunks to answer questions about Rajeev.
You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
This document should probably be split into {how_many} chunks, but you can have more or less as appropriate.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document.document}

Respond with the chunks.
"""


def process_document(document: JsonDocument) -> list[ChunkResult]:
    messages = [{"role": "user", "content": make_prompt(document)}]
    response: Chunks = invoke_llm(input=messages, response_format=Chunks)
    chunked_doc = response.chunks
    return [chunk.as_result(document) for chunk in chunked_doc]


def process_documents(documents: JsonDocumentCollection) -> list[ChunkResult]:
    chunks = []
    for doc in tqdm(documents.json_documents):
        chunks.extend(process_document(doc))
    return chunks


if __name__ == "__main__":
    collection: JsonDocumentCollection = document_loader()
    print(make_prompt(collection.json_documents[0]))
    print(process_document(collection.json_documents[0]))
