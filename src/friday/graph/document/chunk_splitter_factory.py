from copy import deepcopy
from logging import LoggerAdapter
from pydantic import BaseModel, Field
from tqdm import tqdm
from langgraph.runtime import Runtime
from openai.types.responses import ParsedResponse
from friday.core.document import Document
from friday.core.llm import LLM
from friday.graph.document.reducers.document_reducer import DocumentReducerClearAction
from friday.graph.document.states.document_state import DocumentState
from friday.graph.query.contexts.llm_context import LLMContext
from friday.loggers.logger import Logger


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


def chunk_splitter_factory(average_chunk_size: int = 100):
    def make_prompt(document: Document):
        how_many = (len(document.content) // average_chunk_size) + 1
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

    async def process_document(
        document: Document, llm: LLM, logger: LoggerAdapter
    ) -> list[Document]:
        messages = [{"role": "user", "content": make_prompt(document)}]
        logger.trace("query: %s", lambda: messages)

        response: ParsedResponse[Chunks] = await llm.invoke(
            input=messages, response_format=Chunks
        )
        logger.trace("response: %s", lambda: response)

        doc_copy = deepcopy(document)
        return [
            Document(
                id=doc_copy.id + "_chunk_" + str(index + 1),
                content=chunk.get_chunked_content,
                **doc_copy.model_dump(exclude={"content", "id"}),
            )
            for index, chunk in enumerate(response.output_parsed.chunks)
        ]

    async def chunk_splitter(state: DocumentState, runtime: Runtime[LLMContext]):
        logger = Logger.get_logger("node.chunk_splitter")
        logger.debug("splitting document into chunks")
        logger.trace("documents: %s", lambda: state["documents"])

        chunks: list[Document] = []
        for doc in tqdm(state["documents"]):
            chunks += await process_document(doc, runtime.context.llm, logger)
        logger.trace("chunks: %s", lambda: chunks)

        return {"documents": DocumentReducerClearAction(chunks)}

    return chunk_splitter
