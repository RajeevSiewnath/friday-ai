import pytest
import tempfile
import json
from friday.core.document import Document
from friday.graph.document.states.document_state import DocumentState


@pytest.fixture
def sample_documents():
    return [
        Document(
            id="doc1",
            path="profile.json",
            type="profile",
            content="Software Engineer with 5 years of experience in Python and JavaScript.",
            metadata={"author": "Rajeev", "created": "2024-01-01"},
        ),
        Document(
            id="doc2",
            path="experience.json",
            type="experience",
            content="Worked at Tech Company for 3 years building scalable systems.",
            metadata={"position": "Senior Engineer", "duration": "3 years"},
        ),
        Document(
            id="doc3",
            path="skills.json",
            type="skills",
            content="Python, JavaScript, Go, SQL, Docker, Kubernetes, AWS",
            metadata={"category": "technical"},
        ),
    ]


@pytest.fixture
def document_json_folder(sample_documents):
    with tempfile.TemporaryDirectory() as tmpdir:
        profile_dir = f"{tmpdir}/profile"
        experience_dir = f"{tmpdir}/experience"
        skills_dir = f"{tmpdir}/skills"

        import os

        os.makedirs(profile_dir, exist_ok=True)
        os.makedirs(experience_dir, exist_ok=True)
        os.makedirs(skills_dir, exist_ok=True)

        for doc in sample_documents:
            doc_type = doc.type
            filename = f"{tmpdir}/{doc_type}/file.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "document": doc.content,
                        "metadata": doc.metadata,
                        "id": doc.id,
                    },
                    f,
                )

        yield tmpdir


@pytest.fixture
def document_state(sample_documents):
    return DocumentState(documents=sample_documents)


@pytest.fixture
def empty_document_state():
    return DocumentState(documents=[])
