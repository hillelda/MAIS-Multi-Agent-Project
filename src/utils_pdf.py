from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader


def process_pdf_to_markdown(file_path: str) -> str:
    """
    Convert a PDF file to markdown text.
    Tries PyPDFLoader first, falls back to UnstructuredPDFLoader.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        parts = []
        for i, doc in enumerate(documents, start=1):
            content = (doc.page_content or "").strip()
            if content:
                parts.append(f"# Page {i}\n\n{content}\n")
        md = "\n".join(parts).strip()
        if not md:
            raise ValueError("Empty content extracted by PyPDFLoader.")
        return md
    except Exception as primary_err:
        print(f"PyPDFLoader failed: {primary_err}. Trying UnstructuredPDFLoader...")
        try:
            loader = UnstructuredPDFLoader(file_path)
            documents = loader.load()
            parts = []
            for i, doc in enumerate(documents, start=1):
                content = (doc.page_content or "").strip()
                if content:
                    parts.append(f"# Section {i}\n\n{content}\n")
            md = "\n".join(parts).strip()
            if not md:
                raise ValueError("Empty content extracted by UnstructuredPDFLoader.")
            return md
        except Exception as fallback_err:
            raise RuntimeError(
                f"Both PDF loaders failed. Primary: {primary_err}; Fallback: {fallback_err}"
            ) from fallback_err

def load_file_content(file_path: str) -> str:
    """
    Load file content. If PDF, convert to markdown first.
    """
    if file_path.lower().endswith(".pdf"):
        print(f"Processing PDF file: {file_path}")
        content = process_pdf_to_markdown(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    if not content or not content.strip():
        raise ValueError("Loaded article content is empty.")
    return content