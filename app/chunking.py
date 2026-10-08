"""Split long text into overlapping chunks.

Why chunk? An LLM has a limited context window, and a small, focused chunk
matches a question far better than a whole 50-page PDF would.
Why overlap? So a sentence cut at a chunk border still appears whole in the
next chunk.
"""
import re


def split_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer to end the chunk at a sentence boundary in its second half.
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("? "),
                      window.rfind("! "), window.rfind("। "))
            if cut > size * 0.5:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        # do not start a chunk in the middle of a word
        if text[start - 1] != " ":
            nxt = text.find(" ", start)
            start = nxt + 1 if nxt != -1 else len(text)
    return chunks
