from typing import List
from abc import ABC, abstractmethod

class TextChunker(ABC):
    @abstractmethod
    def chunk(self):
        pass

class SimpleTextChunker(TextChunker):
    def __init__(self, chunk_size = 0, overlap = 0):
        if overlap > chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")

        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        text = text.strip()

        chunks = []

        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)
            start = end - self.overlap
        return chunks


class MultiGranularChunking(TextChunker):
    def __init__(self, parent_size = 500, child_size = 100):
        self.parent_size = parent_size
        self.child_size = child_size

    def chunk(self, text):
        if not text:
            return []

        child = []
        simple_chunker = SimpleTextChunker(100, 10)
        parents = text.split("\n")
        idx = 0
        for parent in parents:
            chunks = simple_chunker.chunk(parent)
            child.extend(chunks)
        return [parents, child]