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
            start = end + 1
            end -= self.overlap
        return chunks


