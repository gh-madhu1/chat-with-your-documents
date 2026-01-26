"""
Document processing pipeline for PDF and text files.
"""
from pathlib import Path
from typing import Optional
import pypdf
import pdfplumber
from dataclasses import dataclass
from src.core.file_storage import get_file_storage
from src.core.observability import get_logger, track_metrics

logger = get_logger(__name__)


@dataclass
class Document:
    """Represents a processed document."""
    content: str
    metadata: dict
    source: str
    doc_id: str
    # Maps content ranges to page numbers for multi-page documents
    page_mapping: dict = None


class DocumentProcessor:
    """Process and clean documents from various formats."""

    def __init__(self):
        self.supported_formats = {".pdf", ".txt", ".md"}
        self.file_storage = get_file_storage()

    @track_metrics("document_processing")
    async def process_file(self, file_path: Path, doc_id: str, original_filename: str = None) -> Document:
        """
        Process a file and extract its content.

        Args:
            file_path: Path to the file
            doc_id: Unique document identifier
            original_filename: Original filename before any renaming (optional)

        Returns:
            Processed Document object

        Raises:
            ValueError: If file format is not supported
        """
        suffix = file_path.suffix.lower()

        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Use original filename if provided, otherwise use current filename
        display_filename = original_filename if original_filename else file_path.name

        logger.info("processing_document",
                    file_path=str(file_path), doc_id=doc_id, original_filename=display_filename)

        # Save file to permanent storage
        stored_path = self.file_storage.save_file(
            file_path, doc_id, display_filename)

        if suffix == ".pdf":
            content, metadata, page_mapping = await self._process_pdf(file_path)
        else:
            content, metadata, page_mapping = await self._process_text(file_path)

        # Clean and normalize content
        content = self._clean_text(content)

        metadata.update({
            "file_name": display_filename,
            "file_type": suffix,
            "file_size": file_path.stat().st_size,
            "file_path": stored_path,  # Add file path for citations
            "doc_id": doc_id,
            "page": 1  # Default page, will be overridden by chunking strategy if multi-page
        })

        return Document(
            content=content,
            metadata=metadata,
            source=str(file_path),
            doc_id=doc_id,
            page_mapping=page_mapping
        )

    async def _process_pdf(self, file_path: Path) -> tuple[str, dict, dict]:
        """Extract text and metadata from PDF with page tracking."""
        text_parts = []
        page_mapping = {}  # Track which character positions belong to which pages
        metadata = {"pages": 0, "extraction_method": "pdfplumber"}
        current_position = 0

        try:
            # Try pdfplumber first (better text extraction)
            with pdfplumber.open(file_path) as pdf:
                metadata["pages"] = len(pdf.pages)

                # Pages start at 1
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Record which page this text belongs to
                        start_pos = current_position
                        text_parts.append(text)
                        # +2 for the "\n\n" separator
                        current_position += len(text) + 2
                        page_mapping[f"{start_pos}:{current_position}"] = page_num

                # Extract PDF metadata
                if pdf.metadata:
                    metadata.update({
                        "title": pdf.metadata.get("Title", ""),
                        "author": pdf.metadata.get("Author", ""),
                        "subject": pdf.metadata.get("Subject", ""),
                        "creator": pdf.metadata.get("Creator", ""),
                    })

        except Exception as e:
            logger.warning("pdfplumber_failed_fallback_to_pypdf", error=str(e))

            # Fallback to pypdf
            metadata["extraction_method"] = "pypdf"
            with open(file_path, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)
                metadata["pages"] = len(pdf_reader.pages)
                current_position = 0

                # Pages start at 1
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Record which page this text belongs to
                        start_pos = current_position
                        text_parts.append(text)
                        # +2 for the "\n\n" separator
                        current_position += len(text) + 2
                        page_mapping[f"{start_pos}:{current_position}"] = page_num

                # Extract metadata
                if pdf_reader.metadata:
                    metadata.update({
                        "title": pdf_reader.metadata.get("/Title", ""),
                        "author": pdf_reader.metadata.get("/Author", ""),
                        "subject": pdf_reader.metadata.get("/Subject", ""),
                    })

        content = "\n\n".join(text_parts)
        return content, metadata, page_mapping

    async def _process_text(self, file_path: Path) -> tuple[str, dict, dict]:
        """Extract text from plain text files."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with different encoding
            content = file_path.read_text(encoding="latin-1")

        metadata = {
            "pages": 1,
            "extraction_method": "text_read"
        }

        # Text files are single page
        page_mapping = {"0:" + str(len(content)): 1}

        return content, metadata, page_mapping

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        - Remove excessive whitespace
        - Normalize line breaks
        - Remove special characters that may cause issues
        """
        # Replace multiple spaces with single space
        text = " ".join(text.split())

        # Normalize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove multiple consecutive newlines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def validate_file(self, file_path: Path, max_size_mb: int = 50) -> tuple[bool, Optional[str]]:
        """
        Validate file before processing.

        Returns:
            (is_valid, error_message)
        """
        if not file_path.exists():
            return False, "File does not exist"

        if not file_path.is_file():
            return False, "Path is not a file"

        if file_path.suffix.lower() not in self.supported_formats:
            return False, f"Unsupported format. Supported: {self.supported_formats}"

        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File too large ({size_mb:.1f}MB). Max: {max_size_mb}MB"

        return True, None
