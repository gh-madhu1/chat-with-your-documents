"""
File storage management for uploaded documents.
"""
import shutil
from pathlib import Path
from typing import Optional
from src.config import settings
from src.core.observability import get_logger

logger = get_logger(__name__)


class FileStorage:
    """Manages permanent storage of uploaded files."""
    
    def __init__(self):
        """Initialize file storage."""
        self.storage_dir = settings.file_storage_path
        logger.info("file_storage_initialized", storage_dir=str(self.storage_dir))
    
    def save_file(self, source_path: Path, doc_id: str, original_filename: str) -> str:
        """
        Save uploaded file to permanent storage.
        
        Args:
            source_path: Path to temporary uploaded file
            doc_id: Unique document ID
            original_filename: Original filename
            
        Returns:
            Absolute path to saved file
        """
        # Create subdirectory for this document
        doc_dir = self.storage_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with original filename
        dest_path = doc_dir / original_filename
        shutil.copy2(source_path, dest_path)
        
        logger.info(
            "file_saved",
            doc_id=doc_id,
            filename=original_filename,
            path=str(dest_path)
        )
        
        return str(dest_path.absolute())
    
    def get_file_path(self, doc_id: str, filename: str) -> Optional[str]:
        """
        Get absolute path to a stored file.
        
        Args:
            doc_id: Document ID
            filename: Filename
            
        Returns:
            Absolute path or None if not found
        """
        file_path = self.storage_dir / doc_id / filename
        if file_path.exists():
            return str(file_path.absolute())
        return None
    
    def delete_file(self, doc_id: str):
        """
        Delete stored file and its directory.
        
        Args:
            doc_id: Document ID
        """
        doc_dir = self.storage_dir / doc_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
            logger.info("file_deleted", doc_id=doc_id)


# Global file storage instance
_file_storage = None


def get_file_storage() -> FileStorage:
    """Get the global file storage instance."""
    global _file_storage
    if _file_storage is None:
        _file_storage = FileStorage()
    return _file_storage
