"""
Multi-turn conversation management with context tracking.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
from src.config import settings
from src.core.observability import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Represents a conversation session."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the conversation."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_updated = datetime.now()
    
    def get_recent_messages(self, max_messages: int = None) -> List[Message]:
        """Get recent messages (default from settings)."""
        max_messages = max_messages or settings.max_conversation_history
        return self.messages[-max_messages:]
    
    def get_context_string(self, max_messages: int = None) -> str:
        """
        Get conversation context as a formatted string.
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            Formatted conversation history
        """
        recent_messages = self.get_recent_messages(max_messages)
        
        context_parts = []
        for msg in recent_messages:
            context_parts.append(f"{msg.role.upper()}: {msg.content}")
        
        return "\n\n".join(context_parts)
    
    def is_expired(self) -> bool:
        """Check if conversation has expired."""
        timeout = timedelta(minutes=settings.session_timeout_minutes)
        return datetime.now() - self.last_updated > timeout


class ConversationManager:
    """Manages conversation sessions."""
    
    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}
    
    def create_session(self, metadata: Dict[str, Any] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        conversation = Conversation(
            session_id=session_id,
            metadata=metadata or {}
        )
        
        self.conversations[session_id] = conversation
        
        logger.info("conversation_session_created", session_id=session_id)
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Conversation]:
        """
        Get a conversation session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Conversation object or None if not found/expired
        """
        if session_id not in self.conversations:
            logger.warning("conversation_session_not_found", session_id=session_id)
            return None
        
        conversation = self.conversations[session_id]
        
        # Check if expired
        if conversation.is_expired():
            logger.info("conversation_session_expired", session_id=session_id)
            self.delete_session(session_id)
            return None
        
        return conversation
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Add a message to a conversation.
        
        Args:
            session_id: Session ID
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            True if successful, False if session not found
        """
        conversation = self.get_session(session_id)
        if not conversation:
            return False
        
        conversation.add_message(role, content, metadata)
        
        logger.info(
            "message_added_to_conversation",
            session_id=session_id,
            role=role,
            message_length=len(content)
        )
        
        return True
    
    def get_context(self, session_id: str, max_messages: int = None) -> Optional[str]:
        """
        Get conversation context string.
        
        Args:
            session_id: Session ID
            max_messages: Maximum number of recent messages
            
        Returns:
            Context string or None if session not found
        """
        conversation = self.get_session(session_id)
        if not conversation:
            return None
        
        return conversation.get_context_string(max_messages)
    
    def delete_session(self, session_id: str):
        """Delete a conversation session."""
        if session_id in self.conversations:
            del self.conversations[session_id]
            logger.info("conversation_session_deleted", session_id=session_id)
    
    def cleanup_expired_sessions(self):
        """Remove all expired sessions."""
        expired_sessions = [
            session_id
            for session_id, conv in self.conversations.items()
            if conv.is_expired()
        ]
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(
                "expired_sessions_cleaned",
                count=len(expired_sessions)
            )
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return [
            {
                "session_id": conv.session_id,
                "created_at": conv.created_at.isoformat(),
                "last_updated": conv.last_updated.isoformat(),
                "message_count": len(conv.messages),
                "metadata": conv.metadata
            }
            for conv in self.conversations.values()
            if not conv.is_expired()
        ]


# Global conversation manager
_conversation_manager = None


def get_conversation_manager() -> ConversationManager:
    """Get the global conversation manager instance."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
