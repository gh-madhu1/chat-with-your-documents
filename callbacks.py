"""
Pipeline stage callbacks for observability and streaming.
"""
from typing import Any, Dict, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from src.core.observability import get_logger

logger = get_logger(__name__)


class PipelineStage(str, Enum):
    """Pipeline stages for tracking."""
    INGESTION = "ingestion"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    INFERENCE = "inference"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class StageEvent:
    """Represents a pipeline stage event."""
    stage: PipelineStage
    status: str  # 'started', 'progress', 'completed', 'error'
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage": self.stage.value,
            "status": self.status,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class PipelineCallback:
    """Base callback for pipeline events."""
    
    async def on_stage_start(self, stage: PipelineStage, metadata: Dict[str, Any] = None):
        """Called when a stage starts."""
        pass
    
    async def on_stage_progress(
        self,
        stage: PipelineStage,
        progress: float,
        message: str = "",
        metadata: Dict[str, Any] = None
    ):
        """Called during stage progress."""
        pass
    
    async def on_stage_complete(self, stage: PipelineStage, metadata: Dict[str, Any] = None):
        """Called when a stage completes."""
        pass
    
    async def on_stage_error(self, stage: PipelineStage, error: Exception, metadata: Dict[str, Any] = None):
        """Called when a stage encounters an error."""
        pass


class LoggingCallback(PipelineCallback):
    """Callback that logs all events."""
    
    async def on_stage_start(self, stage: PipelineStage, metadata: Dict[str, Any] = None):
        logger.info(f"{stage.value}_started", **(metadata or {}))
    
    async def on_stage_progress(
        self,
        stage: PipelineStage,
        progress: float,
        message: str = "",
        metadata: Dict[str, Any] = None
    ):
        logger.info(
            f"{stage.value}_progress",
            progress=progress,
            message=message,
            **(metadata or {})
        )
    
    async def on_stage_complete(self, stage: PipelineStage, metadata: Dict[str, Any] = None):
        logger.info(f"{stage.value}_completed", **(metadata or {}))
    
    async def on_stage_error(self, stage: PipelineStage, error: Exception, metadata: Dict[str, Any] = None):
        logger.error(
            f"{stage.value}_error",
            error=str(error),
            error_type=type(error).__name__,
            **(metadata or {})
        )


class StreamingCallback(PipelineCallback):
    """Callback that streams events to a queue for SSE."""
    
    def __init__(self):
        self.event_queue: asyncio.Queue = asyncio.Queue()
    
    async def on_stage_start(self, stage: PipelineStage, metadata: Dict[str, Any] = None):
        event = StageEvent(
            stage=stage,
            status="started",
            message=f"{stage.value} started",
            metadata=metadata or {}
        )
        await self.event_queue.put(event)
    
    async def on_stage_progress(
        self,
        stage: PipelineStage,
        progress: float,
        message: str = "",
        metadata: Dict[str, Any] = None
    ):
        event = StageEvent(
            stage=stage,
            status="progress",
            message=message or f"{stage.value} in progress",
            metadata={**(metadata or {}), "progress": progress}
        )
        await self.event_queue.put(event)
    
    async def on_stage_complete(self, stage: PipelineStage, metadata: Dict[str, Any] = None):
        event = StageEvent(
            stage=stage,
            status="completed",
            message=f"{stage.value} completed",
            metadata=metadata or {}
        )
        await self.event_queue.put(event)
    
    async def on_stage_error(self, stage: PipelineStage, error: Exception, metadata: Dict[str, Any] = None):
        event = StageEvent(
            stage=PipelineStage.ERROR,
            status="error",
            message=str(error),
            metadata={**(metadata or {}), "error_type": type(error).__name__}
        )
        await self.event_queue.put(event)
    
    async def get_event(self) -> Optional[StageEvent]:
        """Get next event from queue."""
        try:
            return await asyncio.wait_for(self.event_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    def has_events(self) -> bool:
        """Check if there are pending events."""
        return not self.event_queue.empty()


class CallbackManager:
    """Manages multiple callbacks."""
    
    def __init__(self):
        self.callbacks: list[PipelineCallback] = []
        # Always add logging callback
        self.callbacks.append(LoggingCallback())
    
    def add_callback(self, callback: PipelineCallback):
        """Add a callback."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: PipelineCallback):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    async def on_stage_start(self, stage: PipelineStage, metadata: Dict[str, Any] = None):
        """Notify all callbacks of stage start."""
        await asyncio.gather(*[
            cb.on_stage_start(stage, metadata)
            for cb in self.callbacks
        ])
    
    async def on_stage_progress(
        self,
        stage: PipelineStage,
        progress: float,
        message: str = "",
        metadata: Dict[str, Any] = None
    ):
        """Notify all callbacks of stage progress."""
        await asyncio.gather(*[
            cb.on_stage_progress(stage, progress, message, metadata)
            for cb in self.callbacks
        ])
    
    async def on_stage_complete(self, stage: PipelineStage, metadata: Dict[str, Any] = None):
        """Notify all callbacks of stage completion."""
        await asyncio.gather(*[
            cb.on_stage_complete(stage, metadata)
            for cb in self.callbacks
        ])
    
    async def on_stage_error(self, stage: PipelineStage, error: Exception, metadata: Dict[str, Any] = None):
        """Notify all callbacks of stage error."""
        await asyncio.gather(*[
            cb.on_stage_error(stage, error, metadata)
            for cb in self.callbacks
        ])
