"""
Observability utilities for structured logging and metrics.
"""
import structlog
import time
from typing import Any, Optional
from contextlib import contextmanager
from functools import wraps
from src.config import settings


def setup_logging():
    """Configure structured logging based on settings."""
    import logging
    
    # Map string log level to logging constant
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    log_level = log_level_map.get(settings.log_level.upper(), logging.INFO)
    
    if settings.log_format == "json":
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance with the given name."""
    return structlog.get_logger(name)


@contextmanager
def log_duration(logger: structlog.BoundLogger, operation: str, **context):
    """Context manager to log operation duration."""
    start_time = time.time()
    logger.info(f"{operation}_started", **context)
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(
            f"{operation}_completed",
            duration_seconds=round(duration, 3),
            **context
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"{operation}_failed",
            duration_seconds=round(duration, 3),
            error=str(e),
            error_type=type(e).__name__,
            **context
        )
        raise


def track_metrics(operation: str):
    """Decorator to track function execution metrics."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"{operation}_completed",
                    function=func.__name__,
                    duration_seconds=round(duration, 3)
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{operation}_failed",
                    function=func.__name__,
                    duration_seconds=round(duration, 3),
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"{operation}_completed",
                    function=func.__name__,
                    duration_seconds=round(duration, 3)
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{operation}_failed",
                    function=func.__name__,
                    duration_seconds=round(duration, 3),
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class MetricsCollector:
    """Collect and aggregate metrics for pipeline stages."""
    
    def __init__(self):
        self.metrics: dict[str, list[float]] = {}
    
    def record(self, metric_name: str, value: float):
        """Record a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_stats(self, metric_name: str) -> Optional[dict[str, float]]:
        """Get statistics for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = self.metrics[metric_name]
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values)
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()


# Global metrics collector
metrics_collector = MetricsCollector()
