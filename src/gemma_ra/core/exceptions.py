class GemmaRAError(Exception):
    """Base exception for the project."""


class ConfigurationError(GemmaRAError):
    """Raised when configuration or skills are invalid."""


class SourceError(GemmaRAError):
    """Raised when paper discovery or ingestion fails."""


class ModelError(GemmaRAError):
    """Raised when the model backend fails."""

