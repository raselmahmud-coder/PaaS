"""Custom exception types for chaos engineering."""


class ChaosException(Exception):
    """Base exception for all chaos-related errors."""
    
    def __init__(self, message: str, fault_type: str = "unknown"):
        super().__init__(message)
        self.fault_type = fault_type
        self.message = message


class AgentCrashException(ChaosException):
    """Exception raised when an agent crash is simulated."""
    
    def __init__(self, message: str = "Simulated agent crash"):
        super().__init__(message, fault_type="crash")


class AgentTimeoutException(ChaosException):
    """Exception raised when an agent timeout is simulated."""
    
    def __init__(self, message: str = "Simulated agent timeout", timeout_seconds: float = 0):
        super().__init__(message, fault_type="timeout")
        self.timeout_seconds = timeout_seconds


class MessageCorruptionException(ChaosException):
    """Exception raised when message corruption is detected."""
    
    def __init__(self, message: str = "Simulated message corruption", corrupted_field: str = ""):
        super().__init__(message, fault_type="corruption")
        self.corrupted_field = corrupted_field


class HallucinationException(ChaosException):
    """Exception raised when LLM hallucination is simulated."""
    
    def __init__(self, message: str = "Simulated LLM hallucination"):
        super().__init__(message, fault_type="hallucination")

