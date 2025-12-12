"""Semantic protocol message types for the handshake protocol."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import uuid


class SemanticMessageHeader(BaseModel):
    """Header for semantic protocol messages."""
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: Literal[
        "HANDSHAKE_INIT",
        "HANDSHAKE_VERIFY", 
        "NEGOTIATE_TERM",
        "TERM_ACCEPTED",
        "TERM_REJECTED",
        "HANDSHAKE_COMPLETE",
    ]
    sender: str
    receiver: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    protocol_version: str = "1.0"


class HandshakeInitPayload(BaseModel):
    """Payload for HANDSHAKE_INIT message."""
    
    terms: Dict[str, str]  # term_name -> definition
    threshold: float = 0.85
    context: str = ""


class HandshakeVerifyPayload(BaseModel):
    """Payload for HANDSHAKE_VERIFY message."""
    
    aligned: bool
    conflicts: List[str] = Field(default_factory=list)
    similarities: Dict[str, float] = Field(default_factory=dict)
    responder_terms: Dict[str, str] = Field(default_factory=dict)


class NegotiateTermPayload(BaseModel):
    """Payload for NEGOTIATE_TERM message."""
    
    term: str
    proposed_definition: str
    original_definition: str = ""
    reasoning: str = ""


class TermAcceptedPayload(BaseModel):
    """Payload for TERM_ACCEPTED/TERM_REJECTED message."""
    
    term: str
    accepted: bool
    remaining_conflicts: List[str] = Field(default_factory=list)
    counter_proposal: Optional[str] = None


class HandshakeCompletePayload(BaseModel):
    """Payload for HANDSHAKE_COMPLETE message."""
    
    agreed_terms: Dict[str, str]
    negotiated_count: int = 0
    total_terms: int = 0
    duration_ms: float = 0.0


# Message classes
class HandshakeInitMessage(BaseModel):
    """HANDSHAKE_INIT message - sent by initiator to start handshake."""
    
    header: SemanticMessageHeader
    payload: HandshakeInitPayload
    
    @classmethod
    def create(
        cls,
        sender: str,
        receiver: str,
        session_id: str,
        terms: Dict[str, str],
        threshold: float = 0.85,
    ) -> "HandshakeInitMessage":
        """Create a HANDSHAKE_INIT message."""
        return cls(
            header=SemanticMessageHeader(
                message_type="HANDSHAKE_INIT",
                sender=sender,
                receiver=receiver,
                session_id=session_id,
            ),
            payload=HandshakeInitPayload(
                terms=terms,
                threshold=threshold,
            ),
        )


class HandshakeVerifyMessage(BaseModel):
    """HANDSHAKE_VERIFY message - sent by responder with alignment results."""
    
    header: SemanticMessageHeader
    payload: HandshakeVerifyPayload
    
    @classmethod
    def create(
        cls,
        sender: str,
        receiver: str,
        session_id: str,
        aligned: bool,
        conflicts: List[str],
        similarities: Dict[str, float],
    ) -> "HandshakeVerifyMessage":
        """Create a HANDSHAKE_VERIFY message."""
        return cls(
            header=SemanticMessageHeader(
                message_type="HANDSHAKE_VERIFY",
                sender=sender,
                receiver=receiver,
                session_id=session_id,
            ),
            payload=HandshakeVerifyPayload(
                aligned=aligned,
                conflicts=conflicts,
                similarities=similarities,
            ),
        )


class NegotiateTermMessage(BaseModel):
    """NEGOTIATE_TERM message - sent by initiator to propose new definition."""
    
    header: SemanticMessageHeader
    payload: NegotiateTermPayload
    
    @classmethod
    def create(
        cls,
        sender: str,
        receiver: str,
        session_id: str,
        term: str,
        proposed_definition: str,
        original_definition: str = "",
    ) -> "NegotiateTermMessage":
        """Create a NEGOTIATE_TERM message."""
        return cls(
            header=SemanticMessageHeader(
                message_type="NEGOTIATE_TERM",
                sender=sender,
                receiver=receiver,
                session_id=session_id,
            ),
            payload=NegotiateTermPayload(
                term=term,
                proposed_definition=proposed_definition,
                original_definition=original_definition,
            ),
        )


class TermAcceptedMessage(BaseModel):
    """TERM_ACCEPTED message - sent by responder to accept proposal."""
    
    header: SemanticMessageHeader
    payload: TermAcceptedPayload
    
    @classmethod
    def create(
        cls,
        sender: str,
        receiver: str,
        session_id: str,
        term: str,
        accepted: bool,
        remaining_conflicts: Optional[List[str]] = None,
    ) -> "TermAcceptedMessage":
        """Create a TERM_ACCEPTED/REJECTED message."""
        return cls(
            header=SemanticMessageHeader(
                message_type="TERM_ACCEPTED" if accepted else "TERM_REJECTED",
                sender=sender,
                receiver=receiver,
                session_id=session_id,
            ),
            payload=TermAcceptedPayload(
                term=term,
                accepted=accepted,
                remaining_conflicts=remaining_conflicts or [],
            ),
        )


class HandshakeCompleteMessage(BaseModel):
    """HANDSHAKE_COMPLETE message - sent when handshake is complete."""
    
    header: SemanticMessageHeader
    payload: HandshakeCompletePayload
    
    @classmethod
    def create(
        cls,
        sender: str,
        receiver: str,
        session_id: str,
        agreed_terms: Dict[str, str],
        negotiated_count: int = 0,
    ) -> "HandshakeCompleteMessage":
        """Create a HANDSHAKE_COMPLETE message."""
        return cls(
            header=SemanticMessageHeader(
                message_type="HANDSHAKE_COMPLETE",
                sender=sender,
                receiver=receiver,
                session_id=session_id,
            ),
            payload=HandshakeCompletePayload(
                agreed_terms=agreed_terms,
                negotiated_count=negotiated_count,
                total_terms=len(agreed_terms),
            ),
        )


# Type for any semantic message
SemanticMessage = (
    HandshakeInitMessage
    | HandshakeVerifyMessage
    | NegotiateTermMessage
    | TermAcceptedMessage
    | HandshakeCompleteMessage
)


def parse_semantic_message(data: Dict[str, Any]) -> Optional[SemanticMessage]:
    """Parse a semantic message from dictionary.
    
    Args:
        data: Message dictionary with header and payload.
        
    Returns:
        Parsed message or None if invalid.
    """
    try:
        header = data.get("header", {})
        msg_type = header.get("message_type")
        
        if msg_type == "HANDSHAKE_INIT":
            return HandshakeInitMessage(**data)
        elif msg_type == "HANDSHAKE_VERIFY":
            return HandshakeVerifyMessage(**data)
        elif msg_type == "NEGOTIATE_TERM":
            return NegotiateTermMessage(**data)
        elif msg_type in ("TERM_ACCEPTED", "TERM_REJECTED"):
            return TermAcceptedMessage(**data)
        elif msg_type == "HANDSHAKE_COMPLETE":
            return HandshakeCompleteMessage(**data)
        else:
            return None
    except Exception:
        return None

