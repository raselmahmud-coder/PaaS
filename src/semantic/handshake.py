"""Semantic handshake protocol for inter-agent term negotiation."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from src.semantic.similarity import SemanticSimilarity, AlignmentResult
from src.semantic.terms import TermRegistry, Term

logger = logging.getLogger(__name__)


class HandshakeState(Enum):
    """States in the semantic handshake protocol."""
    
    IDLE = "idle"
    INIT_SENT = "init_sent"
    INIT_RECEIVED = "init_received"
    VERIFYING = "verifying"
    NEGOTIATING = "negotiating"
    WAITING_ACCEPT = "waiting_accept"
    COMPLETE = "complete"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class HandshakeMessage:
    """A message in the semantic handshake protocol."""
    
    message_type: str  # HANDSHAKE_INIT, HANDSHAKE_VERIFY, etc.
    sender: str
    receiver: str
    session_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_type": self.message_type,
            "sender": self.sender,
            "receiver": self.receiver,
            "session_id": self.session_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandshakeMessage":
        """Create from dictionary."""
        return cls(
            message_type=data["message_type"],
            sender=data["sender"],
            receiver=data["receiver"],
            session_id=data["session_id"],
            payload=data.get("payload", {}),
        )


@dataclass
class NegotiatedTerm:
    """A term that was negotiated during handshake."""
    
    original_term: str
    original_definition_a: str
    original_definition_b: str
    agreed_definition: str
    negotiated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HandshakeResult:
    """Result of a completed handshake."""
    
    session_id: str
    agent_a: str
    agent_b: str
    success: bool
    state: HandshakeState
    agreed_terms: Dict[str, str] = field(default_factory=dict)
    negotiated_terms: List[NegotiatedTerm] = field(default_factory=list)
    conflicts_resolved: int = 0
    total_terms: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "success": self.success,
            "state": self.state.value,
            "agreed_terms": self.agreed_terms,
            "negotiated_terms": [
                {
                    "term": nt.original_term,
                    "agreed_definition": nt.agreed_definition,
                }
                for nt in self.negotiated_terms
            ],
            "conflicts_resolved": self.conflicts_resolved,
            "total_terms": self.total_terms,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class HandshakeSession:
    """A single handshake session between two agents.
    
    Implements the 5-step semantic handshake protocol:
    1. HANDSHAKE_INIT - Agent A sends terms to Agent B
    2. HANDSHAKE_VERIFY - Agent B checks alignment, reports conflicts
    3. NEGOTIATE_TERM - Agent A proposes new definition for conflicting term
    4. TERM_ACCEPTED - Agent B accepts/rejects the proposal
    5. HANDSHAKE_COMPLETE - Both agents have agreed on terms
    """
    
    def __init__(
        self,
        session_id: str,
        initiator: str,
        responder: str,
        initiator_terms: TermRegistry,
        similarity_threshold: float = 0.85,
        negotiator: Optional[Any] = None,  # TermNegotiator
    ):
        """Initialize a handshake session.
        
        Args:
            session_id: Unique session identifier.
            initiator: Agent ID of the initiator (Agent A).
            responder: Agent ID of the responder (Agent B).
            initiator_terms: Term registry of the initiator.
            similarity_threshold: Threshold for semantic alignment.
            negotiator: Optional TermNegotiator for LLM-based negotiation.
        """
        self.session_id = session_id
        self.initiator = initiator
        self.responder = responder
        self.initiator_terms = initiator_terms
        self.similarity_threshold = similarity_threshold
        self.negotiator = negotiator
        
        self.state = HandshakeState.IDLE
        self.responder_terms: Optional[TermRegistry] = None
        
        # Tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.messages: List[HandshakeMessage] = []
        self.conflicts: List[str] = []
        self.pending_negotiations: List[str] = []
        self.agreed_terms: Dict[str, str] = {}
        self.negotiated_terms: List[NegotiatedTerm] = []
        self.alignment_result: Optional[AlignmentResult] = None
        
        self._similarity = SemanticSimilarity(threshold=similarity_threshold)
    
    # =========================================================================
    # Protocol Steps - Initiator Side (Agent A)
    # =========================================================================
    
    def create_init_message(self) -> HandshakeMessage:
        """Step 1: Create HANDSHAKE_INIT message.
        
        Called by the initiator to start the handshake.
        """
        self.start_time = datetime.utcnow()
        self.state = HandshakeState.INIT_SENT
        
        terms_data = {
            term.name: term.definition
            for term in self.initiator_terms.get_all_terms()
        }
        
        message = HandshakeMessage(
            message_type="HANDSHAKE_INIT",
            sender=self.initiator,
            receiver=self.responder,
            session_id=self.session_id,
            payload={
                "terms": terms_data,
                "threshold": self.similarity_threshold,
            },
        )
        
        self.messages.append(message)
        logger.info(f"[{self.session_id}] Handshake initiated with {len(terms_data)} terms")
        return message
    
    def create_negotiate_message(
        self,
        term: str,
        proposed_definition: str,
    ) -> HandshakeMessage:
        """Step 3: Create NEGOTIATE_TERM message.
        
        Called by the initiator to propose a new definition.
        """
        self.state = HandshakeState.WAITING_ACCEPT
        
        message = HandshakeMessage(
            message_type="NEGOTIATE_TERM",
            sender=self.initiator,
            receiver=self.responder,
            session_id=self.session_id,
            payload={
                "term": term,
                "proposed_definition": proposed_definition,
                "original_definition": self.initiator_terms.get_definitions().get(term, ""),
            },
        )
        
        self.messages.append(message)
        logger.info(f"[{self.session_id}] Negotiating term: {term}")
        return message
    
    def create_complete_message(self) -> HandshakeMessage:
        """Step 5: Create HANDSHAKE_COMPLETE message.
        
        Called when all terms are agreed upon.
        """
        self.state = HandshakeState.COMPLETE
        self.end_time = datetime.utcnow()
        
        message = HandshakeMessage(
            message_type="HANDSHAKE_COMPLETE",
            sender=self.initiator,
            receiver=self.responder,
            session_id=self.session_id,
            payload={
                "agreed_terms": self.agreed_terms,
                "negotiated_count": len(self.negotiated_terms),
            },
        )
        
        self.messages.append(message)
        logger.info(f"[{self.session_id}] Handshake completed successfully")
        return message
    
    # =========================================================================
    # Protocol Steps - Responder Side (Agent B)
    # =========================================================================
    
    def handle_init(
        self,
        message: HandshakeMessage,
        responder_terms: TermRegistry,
    ) -> HandshakeMessage:
        """Step 2: Handle HANDSHAKE_INIT and create HANDSHAKE_VERIFY.
        
        Called by the responder to verify term alignment.
        """
        self.state = HandshakeState.VERIFYING
        self.responder_terms = responder_terms
        
        # Extract initiator terms
        init_terms = message.payload.get("terms", {})
        
        # Check alignment
        self.alignment_result = self._similarity.check_alignment(
            terms_a=list(init_terms.keys()),
            terms_b=responder_terms.get_term_names(),
        )
        
        self.conflicts = self.alignment_result.conflicts
        self.pending_negotiations = list(self.conflicts)
        
        # Build agreed terms (non-conflicting)
        for term in init_terms.keys():
            if term not in self.conflicts:
                self.agreed_terms[term] = init_terms[term]
        
        # Create VERIFY response
        response = HandshakeMessage(
            message_type="HANDSHAKE_VERIFY",
            sender=self.responder,
            receiver=self.initiator,
            session_id=self.session_id,
            payload={
                "conflicts": self.conflicts,
                "similarities": self.alignment_result.similarity_scores,
                "aligned": self.alignment_result.aligned,
            },
        )
        
        self.messages.append(response)
        
        if self.alignment_result.aligned:
            logger.info(f"[{self.session_id}] All terms aligned, no negotiation needed")
        else:
            logger.info(f"[{self.session_id}] Found {len(self.conflicts)} conflicts: {self.conflicts}")
        
        return response
    
    def handle_negotiate(
        self,
        message: HandshakeMessage,
        accept: bool = True,
    ) -> HandshakeMessage:
        """Step 4: Handle NEGOTIATE_TERM and create TERM_ACCEPTED/REJECTED.
        
        Called by the responder to accept or reject a proposed definition.
        """
        term = message.payload.get("term", "")
        proposed_def = message.payload.get("proposed_definition", "")
        
        if accept and term in self.pending_negotiations:
            # Accept the proposal
            self.pending_negotiations.remove(term)
            self.agreed_terms[term] = proposed_def
            
            # Record negotiation
            original_def_a = message.payload.get("original_definition", "")
            original_def_b = ""
            if self.responder_terms:
                responder_term = self.responder_terms.get_term(term)
                if responder_term:
                    original_def_b = responder_term.definition
            
            self.negotiated_terms.append(NegotiatedTerm(
                original_term=term,
                original_definition_a=original_def_a,
                original_definition_b=original_def_b,
                agreed_definition=proposed_def,
            ))
        
        response = HandshakeMessage(
            message_type="TERM_ACCEPTED" if accept else "TERM_REJECTED",
            sender=self.responder,
            receiver=self.initiator,
            session_id=self.session_id,
            payload={
                "term": term,
                "accepted": accept,
                "remaining_conflicts": self.pending_negotiations,
            },
        )
        
        self.messages.append(response)
        logger.info(f"[{self.session_id}] Term '{term}' {'accepted' if accept else 'rejected'}")
        
        return response
    
    # =========================================================================
    # Results
    # =========================================================================
    
    def get_result(self) -> HandshakeResult:
        """Get the handshake result."""
        duration = 0.0
        if self.start_time:
            end = self.end_time or datetime.utcnow()
            duration = (end - self.start_time).total_seconds() * 1000
        
        return HandshakeResult(
            session_id=self.session_id,
            agent_a=self.initiator,
            agent_b=self.responder,
            success=self.state == HandshakeState.COMPLETE,
            state=self.state,
            agreed_terms=self.agreed_terms,
            negotiated_terms=self.negotiated_terms,
            conflicts_resolved=len(self.negotiated_terms),
            total_terms=len(self.agreed_terms),
            duration_ms=duration,
        )


class HandshakeManager:
    """Manager for handling multiple handshake sessions.
    
    Provides high-level API for running handshakes between agents.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        negotiator: Optional[Any] = None,
    ):
        """Initialize the handshake manager.
        
        Args:
            similarity_threshold: Default similarity threshold.
            negotiator: Optional TermNegotiator for LLM negotiation.
        """
        self.similarity_threshold = similarity_threshold
        self.negotiator = negotiator
        self._sessions: Dict[str, HandshakeSession] = {}
    
    def create_session(
        self,
        initiator: str,
        responder: str,
        initiator_terms: TermRegistry,
    ) -> HandshakeSession:
        """Create a new handshake session.
        
        Args:
            initiator: Agent ID of the initiator.
            responder: Agent ID of the responder.
            initiator_terms: Term registry of the initiator.
            
        Returns:
            New HandshakeSession instance.
        """
        session_id = str(uuid.uuid4())[:8]
        
        session = HandshakeSession(
            session_id=session_id,
            initiator=initiator,
            responder=responder,
            initiator_terms=initiator_terms,
            similarity_threshold=self.similarity_threshold,
            negotiator=self.negotiator,
        )
        
        self._sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[HandshakeSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    async def run_handshake(
        self,
        initiator: str,
        responder: str,
        initiator_terms: TermRegistry,
        responder_terms: TermRegistry,
        timeout: float = 30.0,
    ) -> HandshakeResult:
        """Run a complete handshake between two agents.
        
        This is a convenience method that runs the full 5-step protocol
        synchronously (simulated async for agent messaging).
        
        Args:
            initiator: Agent ID of the initiator.
            responder: Agent ID of the responder.
            initiator_terms: Term registry of the initiator.
            responder_terms: Term registry of the responder.
            timeout: Maximum time for handshake (seconds).
            
        Returns:
            HandshakeResult with outcome.
        """
        session = self.create_session(initiator, responder, initiator_terms)
        
        try:
            # Step 1: HANDSHAKE_INIT
            init_msg = session.create_init_message()
            
            # Step 2: HANDSHAKE_VERIFY
            verify_msg = session.handle_init(init_msg, responder_terms)
            
            # Check if already aligned
            if verify_msg.payload.get("aligned", False):
                # Step 5: HANDSHAKE_COMPLETE (no negotiation needed)
                session.create_complete_message()
                return session.get_result()
            
            # Steps 3-4: Negotiate each conflict
            conflicts = verify_msg.payload.get("conflicts", [])
            
            for term in conflicts:
                # Get definitions
                init_def = initiator_terms.get_definitions().get(term, "")
                resp_def = responder_terms.get_definitions().get(term, "")
                
                # Negotiate using LLM if available
                if self.negotiator:
                    proposed = await self.negotiator.negotiate_term(
                        term=term,
                        definition_a=init_def,
                        definition_b=resp_def,
                        context_a=f"Agent {initiator}",
                        context_b=f"Agent {responder}",
                    )
                    proposed_def = proposed.agreed_definition
                else:
                    # Simple fallback: use initiator's definition
                    proposed_def = f"{init_def} (aligned with: {resp_def})"
                
                # Step 3: NEGOTIATE_TERM
                negotiate_msg = session.create_negotiate_message(term, proposed_def)
                
                # Step 4: TERM_ACCEPTED
                session.handle_negotiate(negotiate_msg, accept=True)
            
            # Step 5: HANDSHAKE_COMPLETE
            session.create_complete_message()
            
        except asyncio.TimeoutError:
            session.state = HandshakeState.TIMEOUT
            logger.error(f"[{session.session_id}] Handshake timed out")
        except Exception as e:
            session.state = HandshakeState.FAILED
            logger.error(f"[{session.session_id}] Handshake failed: {e}")
        
        return session.get_result()
    
    def run_handshake_sync(
        self,
        initiator: str,
        responder: str,
        initiator_terms: TermRegistry,
        responder_terms: TermRegistry,
    ) -> HandshakeResult:
        """Synchronous version of run_handshake."""
        return asyncio.get_event_loop().run_until_complete(
            self.run_handshake(
                initiator, responder,
                initiator_terms, responder_terms,
            )
        )

