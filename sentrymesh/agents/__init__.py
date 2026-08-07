"""Specialist and supervisor agents."""

from sentrymesh.agents.historical_case_agent import HistoricalCaseAgent
from sentrymesh.agents.identity_signal_agent import IdentitySignalAgent
from sentrymesh.agents.network_analysis_agent import NetworkAnalysisAgent
from sentrymesh.agents.transaction_pattern_agent import TransactionPatternAgent

SPECIALIST_CLASSES = (
    TransactionPatternAgent,
    IdentitySignalAgent,
    NetworkAnalysisAgent,
    HistoricalCaseAgent,
)

__all__ = [
    "TransactionPatternAgent",
    "IdentitySignalAgent",
    "NetworkAnalysisAgent",
    "HistoricalCaseAgent",
    "SPECIALIST_CLASSES",
]
