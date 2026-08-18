"""Structured-output schemas shared across the agent pipeline.

Keeping these in one place means every agent that needs the same shape
(e.g. Finding) imports the same class, instead of each module redefining
its own slightly-different version.
"""

from typing import List
from pydantic import BaseModel, Field


class Finding(BaseModel):
    """A single factual claim pulled from a source."""
    claim: str = Field(description="A concise factual claim or insight.")
    source_index: int = Field(
        description="The 1-based index of the source (from the provided source list) that supports this claim."
    )


class ResearchOutput(BaseModel):
    """Structured research findings for a single sub-question."""
    subtopic: str = Field(description="The sub-question being researched.")
    findings: List[Finding] = Field(description="4-6 concise factual findings, each tied to a source index.")


class FactCheckVerdict(BaseModel):
    """Verdict on whether one finding is actually supported by its cited source."""
    finding_number: int = Field(description="The 1-based position of the finding in the list being checked.")
    supported: bool = Field(description="True if the source text actually supports this claim, False if not or if unclear.")
    reasoning: str = Field(description="One short sentence explaining the verdict.")


class FactCheckOutput(BaseModel):
    """Verdicts for every finding passed in."""
    verdicts: List[FactCheckVerdict]


class RelevanceJudgement(BaseModel):
    """An independent judge's rating of how well a report addresses its topic."""
    score: int = Field(description="1 (irrelevant/incomplete) to 5 (thorough and directly on-topic).")
    reasoning: str = Field(description="One short sentence explaining the score.")