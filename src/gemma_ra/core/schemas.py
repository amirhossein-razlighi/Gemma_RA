from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl


class TaskType(str, Enum):
    ANALYZE_PAPER = "analyze-paper"
    REVIEW_TOPIC = "review-topic"
    FIND_PAPERS = "find-papers"
    GENERATE_IDEAS = "generate-ideas"
    SUGGEST_EXPERIMENTS = "suggest-experiments"
    MAP_RESEARCH_OPPORTUNITIES = "map-research-opportunities"
    RUN_INSTRUCTIONS = "run-instructions"


class ToolKind(str, Enum):
    SOURCE = "source"
    ANALYSIS = "analysis"
    FUTURE = "future"


class ToolSpec(BaseModel):
    name: str
    kind: ToolKind
    description: str
    entrypoint: str


class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    published: datetime | None = None
    updated: datetime | None = None
    pdf_url: HttpUrl | None = None
    local_path: Path | None = None
    source: str


class PaperDocument(BaseModel):
    metadata: PaperMetadata
    content: str
    sections: list[str] = Field(default_factory=list)


class PaperAnalysis(BaseModel):
    paper: PaperMetadata
    problem: str
    inputs: list[str]
    processing: str
    outputs: list[str]
    key_ideas: list[str]
    contributions: list[str]
    limitations: list[str]


class LiteratureReview(BaseModel):
    topic: str
    papers: list[PaperMetadata]
    themes: list[str]
    gaps: list[str]
    disagreements: list[str]
    synthesis: str


class ResearchIdea(BaseModel):
    title: str
    motivation: str
    novelty_rationale: str
    related_papers: list[str]
    proposed_method: str
    risks: list[str]


class ExperimentSuggestion(BaseModel):
    title: str
    hypothesis: str
    setup: str
    metrics: list[str]
    expected_signal: str
    failure_conditions: list[str]


class ResearchOpportunityMap(BaseModel):
    topic: str
    suggested_professors: list[str]
    coauthor_leads: list[str]
    papers: list[PaperMetadata]
    field_summary: str
    hot_topics: list[str]
    open_problems: list[str]
    opportunities: list[ResearchIdea]
    experiments: list[ExperimentSuggestion]
    search_strategy: list[str]


class InstructionRunResult(BaseModel):
    request: str
    summary: str
    actions_taken: list[str]
    papers_considered: list[PaperMetadata]
    key_findings: list[str]
    hot_topics: list[str]
    research_opportunities: list[ResearchIdea]
    experiment_suggestions: list[ExperimentSuggestion]
    next_steps: list[str]


class ArtifactRecord(BaseModel):
    task: TaskType
    task_name: str
    generated_at: datetime
    markdown_path: Path
    json_path: Path


class ResearchContext(BaseModel):
    task: TaskType
    topic: str | None = None
    professors: list[str] = Field(default_factory=list)
    local_paper_paths: list[Path] = Field(default_factory=list)
    papers: list[PaperDocument] = Field(default_factory=list)
    discovery_notes: list[str] = Field(default_factory=list)
    instructions: str | None = None
    instructions_path: Path | None = None
