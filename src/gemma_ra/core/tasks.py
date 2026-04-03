from __future__ import annotations

from dataclasses import dataclass, field

from gemma_ra.core.schemas import TaskType


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_type: TaskType
    purpose: str
    allowed_tools: list[str]
    output_sections: list[str]
    prompt_template: str
    constraints: list[str] = field(default_factory=list)


TASK_SPECS: dict[TaskType, TaskSpec] = {
    TaskType.ANALYZE_PAPER: TaskSpec(
        task_type=TaskType.ANALYZE_PAPER,
        purpose="Extract the core research structure from a paper.",
        allowed_tools=["local_pdf_reader", "analysis_engine"],
        output_sections=[
            "Problem",
            "Inputs",
            "Processing",
            "Outputs",
            "Key Ideas",
            "Contributions",
            "Limitations",
        ],
        constraints=[
            "Stay faithful to the provided paper content.",
            "Separate direct claims from inferred interpretation.",
        ],
        prompt_template=(
            "You are a careful research assistant.\n"
            "Analyze the provided paper and return valid JSON matching the requested schema.\n\n"
            "Focus sections: {output_sections}\n"
            "Constraints:\n{constraints}\n\n"
            "Paper context:\n{papers}\n"
        ),
    ),
    TaskType.REVIEW_TOPIC: TaskSpec(
        task_type=TaskType.REVIEW_TOPIC,
        purpose="Synthesize themes and research gaps across multiple papers.",
        allowed_tools=["local_pdf_reader", "arxiv_search", "analysis_engine"],
        output_sections=["Themes", "Research Gaps", "Disagreements", "Synthesis"],
        constraints=[
            "Highlight concrete disagreements or tension between papers when present.",
            "Focus on research-relevant synthesis rather than broad survey prose.",
        ],
        prompt_template=(
            "You are writing a structured literature review.\n"
            "Return valid JSON matching the schema.\n\n"
            "Topic: {topic}\n"
            "Professors: {professors}\n"
            "Output sections: {output_sections}\n"
            "Constraints:\n{constraints}\n\n"
            "Papers:\n{papers}\n"
        ),
    ),
    TaskType.FIND_PAPERS: TaskSpec(
        task_type=TaskType.FIND_PAPERS,
        purpose="Rank and explain recent relevant papers from arXiv search results.",
        allowed_tools=["arxiv_search", "analysis_engine"],
        output_sections=["Topic", "Results"],
        constraints=[
            "Use only the supplied paper metadata and abstracts.",
            "Prefer recency and relevance to the stated topic.",
        ],
        prompt_template=(
            "You are triaging arXiv search results for a research assistant workflow.\n"
            "Return valid JSON matching the schema.\n\n"
            "Topic: {topic}\n"
            "Professors: {professors}\n"
            "Output sections: {output_sections}\n"
            "Constraints:\n{constraints}\n\n"
            "Search results:\n{papers}\n"
        ),
    ),
    TaskType.GENERATE_IDEAS: TaskSpec(
        task_type=TaskType.GENERATE_IDEAS,
        purpose="Generate plausible research directions grounded in the provided papers.",
        allowed_tools=["local_pdf_reader", "arxiv_search", "analysis_engine"],
        output_sections=["Topic", "Ideas"],
        constraints=[
            "Every idea should connect to at least one supplied paper.",
            "Explain why each idea may be novel or underexplored.",
        ],
        prompt_template=(
            "You are proposing grounded research directions based on a set of papers.\n"
            "Return valid JSON matching the schema.\n\n"
            "Topic: {topic}\n"
            "Professors: {professors}\n"
            "Constraints:\n{constraints}\n\n"
            "Source papers:\n{papers}\n"
        ),
    ),
    TaskType.SUGGEST_EXPERIMENTS: TaskSpec(
        task_type=TaskType.SUGGEST_EXPERIMENTS,
        purpose="Design lightweight experiments that help validate whether an idea is worth pursuing.",
        allowed_tools=["local_pdf_reader", "arxiv_search", "analysis_engine", "future_experiment_runner"],
        output_sections=["Topic", "Experiments"],
        constraints=[
            "Keep experiments lightweight and fast to evaluate.",
            "Prefer experiments that test feasibility before scaling.",
        ],
        prompt_template=(
            "You are designing small validation experiments for a research idea exploration workflow.\n"
            "Return valid JSON matching the schema.\n\n"
            "Topic: {topic}\n"
            "Professors: {professors}\n"
            "Constraints:\n{constraints}\n\n"
            "Source papers:\n{papers}\n"
        ),
    ),
    TaskType.MAP_RESEARCH_OPPORTUNITIES: TaskSpec(
        task_type=TaskType.MAP_RESEARCH_OPPORTUNITIES,
        purpose="Map a field from seed professors and co-authors, then propose novel opportunities and experiments.",
        allowed_tools=["local_pdf_reader", "arxiv_search", "analysis_engine", "future_experiment_runner"],
        output_sections=[
            "Suggested Professors",
            "Co-Author Leads",
            "Field Summary",
            "Hot Topics",
            "Open Problems",
            "Opportunities",
            "Experiments",
            "Search Strategy",
        ],
        constraints=[
            "If no professor is provided, suggest a few plausible seed professors for the topic before expanding.",
            "Use co-authors only when they appear repeatedly or seem central to the topic.",
            "Ground every opportunity in the supplied papers and discovered research threads.",
            "Prefer concrete, testable, near-term research directions over vague moonshots.",
        ],
        prompt_template=(
            "You are mapping a research field for a scientist looking for the next novel project.\n"
            "Return valid JSON matching the schema.\n\n"
            "Topic: {topic}\n"
            "Seed professors: {professors}\n"
            "Requested sections: {output_sections}\n"
            "Constraints:\n{constraints}\n\n"
            "Use the papers below to:\n"
            "1. identify good seed professors if needed,\n"
            "2. understand the field and hot topics,\n"
            "3. infer promising co-author leads,\n"
            "4. surface concrete open problems,\n"
            "5. propose novel but plausible research opportunities,\n"
            "6. suggest lightweight experiments to validate them.\n\n"
            "Papers:\n{papers}\n"
        ),
    ),
    TaskType.RUN_INSTRUCTIONS: TaskSpec(
        task_type=TaskType.RUN_INSTRUCTIONS,
        purpose="Read a free-form instruction file, choose tools, and produce a structured research artifact.",
        allowed_tools=["local_pdf_reader", "arxiv_search", "analysis_engine", "future_experiment_runner"],
        output_sections=[
            "Summary",
            "Actions Taken",
            "Key Findings",
            "Hot Topics",
            "Research Opportunities",
            "Experiment Suggestions",
            "Next Steps",
        ],
        constraints=[
            "Treat the instruction file as the source of truth for the user request.",
            "Use tools only when they materially help answer the instruction.",
            "Ground research opportunities and experiments in discovered papers or the provided idea.",
            "Be explicit about what you actually searched or loaded.",
            "If the instructions define success criteria, do not stop until they are satisfied or you have exhausted the iteration budget.",
            "If you decide something should be changed in a file, call a tool to do it instead of only describing the intended change.",
        ],
        prompt_template=(
            "You are a research assistant operating from a free-form instruction file.\n"
            "This is an action-oriented tool loop, not a planning-only conversation.\n"
            "When the instructions require file edits, experiments, reruns, or log inspection, you must call tools to perform those actions.\n"
            "Do not stop after describing a next step; stop only when the success criteria are met or you cannot make further progress.\n"
            "Return valid JSON matching the schema.\n\n"
            "Instruction file contents:\n{instructions}\n\n"
            "Topic: {topic}\n"
            "Known professors: {professors}\n"
            "Requested sections: {output_sections}\n"
            "Constraints:\n{constraints}\n\n"
            "Discovered papers and context:\n{papers}\n"
        ),
    ),
}


def get_task_spec(task: TaskType) -> TaskSpec:
    return TASK_SPECS[task]
