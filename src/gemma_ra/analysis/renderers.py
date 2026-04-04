from __future__ import annotations

from typing import Any

from gemma_ra.analysis.engine import TaskResult
from gemma_ra.core.schemas import TaskType


def render_analysis_result(result: TaskResult) -> str:
    lines = [f"# {result.title}", "", f"Task: `{result.task.value}`", ""]
    content = result.content
    if result.task == TaskType.ANALYZE_PAPER:
        lines.extend(_section("Problem", content["problem"]))
        lines.extend(_section("Inputs", _render_list(content["inputs"])))
        lines.extend(_section("Processing", content["processing"]))
        lines.extend(_section("Outputs", _render_list(content["outputs"])))
        lines.extend(_section("Key Ideas", _render_list(content["key_ideas"])))
        lines.extend(_section("Contributions", _render_list(content["contributions"])))
        lines.extend(_section("Limitations", _render_list(content["limitations"])))
    elif result.task == TaskType.REVIEW_TOPIC:
        lines.extend(_section("Field Overview", content["field_overview"]))
        lines.extend(_section("Key Problems", _render_list(content["key_problems"])))
        for item in content["paper_summaries"]:
            body = (
                f"Problem: {item['problem']}\n\n"
                f"Method: {item['method']}\n\n"
                f"Contributions: {', '.join(item['contributions'])}\n\n"
                f"Limitations: {', '.join(item['limitations'])}"
            )
            lines.extend(_section(f"Paper: {item['paper']['title']}", body))
        lines.extend(_section("Themes", _render_list(content["themes"])))
        lines.extend(_section("Methodological Trends", _render_list(content["methodological_trends"])))
        lines.extend(_section("Research Gaps", _render_list(content["gaps"])))
        lines.extend(_section("Disagreements", _render_list(content["disagreements"])))
        lines.extend(_section("Synthesis", content["synthesis"]))
    elif result.task == TaskType.FIND_PAPERS:
        lines.extend(_section("Topic", content["topic"]))
        if content.get("note"):
            lines.extend(_section("Note", content["note"]))
        if content.get("discovery_notes"):
            lines.extend(_section("Search Trace", _render_list(content["discovery_notes"])))
        for item in content["results"]:
            body = (
                f"Authors: {', '.join(item['authors'])}\n\n"
                f"Problem: {item['problem']}\n\n"
                f"Method: {item['method']}\n\n"
                f"Contributions: {', '.join(item['contributions'])}\n\n"
                f"Why relevant: {item['why_relevant']}"
            )
            lines.extend(_section(item["title"], body))
    elif result.task == TaskType.GENERATE_IDEAS:
        lines.extend(_section("Topic", content["topic"]))
        for idea in content["ideas"]:
            body = (
                f"Problem targeted: {idea['problem_targeted']}\n\n"
                f"Motivation: {idea['motivation']}\n\n"
                f"Novelty: {idea['novelty_rationale']}\n\n"
                f"Grounding: {idea['grounding']}\n\n"
                f"Expected contribution: {idea['expected_contribution']}\n\n"
                f"Method: {idea['proposed_method']}\n\n"
                f"Related papers: {', '.join(idea['related_papers'])}\n\n"
                f"Risks: {', '.join(idea['risks'])}"
            )
            lines.extend(_section(idea["title"], body))
    elif result.task == TaskType.SUGGEST_EXPERIMENTS:
        lines.extend(_section("Topic", content["topic"]))
        for experiment in content["experiments"]:
            body = (
                f"Related papers: {', '.join(experiment['related_papers'])}\n\n"
                f"Baseline: {experiment['baseline']}\n\n"
                f"Hypothesis: {experiment['hypothesis']}\n\n"
                f"Setup: {experiment['setup']}\n\n"
                f"Metrics: {', '.join(experiment['metrics'])}\n\n"
                f"Expected signal: {experiment['expected_signal']}\n\n"
                f"Why this matters: {experiment['why_this_matters']}\n\n"
                f"Failure conditions: {', '.join(experiment['failure_conditions'])}"
            )
            lines.extend(_section(experiment["title"], body))
    elif result.task == TaskType.MAP_RESEARCH_OPPORTUNITIES:
        lines.extend(_section("Topic", content["topic"]))
        lines.extend(_section("Suggested Professors", _render_list(content["suggested_professors"])))
        lines.extend(_section("Co-Author Leads", _render_list(content["coauthor_leads"])))
        lines.extend(_section("Field Summary", content["field_summary"]))
        lines.extend(_section("Hot Topics", _render_list(content["hot_topics"])))
        lines.extend(_section("Open Problems", _render_list(content["open_problems"])))
        for idea in content["opportunities"]:
            body = (
                f"Problem targeted: {idea['problem_targeted']}\n\n"
                f"Motivation: {idea['motivation']}\n\n"
                f"Novelty: {idea['novelty_rationale']}\n\n"
                f"Grounding: {idea['grounding']}\n\n"
                f"Expected contribution: {idea['expected_contribution']}\n\n"
                f"Method: {idea['proposed_method']}\n\n"
                f"Related papers: {', '.join(idea['related_papers'])}\n\n"
                f"Risks: {', '.join(idea['risks'])}"
            )
            lines.extend(_section(f"Opportunity: {idea['title']}", body))
        for experiment in content["experiments"]:
            body = (
                f"Related papers: {', '.join(experiment['related_papers'])}\n\n"
                f"Baseline: {experiment['baseline']}\n\n"
                f"Hypothesis: {experiment['hypothesis']}\n\n"
                f"Setup: {experiment['setup']}\n\n"
                f"Metrics: {', '.join(experiment['metrics'])}\n\n"
                f"Expected signal: {experiment['expected_signal']}\n\n"
                f"Why this matters: {experiment['why_this_matters']}\n\n"
                f"Failure conditions: {', '.join(experiment['failure_conditions'])}"
            )
            lines.extend(_section(f"Experiment: {experiment['title']}", body))
        lines.extend(_section("Search Strategy", _render_list(content["search_strategy"])))
    elif result.task == TaskType.RUN_INSTRUCTIONS:
        lines.extend(_section("Request", content["request"]))
        lines.extend(_section("Summary", content["summary"]))
        lines.extend(_section("Actions Taken", _render_list(content["actions_taken"])))
        lines.extend(_section("Key Findings", _render_list(content["key_findings"])))
        lines.extend(_section("Hot Topics", _render_list(content["hot_topics"])))
        for idea in content["research_opportunities"]:
            body = (
                f"Problem targeted: {idea['problem_targeted']}\n\n"
                f"Motivation: {idea['motivation']}\n\n"
                f"Novelty: {idea['novelty_rationale']}\n\n"
                f"Grounding: {idea['grounding']}\n\n"
                f"Expected contribution: {idea['expected_contribution']}\n\n"
                f"Method: {idea['proposed_method']}\n\n"
                f"Related papers: {', '.join(idea['related_papers'])}\n\n"
                f"Risks: {', '.join(idea['risks'])}"
            )
            lines.extend(_section(f"Opportunity: {idea['title']}", body))
        for experiment in content["experiment_suggestions"]:
            body = (
                f"Related papers: {', '.join(experiment['related_papers'])}\n\n"
                f"Baseline: {experiment['baseline']}\n\n"
                f"Hypothesis: {experiment['hypothesis']}\n\n"
                f"Setup: {experiment['setup']}\n\n"
                f"Metrics: {', '.join(experiment['metrics'])}\n\n"
                f"Expected signal: {experiment['expected_signal']}\n\n"
                f"Why this matters: {experiment['why_this_matters']}\n\n"
                f"Failure conditions: {', '.join(experiment['failure_conditions'])}"
            )
            lines.extend(_section(f"Experiment: {experiment['title']}", body))
        lines.extend(_section("Next Steps", _render_list(content["next_steps"])))
    return "\n".join(lines).strip() + "\n"


def _section(title: str, body: str) -> list[str]:
    return [f"## {title}", "", body, ""]


def _render_list(items: list[Any]) -> str:
    return "\n".join(f"- {item}" for item in items)
