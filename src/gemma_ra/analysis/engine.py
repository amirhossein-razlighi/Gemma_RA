from __future__ import annotations

import ast
import json
from typing import Any

from pydantic import BaseModel

from gemma_ra.core.model_client import OllamaClient
from gemma_ra.core.schemas import (
    ExperimentSuggestion,
    InstructionRunResult,
    LiteratureReview,
    PaperAnalysis,
    PaperDigest,
    PaperDocument,
    ReviewPaperSummary,
    ResearchContext,
    ResearchIdea,
    ResearchOpportunityMap,
    TaskType,
)
from gemma_ra.core.tasks import TaskSpec
from gemma_ra.core.workspace import WorkspaceExecutor


class TaskResult(BaseModel):
    title: str
    task: TaskType
    content: dict[str, Any]


class AnalysisEngine:
    def __init__(
        self,
        model_client: OllamaClient,
        arxiv_search=None,
        arxiv_fetch=None,
        workspace: WorkspaceExecutor | None = None,
        max_iterations: int = 3,
        reporter=None,
        stream_chat: bool = False,
        interactive_guidance=None,
    ) -> None:
        self.model_client = model_client
        self.arxiv_search = arxiv_search
        self.arxiv_fetch = arxiv_fetch
        self.workspace = workspace
        self.max_iterations = max_iterations
        self.reporter = reporter
        self.stream_chat = stream_chat
        self.interactive_guidance = interactive_guidance

    def run(self, task: TaskType, task_spec: TaskSpec, context: ResearchContext) -> TaskResult:
        context = self._run_tool_loop(task=task, task_spec=task_spec, context=context)
        if task == TaskType.FIND_PAPERS and not context.papers:
            return TaskResult(
                title=context.topic or task.value,
                task=task,
                content={
                    "topic": context.topic or "Paper discovery",
                    "results": [],
                    "note": "No papers were found. Try a broader topic, a shorter professor name, or no topic filter.",
                    "discovery_notes": context.discovery_notes,
                },
            )
        if task == TaskType.MAP_RESEARCH_OPPORTUNITIES and not context.papers:
            return TaskResult(
                title=context.topic or task.value,
                task=task,
                content={
                    "topic": context.topic or "Research opportunity mapping",
                    "suggested_professors": context.professors,
                    "coauthor_leads": [],
                    "papers": [],
                    "field_summary": "No papers were found to map the field.",
                    "hot_topics": [],
                    "open_problems": [],
                    "opportunities": [],
                    "experiments": [],
                    "search_strategy": context.discovery_notes
                    + ["Try broader topic keywords or seed one known professor in the field."],
                },
            )
        if task == TaskType.RUN_INSTRUCTIONS and not context.papers and not context.instructions:
            return TaskResult(
                title="instructions",
                task=task,
                content={
                    "request": "",
                    "summary": "No instructions were provided.",
                    "actions_taken": [],
                    "papers_considered": [],
                    "key_findings": [],
                    "hot_topics": [],
                    "research_opportunities": [],
                    "experiment_suggestions": [],
                    "next_steps": ["Add content to INSTRUCTIONS.md and run the command again."],
                },
            )
        prompt = self._build_prompt(task_spec, context)
        schema = self._response_schema(task)
        raw = self.model_client.generate_structured(prompt=prompt, schema=schema)
        normalized = self._validate_response(task=task, raw=raw, context=context)
        title = context.topic or (context.papers[0].metadata.title if context.papers else task.value)
        return TaskResult(title=title, task=task, content=normalized)

    def _run_tool_loop(self, task: TaskType, task_spec: TaskSpec, context: ResearchContext) -> ResearchContext:
        tool_specs, tool_impl = self._build_runtime_tools(task_spec=task_spec, context=context)
        if not tool_specs:
            return context
        success_criteria = self._success_criteria_summary(context.instructions)
        latest_instruction_state = ""
        last_tool_signature: tuple[str, str] | None = None
        repeated_tool_calls = 0

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant in a tool loop. "
                    "Call tools when you need more papers or metadata. "
                    "You may inspect local paper folders, search arXiv by professor, and then synthesize an answer. "
                    "This tool loop is an internal gather phase for a non-interactive CLI run. "
                    "Do not ask the user follow-up questions, do not present final findings, and do not offer optional next steps here. "
                    "When you have enough context, reply exactly with READY and no tool calls. "
                    "If a paper is already loaded and the task is to analyze or review it, do not ask the user for clarification before reading the available paper content. "
                    "When reading paper text, prefer larger chunks so you can see method and experiment details in one pass; use around 20000-30000 characters unless you have a strong reason to read less. "
                    "For review, idea-generation, and opportunity-mapping tasks, work breadth-first across multiple papers instead of exhausting one paper before looking at the rest. "
                    "After you understand a paper well enough, save a compact digest before moving on."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {task.value}\n"
                    f"Topic: {context.topic or 'Not specified'}\n"
                    f"Professors: {', '.join(context.professors) or 'None'}\n"
                    f"Preloaded papers: {len(context.papers)}\n"
                    f"Local paper paths: {', '.join(str(path) for path in context.local_paper_paths) or 'None'}\n"
                    f"Instructions: {context.instructions or 'None'}\n"
                    f"Active success criteria: {success_criteria}"
                ),
            },
        ]

        for iteration_index in range(self.max_iterations):
            guidance_messages = self._drain_interactive_guidance()
            for guidance in guidance_messages:
                messages.append({"role": "user", "content": guidance})
                self._report("agent", f"received user guidance: {guidance}")
            self._report("agent", self._iteration_status(task, context, iteration_index + 1))
            try:
                response = self.model_client.chat(
                    messages=messages,
                    tools=tool_specs,
                    stream=self.stream_chat,
                    on_chunk=self._handle_stream_chunk,
                )
            except TypeError:
                response = self.model_client.chat(messages=messages, tools=tool_specs)
            message = response.get("message", {})
            messages.append(message)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                if self._should_continue_instruction_loop(task=task, message=message):
                    self._report("agent", "model described a next step without calling a tool; requesting an explicit action")
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You have not finished yet. "
                                f"Active success criteria: {success_criteria}. "
                                f"Latest observed state: {latest_instruction_state or 'No concrete state update yet.'} "
                                "If those success criteria are not met, call the next tool now. "
                                "Do not just describe the action or analyze the logs in prose. "
                                "Choose a concrete next action such as reading a file, editing code, changing JSON config, checking Python syntax, or rerunning the program. "
                                "Use existing workspace-relative paths only, such as '.' , 'config.json', 'train.py', or 'logs/latest.json'. "
                                "If a previous tool failed because of a bad path, correct the path before trying again. "
                                "If and only if the success criteria are already met, reply exactly with FINISHED."
                            ),
                        }
                    )
                    continue
                paper_action_prompt = self._paper_explicit_action_prompt(task=task, message=message, messages=messages)
                if paper_action_prompt:
                    self._report("agent", "model described a paper next step without a concrete tool; requesting an explicit paper action")
                    messages.append({"role": "user", "content": paper_action_prompt})
                    continue
                paper_followup = self._paper_followup_prompt(task=task, messages=messages)
                if paper_followup:
                    self._report("agent", "model stopped after a partial paper read; requesting an explicit next paper action")
                    messages.append({"role": "user", "content": paper_followup})
                    continue
                if task != TaskType.RUN_INSTRUCTIONS:
                    self._report("agent", "tool loop complete; proceeding to structured synthesis")
                else:
                    self._report("agent", "model stopped requesting tools")
                break
            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments", {})
                if name not in tool_impl:
                    continue
                self._report("tool", f"{name}({arguments})")
                try:
                    result = tool_impl[name](**arguments)
                except Exception as exc:  # noqa: BLE001
                    result = f"Tool error: {type(exc).__name__}: {exc}"
                self._report("tool_result", str(result)[:800])
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": result,
                    }
                )
                tool_signature = (name, json.dumps(arguments, sort_keys=True, default=str))
                if tool_signature == last_tool_signature:
                    repeated_tool_calls += 1
                else:
                    repeated_tool_calls = 1
                    last_tool_signature = tool_signature
                if task == TaskType.RUN_INSTRUCTIONS:
                    latest_instruction_state = self._instruction_state_update(
                        tool_name=name,
                        arguments=arguments,
                        result=result,
                    )
                    if latest_instruction_state:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Latest observed state: {latest_instruction_state} "
                                    "If success criteria are still not met, continue with a concrete tool call instead of a prose summary."
                                ),
                            }
                        )
                    next_action_prompt = self._next_action_instruction(
                        tool_name=name,
                        arguments=arguments,
                        result=result,
                    )
                    if next_action_prompt:
                        messages.append({"role": "user", "content": next_action_prompt})
                    anti_loop_prompt = self._anti_loop_instruction(
                        tool_name=name,
                        arguments=arguments,
                        repeat_count=repeated_tool_calls,
                    )
                    if anti_loop_prompt:
                        messages.append({"role": "user", "content": anti_loop_prompt})
                else:
                    paper_next_action_prompt = self._paper_next_action_instruction(
                        task=task,
                        tool_name=name,
                        result=result,
                    )
                    if paper_next_action_prompt:
                        messages.append({"role": "user", "content": paper_next_action_prompt})
                guidance_messages = self._drain_interactive_guidance()
                if guidance_messages:
                    for guidance in guidance_messages:
                        messages.append({"role": "user", "content": guidance})
                        self._report("agent", f"received user guidance: {guidance}")
                    break
        return context

    def _iteration_status(self, task: TaskType, context: ResearchContext, iteration: int) -> str:
        if task == TaskType.RUN_INSTRUCTIONS:
            instruction_state = "with instructions loaded" if context.instructions else "with no instructions"
            return f"iteration {iteration}/{self.max_iterations} in workspace mode, {instruction_state}"
        if task in {TaskType.ANALYZE_PAPER, TaskType.REVIEW_TOPIC, TaskType.FIND_PAPERS, TaskType.GENERATE_IDEAS, TaskType.SUGGEST_EXPERIMENTS, TaskType.MAP_RESEARCH_OPPORTUNITIES}:
            return f"iteration {iteration}/{self.max_iterations} with {len(context.papers)} loaded paper(s)"
        return f"iteration {iteration}/{self.max_iterations}"

    @staticmethod
    def _should_continue_instruction_loop(task: TaskType, message: dict[str, Any]) -> bool:
        if task != TaskType.RUN_INSTRUCTIONS:
            return False
        content = (message.get("content") or "").strip()
        thinking = (message.get("thinking") or "").strip()
        if content == "FINISHED":
            return False
        return bool(content or thinking)

    @staticmethod
    def _paper_explicit_action_prompt(task: TaskType, message: dict[str, Any], messages: list[dict[str, Any]]) -> str | None:
        if task not in {
            TaskType.ANALYZE_PAPER,
            TaskType.REVIEW_TOPIC,
            TaskType.FIND_PAPERS,
            TaskType.GENERATE_IDEAS,
            TaskType.SUGGEST_EXPERIMENTS,
            TaskType.MAP_RESEARCH_OPPORTUNITIES,
        }:
            return None
        content = (message.get("content") or "").strip()
        thinking = (message.get("thinking") or "").strip()
        if content == "READY":
            return None
        if not (content or thinking):
            return None
        for prior in reversed(messages[:-1]):
            if prior.get("role") != "tool":
                continue
            tool_name = prior.get("tool_name")
            payload = AnalysisEngine._parse_tool_payload(prior.get("content", ""))
            if tool_name == "read_loaded_paper_content" and isinstance(payload, dict):
                paper = payload.get("paper") if isinstance(payload.get("paper"), dict) else {}
                paper_id = paper.get("paper_id")
                title = paper.get("title")
                remaining = payload.get("remaining_chars")
                excerpt = payload.get("content", "")
                excerpt_block = ""
                if isinstance(excerpt, str) and excerpt.strip():
                    excerpt_block = (
                        "\nRecent paper excerpt from the immediately preceding tool result:\n"
                        f"{excerpt[:1500]}\n"
                    )
                if isinstance(remaining, int) and remaining > 0:
                    return (
                        "This is an internal tool loop, so do not narrate what you might do next. "
                        f'You are currently reading "{title or paper_id or "the current paper"}". '
                        "Call the next paper tool now. "
                        "Either continue the same paper with `read_loaded_paper_content(...)`, "
                        "or, if you already understand the paper well enough, save a compact digest with `save_paper_digest(...)` before moving on."
                    )
                return (
                    "This is an internal tool loop, so do not narrate what you might do next. "
                    f'You have finished reading "{title or paper_id or "the current paper"}". '
                    f"{excerpt_block}"
                    "Use the paper text from the immediately preceding tool result to save a compact digest right now with `save_paper_digest(...)`. "
                    "Do not claim that you lack context when you have just read the paper. "
                    "After saving the digest, fetch/read another relevant paper, inspect the loaded papers to choose the next one, or reply exactly with READY if you already have enough cross-paper context for synthesis."
                )
            if tool_name == "fetch_arxiv_full_text" and isinstance(payload, dict):
                paper_id = payload.get("paper_id")
                title = payload.get("title")
                return (
                    "This is an internal tool loop, so do not narrate what you might do next. "
                    f'You just fetched full text for "{title or paper_id or "a paper"}". '
                    "Call the next paper tool now: either read this paper with `read_loaded_paper_content(...)`, fetch another important paper, or reply exactly with READY if you already have enough cross-paper context."
                )
            break
        return (
            "This is an internal tool loop, so do not narrate what you might do next. "
            "Call a concrete paper-related tool now using the evidence already present in the recent tool results, or reply exactly with READY if you already have enough context."
        )

    @staticmethod
    def _paper_next_action_instruction(task: TaskType, tool_name: str, result: Any) -> str | None:
        if task not in {
            TaskType.ANALYZE_PAPER,
            TaskType.REVIEW_TOPIC,
            TaskType.FIND_PAPERS,
            TaskType.GENERATE_IDEAS,
            TaskType.SUGGEST_EXPERIMENTS,
            TaskType.MAP_RESEARCH_OPPORTUNITIES,
        }:
            return None
        if tool_name != "read_loaded_paper_content":
            return None
        payload = AnalysisEngine._parse_tool_payload(result)
        if not isinstance(payload, dict):
            return None
        remaining = payload.get("remaining_chars")
        paper = payload.get("paper") if isinstance(payload.get("paper"), dict) else {}
        paper_id = paper.get("paper_id")
        title = paper.get("title")
        current_offset = payload.get("offset")
        returned_chars = payload.get("returned_chars")
        next_offset = None
        if isinstance(current_offset, int) and isinstance(returned_chars, int):
            next_offset = current_offset + returned_chars
        if isinstance(remaining, int) and remaining > 0:
            return (
                f'You have only read part of the paper "{title or paper_id or "current paper"}". '
                f"There are still {remaining} unread characters remaining. "
                "Do not stop or wait for the user yet. "
                f"Continue the same paper by calling `read_loaded_paper_content(paper_id={paper_id!r}, title={title!r}, offset={next_offset}, max_chars=24000)` "
                "to read the next chunk, "
                "or, if you already have enough evidence from the paper text, save a digest with `save_paper_digest(...)` before moving to another paper."
            )
        return None

    @staticmethod
    def _paper_followup_prompt(task: TaskType, messages: list[dict[str, Any]]) -> str | None:
        if task not in {
            TaskType.ANALYZE_PAPER,
            TaskType.REVIEW_TOPIC,
            TaskType.FIND_PAPERS,
            TaskType.GENERATE_IDEAS,
            TaskType.SUGGEST_EXPERIMENTS,
            TaskType.MAP_RESEARCH_OPPORTUNITIES,
        }:
            return None
        for message in reversed(messages):
            if message.get("role") != "tool":
                continue
            if message.get("tool_name") != "read_loaded_paper_content":
                return None
            payload = AnalysisEngine._parse_tool_payload(message.get("content", ""))
            if not isinstance(payload, dict):
                return None
            remaining = payload.get("remaining_chars")
            paper = payload.get("paper") if isinstance(payload.get("paper"), dict) else {}
            paper_id = paper.get("paper_id")
            title = paper.get("title")
            current_offset = payload.get("offset")
            returned_chars = payload.get("returned_chars")
            next_offset = None
            if isinstance(current_offset, int) and isinstance(returned_chars, int):
                next_offset = current_offset + returned_chars
            if isinstance(remaining, int) and remaining > 0:
                return (
                    f'You just read only part of the loaded paper "{title or paper_id or "current paper"}" and there is more paper text remaining. '
                    "Do not wait for the user. "
                    f"Call `read_loaded_paper_content(paper_id={paper_id!r}, title={title!r}, offset={next_offset}, max_chars=24000)` to continue the same paper, "
                    "or save a digest with `save_paper_digest(...)` if you already have enough evidence."
                )
            return None
        return None

    @staticmethod
    def _parse_tool_payload(result: Any) -> Any:
        if isinstance(result, dict):
            return result
        if not isinstance(result, str):
            return None
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(result)
            except (ValueError, SyntaxError):
                return None

    @staticmethod
    def _paper_summary(paper: PaperDocument) -> dict[str, Any]:
        return {
            "paper_id": paper.metadata.paper_id,
            "title": paper.metadata.title,
            "source": paper.metadata.source,
            "authors": paper.metadata.authors,
            "pdf_url": str(paper.metadata.pdf_url) if paper.metadata.pdf_url else None,
            "local_path": str(paper.metadata.local_path) if paper.metadata.local_path else None,
        }

    def _build_runtime_tools(self, task_spec: TaskSpec, context: ResearchContext) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tool_specs: list[dict[str, Any]] = []
        tool_impl: dict[str, Any] = {}

        if "arxiv_search" in task_spec.allowed_tools:
            def search_arxiv_papers(professor: str, topic: str | None = None) -> str:
                if self.arxiv_search is None:
                    raise RuntimeError("arXiv search tool is not configured.")
                papers, notes = self.arxiv_search([professor], topic)
                existing_ids = {paper.metadata.paper_id for paper in context.papers}
                for paper in papers:
                    if paper.metadata.paper_id not in existing_ids:
                        context.papers.append(paper)
                        existing_ids.add(paper.metadata.paper_id)
                context.discovery_notes.extend(notes)
                return str(
                    {
                        "papers_added": len(papers),
                        "notes": notes,
                        "papers": [self._paper_summary(paper) for paper in papers],
                    }
                )

            tool_specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": "search_arxiv_papers",
                        "description": "Search arXiv papers for a professor name with an optional topic filter.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "professor": {"type": "string"},
                                "topic": {"type": "string"},
                            },
                            "required": ["professor"],
                        },
                    },
                }
            )
            tool_impl["search_arxiv_papers"] = search_arxiv_papers

            def fetch_arxiv_full_text(paper_id: str | None = None, title: str | None = None) -> str:
                if self.arxiv_fetch is None or self.workspace is None:
                    raise RuntimeError("arXiv full-text fetch is not configured.")
                if not paper_id and not title:
                    raise RuntimeError("Provide either paper_id or title.")

                matching = None
                if paper_id:
                    matching = next((paper for paper in context.papers if paper.metadata.paper_id == paper_id), None)
                elif title:
                    normalized_title = title.strip().casefold()
                    matches = [
                        paper
                        for paper in context.papers
                        if paper.metadata.title.strip().casefold() == normalized_title
                    ]
                    if len(matches) > 1:
                        raise RuntimeError(
                            f'Multiple loaded papers match title "{title}". Use paper_id instead.'
                        )
                    matching = matches[0] if matches else None
                if matching is None:
                    if paper_id:
                        raise RuntimeError(f"Paper id {paper_id} is not currently loaded in context.")
                    raise RuntimeError(f'Paper title "{title}" is not currently loaded in context.')
                document = self.arxiv_fetch(matching.metadata, self.workspace.resolve_path(".") / "arxiv_papers")
                existing_ids = {paper.metadata.paper_id for paper in context.papers}
                if document.metadata.paper_id not in existing_ids:
                    context.papers.append(document)
                else:
                    for index, paper in enumerate(context.papers):
                        if paper.metadata.paper_id == document.metadata.paper_id:
                            context.papers[index] = document
                            break
                note = f'Fetched full PDF text for "{document.metadata.title}".'
                context.discovery_notes.append(note)
                return str(
                    {
                        "paper_id": document.metadata.paper_id,
                        "title": document.metadata.title,
                        "local_path": str(document.metadata.local_path) if document.metadata.local_path else None,
                        "content_chars": len(document.content),
                        "note": note,
                    }
                )

            tool_specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_arxiv_full_text",
                        "description": "Download and parse the full PDF for an arXiv paper already present in context, then add its full text to the working set.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "paper_id": {"type": "string"},
                                "title": {"type": "string"},
                            },
                        },
                    },
                }
            )
            tool_impl["fetch_arxiv_full_text"] = fetch_arxiv_full_text

        if "local_pdf_reader" in task_spec.allowed_tools:
            def load_local_papers(directory: str | None = None, limit: int = 10) -> str:
                from pathlib import Path

                from gemma_ra.sources.local_papers import LocalPaperSource

                source = LocalPaperSource()
                target_dir = Path(directory) if directory else Path("./papers")
                paths = source.discover(target_dir)[: max(1, limit)]
                papers = source.read_many(paths)
                existing_ids = {paper.metadata.paper_id for paper in context.papers}
                added_titles: list[str] = []
                for paper in papers:
                    if paper.metadata.paper_id not in existing_ids:
                        context.papers.append(paper)
                        existing_ids.add(paper.metadata.paper_id)
                        if paper.metadata.local_path is not None:
                            context.local_paper_paths.append(paper.metadata.local_path)
                        added_titles.append(paper.metadata.title)
                note = f'Loaded {len(added_titles)} local paper(s) from "{target_dir}".'
                context.discovery_notes.append(note)
                return str({"papers_added": len(added_titles), "titles": added_titles, "note": note})

            def inspect_loaded_papers() -> str:
                return str([self._paper_summary(paper) for paper in context.papers])

            def read_loaded_paper_content(
                paper_id: str | None = None,
                title: str | None = None,
                max_chars: int = 24_000,
                offset: int = 0,
            ) -> str:
                if not paper_id and not title:
                    raise RuntimeError("Provide either paper_id or title.")
                if max_chars <= 0:
                    raise RuntimeError("max_chars must be positive.")
                if offset < 0:
                    raise RuntimeError("offset must be non-negative.")

                matching = None
                if paper_id:
                    matching = next((paper for paper in context.papers if paper.metadata.paper_id == paper_id), None)
                elif title:
                    normalized_title = title.strip().casefold()
                    matches = [
                        paper
                        for paper in context.papers
                        if paper.metadata.title.strip().casefold() == normalized_title
                    ]
                    if len(matches) > 1:
                        raise RuntimeError(
                            f'Multiple loaded papers match title "{title}". Use paper_id instead.'
                        )
                    matching = matches[0] if matches else None

                if matching is None:
                    if paper_id:
                        raise RuntimeError(f"Paper id {paper_id} is not currently loaded in context.")
                    raise RuntimeError(f'Paper title "{title}" is not currently loaded in context.')

                content = matching.content or ""
                start = min(offset, len(content))
                end = min(start + max_chars, len(content))
                excerpt = content[start:end]
                return str(
                    {
                        "paper": self._paper_summary(matching),
                        "offset": start,
                        "returned_chars": len(excerpt),
                        "remaining_chars": max(0, len(content) - end),
                        "content": excerpt,
                    }
                )

            tool_specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": "load_local_papers",
                        "description": "Load local PDF papers from a directory into the current working context. Use paths relative to the workspace root, usually '.' or a child folder name.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "directory": {"type": "string"},
                                "limit": {"type": "integer"},
                            },
                        },
                    },
                }
            )
            tool_impl["load_local_papers"] = load_local_papers

            tool_specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": "inspect_loaded_papers",
                        "description": "Inspect the already loaded local or remote papers available for analysis.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
            )
            tool_impl["inspect_loaded_papers"] = inspect_loaded_papers

            tool_specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": "read_loaded_paper_content",
                        "description": "Read the parsed content of a currently loaded paper. Use this after loading or fetching a PDF when you need the actual paper text, not just metadata. Prefer larger chunks around 20000-30000 characters for methods and experiments.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "paper_id": {"type": "string"},
                                "title": {"type": "string"},
                                "max_chars": {"type": "integer"},
                                "offset": {"type": "integer"},
                            },
                        },
                    },
                }
            )
            tool_impl["read_loaded_paper_content"] = read_loaded_paper_content

            def save_paper_digest(
                paper_id: str,
                title: str,
                summary: str,
                method: str,
                contributions: list[str],
                open_questions: list[str],
            ) -> str:
                digest = PaperDigest(
                    paper_id=paper_id,
                    title=title,
                    summary=summary,
                    method=method,
                    contributions=contributions,
                    open_questions=open_questions,
                )
                replaced = False
                for index, existing in enumerate(context.paper_digests):
                    if existing.paper_id == paper_id:
                        context.paper_digests[index] = digest
                        replaced = True
                        break
                if not replaced:
                    context.paper_digests.append(digest)
                note = f'Saved paper digest for "{title}".'
                context.discovery_notes.append(note)
                return str(digest.model_dump(mode="json"))

            tool_specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": "save_paper_digest",
                        "description": "Save a compact working-memory digest for a paper after reading it. Use this to preserve summary, method, contributions, and open questions before moving to other papers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "paper_id": {"type": "string"},
                                "title": {"type": "string"},
                                "summary": {"type": "string"},
                                "method": {"type": "string"},
                                "contributions": {"type": "array", "items": {"type": "string"}},
                                "open_questions": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["paper_id", "title", "summary", "method", "contributions", "open_questions"],
                        },
                    },
                }
            )
            tool_impl["save_paper_digest"] = save_paper_digest

        if "future_experiment_runner" in task_spec.allowed_tools and self.workspace is not None:
            def list_workspace_files(directory: str = ".", pattern: str = "*") -> str:
                files = self.workspace.list_files(directory=directory, pattern=pattern)
                return str({"files": files[:200], "count": len(files)})

            def read_workspace_file(path: str, max_bytes: int = 200_000) -> str:
                return self.workspace.read_file(path=path, max_bytes=max_bytes)

            def write_workspace_file(path: str, content: str) -> str:
                written = self.workspace.write_file(path=path, content=content)
                note = f'Wrote file "{written}".'
                context.discovery_notes.append(note)
                return note

            def append_workspace_file(path: str, content: str) -> str:
                result = self.workspace.append_file(path=path, content=content)
                note = f'Appended to file "{result["path"]}".'
                context.discovery_notes.append(note)
                return str(result)

            def replace_in_workspace_file(path: str, old_text: str, new_text: str, count: int = 1) -> str:
                result = self.workspace.replace_in_file(
                    path=path,
                    old_text=old_text,
                    new_text=new_text,
                    count=count,
                )
                note = f'Replaced text in "{result["path"]}".'
                context.discovery_notes.append(note)
                return str(result)

            def update_json_field(path: str, field: str, value: Any) -> str:
                result = self.workspace.update_json_field(path=path, field=field, value=value)
                note = f'Updated JSON field "{field}" in "{result["path"]}" to {value!r}.'
                context.discovery_notes.append(note)
                return str(result)

            def python_syntax_check(path: str) -> str:
                result = self.workspace.python_syntax_check(path=path)
                note = (
                    f'Python syntax check passed for "{result["path"]}".'
                    if result["ok"]
                    else f'Python syntax check failed for "{result["path"]}".'
                )
                context.discovery_notes.append(note)
                return str(result)

            def upsert_python_function(
                path: str,
                function_name: str,
                function_source: str,
                class_name: str | None = None,
            ) -> str:
                result = self.workspace.upsert_python_function(
                    path=path,
                    function_name=function_name,
                    function_source=function_source,
                    class_name=class_name,
                )
                note = f'{result["action"].title()} Python function "{function_name}" in "{result["path"]}".'
                context.discovery_notes.append(note)
                return str(result)

            def run_uv_python(
                script_path: str,
                args: list[str] | None = None,
                working_directory: str = ".",
                log_paths: list[str] | None = None,
            ) -> str:
                result = self.workspace.run_uv_python(
                    script_path=script_path,
                    args=args or [],
                    working_directory=working_directory,
                    log_paths=log_paths or [],
                )
                context.discovery_notes.append(
                    f"Ran uv python script {script_path} with return code {result['returncode']}."
                )
                return str(result)

            tool_specs.extend(
                [
                    {
                    "type": "function",
                    "function": {
                        "name": "list_workspace_files",
                        "description": "List files under the allowed workspace folder only. Prefer workspace-relative paths such as '.' or 'logs'.",
                        "parameters": {
                                "type": "object",
                                "properties": {
                                    "directory": {"type": "string"},
                                    "pattern": {"type": "string"},
                                },
                            },
                        },
                    },
                    {
                    "type": "function",
                    "function": {
                        "name": "read_workspace_file",
                        "description": "Read a text or log file from the allowed workspace folder only. Prefer workspace-relative paths such as 'config.json' or 'logs/latest.json'.",
                        "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "max_bytes": {"type": "integer"},
                                },
                                "required": ["path"],
                            },
                        },
                    },
                    {
                    "type": "function",
                    "function": {
                        "name": "write_workspace_file",
                        "description": "Write or overwrite a file inside the allowed workspace folder only. Prefer workspace-relative paths.",
                        "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["path", "content"],
                            },
                        },
                    },
                    {
                    "type": "function",
                    "function": {
                        "name": "append_workspace_file",
                        "description": "Append text to an existing file inside the allowed workspace. Good for adding helper code, notes, or small blocks.",
                        "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["path", "content"],
                            },
                        },
                    },
                    {
                    "type": "function",
                    "function": {
                        "name": "replace_in_workspace_file",
                        "description": "Replace an exact text snippet in a file inside the allowed workspace. Prefer this over full-file rewrites when editing code.",
                        "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "old_text": {"type": "string"},
                                    "new_text": {"type": "string"},
                                    "count": {"type": "integer"},
                                },
                                "required": ["path", "old_text", "new_text"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "update_json_field",
                            "description": "Update a single top-level field in a JSON file inside the allowed workspace folder. Prefer this for hyperparameter edits like config.json.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "field": {"type": "string"},
                                    "value": {},
                                },
                                "required": ["path", "field", "value"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "python_syntax_check",
                            "description": "Run a Python syntax check on a file inside the allowed workspace using py_compile.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                },
                                "required": ["path"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "upsert_python_function",
                            "description": "Create or replace a Python function in a .py file. Can target a top-level function or a method inside an existing class.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "function_name": {"type": "string"},
                                    "function_source": {"type": "string"},
                                    "class_name": {"type": "string"},
                                },
                                "required": ["path", "function_name", "function_source"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "run_uv_python",
                            "description": "Run a Python script with uv inside the allowed workspace folder only. Prefer workspace-relative script paths such as 'train.py'. If logs are expected, pass log_paths so execution watches them before returning.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "script_path": {"type": "string"},
                                    "args": {"type": "array", "items": {"type": "string"}},
                                    "working_directory": {"type": "string"},
                                    "log_paths": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["script_path"],
                            },
                        },
                    },
                ]
            )
            tool_impl["list_workspace_files"] = list_workspace_files
            tool_impl["read_workspace_file"] = read_workspace_file
            tool_impl["write_workspace_file"] = write_workspace_file
            tool_impl["append_workspace_file"] = append_workspace_file
            tool_impl["replace_in_workspace_file"] = replace_in_workspace_file
            tool_impl["update_json_field"] = update_json_field
            tool_impl["python_syntax_check"] = python_syntax_check
            tool_impl["upsert_python_function"] = upsert_python_function
            tool_impl["run_uv_python"] = run_uv_python

        return tool_specs, tool_impl

    def _handle_stream_chunk(self, chunk_type: str, text: str) -> None:
        if chunk_type == "thinking":
            self._report("thinking", text, end="")
        elif chunk_type == "content":
            self._report("model", text, end="")

    def _report(self, kind: str, message: str, end: str = "\n") -> None:
        if self.reporter is None:
            return
        self.reporter(kind, message, end=end)

    def _drain_interactive_guidance(self) -> list[str]:
        if self.interactive_guidance is None:
            return []
        messages = self.interactive_guidance()
        return [message.strip() for message in messages if message and message.strip()]

    @staticmethod
    def _anti_loop_instruction(tool_name: str, arguments: dict[str, Any], repeat_count: int) -> str:
        if repeat_count < 2:
            return ""
        path = str(arguments.get("path", ""))
        if tool_name == "read_workspace_file" and path == "logs/latest.json":
            return (
                "You have already read logs/latest.json and know the current metrics. "
                "Do not read the same log file again until you change a file or rerun training. "
                "Your next action must be one of: update_json_field, replace_in_workspace_file, "
                "upsert_python_function, python_syntax_check, or run_uv_python."
            )
        return ""

    @staticmethod
    def _instruction_state_update(tool_name: str, arguments: dict[str, Any], result: Any) -> str:
        if tool_name != "read_workspace_file":
            return ""
        path = str(arguments.get("path", ""))
        if not path.endswith(".json"):
            return ""
        try:
            payload = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return ""
        if not isinstance(payload, dict):
            return ""

        state_parts: list[str] = [f"Read {path}."]
        if "success" in payload:
            state_parts.append(f"success={payload['success']}.")
        if "final_loss" in payload and "target_loss" in payload:
            state_parts.append(
                f"final_loss={payload['final_loss']} and target_loss={payload['target_loss']}."
            )
        if "validation_accuracy" in payload and "target_accuracy" in payload:
            state_parts.append(
                f"validation_accuracy={payload['validation_accuracy']} and target_accuracy={payload['target_accuracy']}."
            )
        return " ".join(state_parts)

    @staticmethod
    def _next_action_instruction(tool_name: str, arguments: dict[str, Any], result: Any) -> str:
        if tool_name != "read_workspace_file":
            return ""
        path = str(arguments.get("path", ""))
        if path != "logs/latest.json":
            return ""
        try:
            payload = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return ""
        if not isinstance(payload, dict):
            return ""
        if payload.get("success") is True:
            return ""

        suggestions: list[str] = []
        if "learning_rate" in payload:
            suggestions.append("adjust `learning_rate` in `config.json` with `update_json_field`")
        if "epochs" in payload:
            suggestions.append("increase `epochs` in `config.json` with `update_json_field`")
        for field in ("hidden_size", "depth", "dropout", "weight_decay"):
            if field in payload:
                suggestions.append(f"change `{field}` in `config.json` with `update_json_field`")

        suggestions_text = "; ".join(suggestions[:5]) or (
            "edit `config.json` or `train.py`, then rerun training"
        )
        return (
            "The latest metrics do not meet success criteria. "
            "Do not write a prose analysis. "
            "Your next step should be a concrete change followed by a rerun. "
            f"Prefer `config.json` before `train.py`. Likely next actions: {suggestions_text}. "
            "After making one concrete change, use `run_uv_python` on `train.py`."
        )

    @staticmethod
    def _success_criteria_summary(instructions: str | None) -> str:
        if not instructions:
            return "No explicit success criteria were provided."

        lines = [line.rstrip() for line in instructions.splitlines()]
        start_index = None
        for index, line in enumerate(lines):
            if line.strip().lower().startswith("success criteria"):
                start_index = index + 1
                break
        if start_index is None:
            return "No explicit success criteria were provided."

        collected: list[str] = []
        for line in lines[start_index:]:
            stripped = line.strip()
            if not stripped:
                if collected:
                    break
                continue
            if stripped.startswith(("-", "*")):
                collected.append(stripped[1:].strip())
                continue
            lowered = stripped.lower()
            if lowered.startswith("rules:") or lowered.startswith("available behaviors"):
                break
            if collected:
                break
        if not collected:
            return "No explicit success criteria were provided."
        return "; ".join(collected)

    def _build_prompt(self, task_spec: TaskSpec, context: ResearchContext) -> str:
        digested_ids = {digest.paper_id for digest in context.paper_digests}
        raw_papers = [doc for doc in context.papers if doc.metadata.paper_id not in digested_ids][:10]
        raw_char_limit = 3000 if context.paper_digests else 12000
        papers = "\n\n".join(self._format_paper(doc, max_chars=raw_char_limit) for doc in raw_papers)
        paper_digests = "\n\n".join(self._format_paper_digest(digest) for digest in context.paper_digests[:10])
        professors = ", ".join(context.professors) if context.professors else "None"
        topic = context.topic or "Not specified"
        return task_spec.prompt_template.format(
            topic=topic,
            professors=professors,
            papers=(
                ("Saved paper digests:\n" + paper_digests + "\n\n" if paper_digests else "")
                + (papers or "No paper contents available.")
            ),
            output_sections=", ".join(task_spec.output_sections),
            constraints="\n".join(f"- {item}" for item in task_spec.constraints) or "- None",
            instructions=context.instructions or "No instructions provided.",
        )

    @staticmethod
    def _format_paper(doc: PaperDocument, max_chars: int = 12000) -> str:
        metadata = doc.metadata
        return (
            f"Title: {metadata.title}\n"
            f"Authors: {', '.join(metadata.authors) or 'Unknown'}\n"
            f"Source: {metadata.source}\n"
            f"Abstract: {metadata.abstract or 'N/A'}\n"
            f"Content:\n{doc.content[:max_chars]}"
        )

    @staticmethod
    def _format_paper_digest(digest: PaperDigest) -> str:
        return (
            f"Paper: {digest.title}\n"
            f"Summary: {digest.summary}\n"
            f"Method: {digest.method}\n"
            f"Contributions: {', '.join(digest.contributions)}\n"
            f"Open questions: {', '.join(digest.open_questions)}"
        )

    @staticmethod
    def _response_schema(task: TaskType) -> dict[str, Any]:
        schema_map = {
            TaskType.ANALYZE_PAPER: {
                "type": "object",
                "properties": {
                    "problem": {"type": "string"},
                    "inputs": {"type": "array", "items": {"type": "string"}},
                    "processing": {"type": "string"},
                    "outputs": {"type": "array", "items": {"type": "string"}},
                    "key_ideas": {"type": "array", "items": {"type": "string"}},
                    "contributions": {"type": "array", "items": {"type": "string"}},
                    "limitations": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "problem",
                    "inputs",
                    "processing",
                    "outputs",
                    "key_ideas",
                    "contributions",
                    "limitations",
                ],
            },
            TaskType.REVIEW_TOPIC: {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "field_overview": {"type": "string"},
                    "key_problems": {"type": "array", "items": {"type": "string"}},
                    "paper_summaries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "paper_title": {"type": "string"},
                                "problem": {"type": "string"},
                                "method": {"type": "string"},
                                "contributions": {"type": "array", "items": {"type": "string"}},
                                "limitations": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "paper_title",
                                "problem",
                                "method",
                                "contributions",
                                "limitations",
                            ],
                        },
                    },
                    "themes": {"type": "array", "items": {"type": "string"}},
                    "methodological_trends": {"type": "array", "items": {"type": "string"}},
                    "gaps": {"type": "array", "items": {"type": "string"}},
                    "disagreements": {"type": "array", "items": {"type": "string"}},
                    "synthesis": {"type": "string"},
                },
                "required": [
                    "topic",
                    "field_overview",
                    "key_problems",
                    "paper_summaries",
                    "themes",
                    "methodological_trends",
                    "gaps",
                    "disagreements",
                    "synthesis",
                ],
            },
            TaskType.FIND_PAPERS: {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "note": {"type": "string"},
                    "discovery_notes": {"type": "array", "items": {"type": "string"}},
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "authors": {"type": "array", "items": {"type": "string"}},
                                "problem": {"type": "string"},
                                "method": {"type": "string"},
                                "contributions": {"type": "array", "items": {"type": "string"}},
                                "why_relevant": {"type": "string"},
                            },
                            "required": [
                                "title",
                                "authors",
                                "problem",
                                "method",
                                "contributions",
                                "why_relevant",
                            ],
                        },
                    },
                },
                "required": ["topic", "results"],
            },
            TaskType.GENERATE_IDEAS: {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "ideas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "problem_targeted": {"type": "string"},
                                "motivation": {"type": "string"},
                                "novelty_rationale": {"type": "string"},
                                "grounding": {"type": "string"},
                                "expected_contribution": {"type": "string"},
                                "related_papers": {"type": "array", "items": {"type": "string"}},
                                "proposed_method": {"type": "string"},
                                "risks": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "title",
                                "problem_targeted",
                                "motivation",
                                "novelty_rationale",
                                "grounding",
                                "expected_contribution",
                                "related_papers",
                                "proposed_method",
                                "risks",
                            ],
                        },
                    },
                },
                "required": ["topic", "ideas"],
            },
            TaskType.SUGGEST_EXPERIMENTS: {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "experiments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "related_papers": {"type": "array", "items": {"type": "string"}},
                                "baseline": {"type": "string"},
                                "hypothesis": {"type": "string"},
                                "setup": {"type": "string"},
                                "metrics": {"type": "array", "items": {"type": "string"}},
                                "expected_signal": {"type": "string"},
                                "why_this_matters": {"type": "string"},
                                "failure_conditions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "title",
                                "related_papers",
                                "baseline",
                                "hypothesis",
                                "setup",
                                "metrics",
                                "expected_signal",
                                "why_this_matters",
                                "failure_conditions",
                            ],
                        },
                    },
                },
                "required": ["topic", "experiments"],
            },
            TaskType.MAP_RESEARCH_OPPORTUNITIES: {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "suggested_professors": {"type": "array", "items": {"type": "string"}},
                    "coauthor_leads": {"type": "array", "items": {"type": "string"}},
                    "field_summary": {"type": "string"},
                    "hot_topics": {"type": "array", "items": {"type": "string"}},
                    "open_problems": {"type": "array", "items": {"type": "string"}},
                    "opportunities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "problem_targeted": {"type": "string"},
                                "motivation": {"type": "string"},
                                "novelty_rationale": {"type": "string"},
                                "grounding": {"type": "string"},
                                "expected_contribution": {"type": "string"},
                                "related_papers": {"type": "array", "items": {"type": "string"}},
                                "proposed_method": {"type": "string"},
                                "risks": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "title",
                                "problem_targeted",
                                "motivation",
                                "novelty_rationale",
                                "grounding",
                                "expected_contribution",
                                "related_papers",
                                "proposed_method",
                                "risks",
                            ],
                        },
                    },
                    "experiments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "related_papers": {"type": "array", "items": {"type": "string"}},
                                "baseline": {"type": "string"},
                                "hypothesis": {"type": "string"},
                                "setup": {"type": "string"},
                                "metrics": {"type": "array", "items": {"type": "string"}},
                                "expected_signal": {"type": "string"},
                                "why_this_matters": {"type": "string"},
                                "failure_conditions": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "title",
                                "related_papers",
                                "baseline",
                                "hypothesis",
                                "setup",
                                "metrics",
                                "expected_signal",
                                "why_this_matters",
                                "failure_conditions",
                            ],
                        },
                    },
                    "search_strategy": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "topic",
                    "suggested_professors",
                    "coauthor_leads",
                    "field_summary",
                    "hot_topics",
                    "open_problems",
                    "opportunities",
                    "experiments",
                    "search_strategy",
                ],
            },
            TaskType.RUN_INSTRUCTIONS: {
                "type": "object",
                "properties": {
                    "request": {"type": "string"},
                    "summary": {"type": "string"},
                    "actions_taken": {"type": "array", "items": {"type": "string"}},
                    "key_findings": {"type": "array", "items": {"type": "string"}},
                    "hot_topics": {"type": "array", "items": {"type": "string"}},
                    "research_opportunities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "problem_targeted": {"type": "string"},
                                "motivation": {"type": "string"},
                                "novelty_rationale": {"type": "string"},
                                "grounding": {"type": "string"},
                                "expected_contribution": {"type": "string"},
                                "related_papers": {"type": "array", "items": {"type": "string"}},
                                "proposed_method": {"type": "string"},
                                "risks": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "title",
                                "problem_targeted",
                                "motivation",
                                "novelty_rationale",
                                "grounding",
                                "expected_contribution",
                                "related_papers",
                                "proposed_method",
                                "risks",
                            ],
                        },
                    },
                    "experiment_suggestions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "related_papers": {"type": "array", "items": {"type": "string"}},
                                "baseline": {"type": "string"},
                                "hypothesis": {"type": "string"},
                                "setup": {"type": "string"},
                                "metrics": {"type": "array", "items": {"type": "string"}},
                                "expected_signal": {"type": "string"},
                                "why_this_matters": {"type": "string"},
                                "failure_conditions": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "title",
                                "related_papers",
                                "baseline",
                                "hypothesis",
                                "setup",
                                "metrics",
                                "expected_signal",
                                "why_this_matters",
                                "failure_conditions",
                            ],
                        },
                    },
                    "next_steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "request",
                    "summary",
                    "actions_taken",
                    "key_findings",
                    "hot_topics",
                    "research_opportunities",
                    "experiment_suggestions",
                    "next_steps",
                ],
            },
        }
        return schema_map[task]

    @staticmethod
    def _validate_response(task: TaskType, raw: dict[str, Any], context: ResearchContext) -> dict[str, Any]:
        if task == TaskType.ANALYZE_PAPER:
            metadata = context.papers[0].metadata
            raw["paper"] = metadata.model_dump(mode="json")
            return PaperAnalysis.model_validate(raw).model_dump(mode="json")
        if task == TaskType.REVIEW_TOPIC:
            raw["papers"] = [paper.metadata.model_dump(mode="json") for paper in context.papers]
            raw.setdefault("topic", context.topic or "Research review")
            paper_lookup = {paper.metadata.title: paper.metadata for paper in context.papers}
            raw["paper_summaries"] = [
                ReviewPaperSummary(
                    paper=paper_lookup[item["paper_title"]].model_dump(mode="json"),
                    problem=item["problem"],
                    method=item["method"],
                    contributions=item["contributions"],
                    limitations=item["limitations"],
                ).model_dump(mode="json")
                for item in raw["paper_summaries"]
                if item["paper_title"] in paper_lookup
            ]
            return LiteratureReview.model_validate(raw).model_dump(mode="json")
        if task == TaskType.GENERATE_IDEAS:
            ideas = [ResearchIdea.model_validate(item).model_dump(mode="json") for item in raw["ideas"]]
            return {"topic": raw["topic"], "ideas": ideas}
        if task == TaskType.SUGGEST_EXPERIMENTS:
            experiments = [
                ExperimentSuggestion.model_validate(item).model_dump(mode="json")
                for item in raw["experiments"]
            ]
            return {"topic": raw["topic"], "experiments": experiments}
        if task == TaskType.MAP_RESEARCH_OPPORTUNITIES:
            opportunities = [
                ResearchIdea.model_validate(item).model_dump(mode="json")
                for item in raw["opportunities"]
            ]
            experiments = [
                ExperimentSuggestion.model_validate(item).model_dump(mode="json")
                for item in raw["experiments"]
            ]
            raw["papers"] = [paper.metadata.model_dump(mode="json") for paper in context.papers]
            raw["search_strategy"] = raw.get("search_strategy", []) + context.discovery_notes
            raw["opportunities"] = opportunities
            raw["experiments"] = experiments
            return ResearchOpportunityMap.model_validate(raw).model_dump(mode="json")
        if task == TaskType.RUN_INSTRUCTIONS:
            opportunities = [
                ResearchIdea.model_validate(item).model_dump(mode="json")
                for item in raw["research_opportunities"]
            ]
            experiments = [
                ExperimentSuggestion.model_validate(item).model_dump(mode="json")
                for item in raw["experiment_suggestions"]
            ]
            raw["papers_considered"] = [paper.metadata.model_dump(mode="json") for paper in context.papers]
            raw["actions_taken"] = raw.get("actions_taken", []) + context.discovery_notes
            raw["research_opportunities"] = opportunities
            raw["experiment_suggestions"] = experiments
            raw.setdefault("request", context.instructions or "")
            return InstructionRunResult.model_validate(raw).model_dump(mode="json")
        raw.setdefault("topic", context.topic or "Paper discovery")
        raw.setdefault("results", [])
        raw.setdefault("note", "")
        raw.setdefault("discovery_notes", context.discovery_notes)
        return raw
