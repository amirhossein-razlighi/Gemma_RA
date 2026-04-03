from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from gemma_ra.core.model_client import OllamaClient
from gemma_ra.core.schemas import (
    ExperimentSuggestion,
    InstructionRunResult,
    LiteratureReview,
    PaperAnalysis,
    PaperDocument,
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
        workspace: WorkspaceExecutor | None = None,
        max_iterations: int = 3,
        reporter=None,
        stream_chat: bool = False,
    ) -> None:
        self.model_client = model_client
        self.arxiv_search = arxiv_search
        self.workspace = workspace
        self.max_iterations = max_iterations
        self.reporter = reporter
        self.stream_chat = stream_chat

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

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant in a tool loop. "
                    "Call tools when you need more papers or metadata. "
                    "You may inspect local paper folders, search arXiv by professor, and then synthesize an answer. "
                    "When you have enough context, respond without tool calls."
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
                    f"Instructions: {context.instructions or 'None'}"
                ),
            },
        ]

        for iteration_index in range(self.max_iterations):
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
                                "If success criteria are not met, call the next tool now. "
                                "Do not just describe the action. "
                                "Use existing workspace-relative paths only, such as '.' , 'config.json', 'train.py', or 'logs/latest.json'. "
                                "If a previous tool failed because of a bad path, correct the path before trying again. "
                                "If and only if the success criteria are already met, reply exactly with FINISHED."
                            ),
                        }
                    )
                    continue
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
        if content == "FINISHED":
            return False
        return True

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
                        "titles": [paper.metadata.title for paper in papers],
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
                return str(
                    [
                        {
                            "title": paper.metadata.title,
                            "source": paper.metadata.source,
                            "authors": paper.metadata.authors,
                        }
                        for paper in context.papers
                    ]
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

            def update_json_field(path: str, field: str, value: Any) -> str:
                result = self.workspace.update_json_field(path=path, field=field, value=value)
                note = f'Updated JSON field "{field}" in "{result["path"]}" to {value!r}.'
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
            tool_impl["update_json_field"] = update_json_field
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

    def _build_prompt(self, task_spec: TaskSpec, context: ResearchContext) -> str:
        papers = "\n\n".join(self._format_paper(doc) for doc in context.papers[:10])
        professors = ", ".join(context.professors) if context.professors else "None"
        topic = context.topic or "Not specified"
        return task_spec.prompt_template.format(
            topic=topic,
            professors=professors,
            papers=papers or "No paper contents available.",
            output_sections=", ".join(task_spec.output_sections),
            constraints="\n".join(f"- {item}" for item in task_spec.constraints) or "- None",
            instructions=context.instructions or "No instructions provided.",
        )

    @staticmethod
    def _format_paper(doc: PaperDocument) -> str:
        metadata = doc.metadata
        return (
            f"Title: {metadata.title}\n"
            f"Authors: {', '.join(metadata.authors) or 'Unknown'}\n"
            f"Source: {metadata.source}\n"
            f"Abstract: {metadata.abstract or 'N/A'}\n"
            f"Content:\n{doc.content[:12000]}"
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
                    "themes": {"type": "array", "items": {"type": "string"}},
                    "gaps": {"type": "array", "items": {"type": "string"}},
                    "disagreements": {"type": "array", "items": {"type": "string"}},
                    "synthesis": {"type": "string"},
                },
                "required": ["topic", "themes", "gaps", "disagreements", "synthesis"],
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
                                "why_relevant": {"type": "string"},
                            },
                            "required": ["title", "authors", "why_relevant"],
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
                                "motivation": {"type": "string"},
                                "novelty_rationale": {"type": "string"},
                                "related_papers": {"type": "array", "items": {"type": "string"}},
                                "proposed_method": {"type": "string"},
                                "risks": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "title",
                                "motivation",
                                "novelty_rationale",
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
                                "hypothesis": {"type": "string"},
                                "setup": {"type": "string"},
                                "metrics": {"type": "array", "items": {"type": "string"}},
                                "expected_signal": {"type": "string"},
                                "failure_conditions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "title",
                                "hypothesis",
                                "setup",
                                "metrics",
                                "expected_signal",
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
                                "motivation": {"type": "string"},
                                "novelty_rationale": {"type": "string"},
                                "related_papers": {"type": "array", "items": {"type": "string"}},
                                "proposed_method": {"type": "string"},
                                "risks": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "title",
                                "motivation",
                                "novelty_rationale",
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
                                "hypothesis": {"type": "string"},
                                "setup": {"type": "string"},
                                "metrics": {"type": "array", "items": {"type": "string"}},
                                "expected_signal": {"type": "string"},
                                "failure_conditions": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "title",
                                "hypothesis",
                                "setup",
                                "metrics",
                                "expected_signal",
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
                                "motivation": {"type": "string"},
                                "novelty_rationale": {"type": "string"},
                                "related_papers": {"type": "array", "items": {"type": "string"}},
                                "proposed_method": {"type": "string"},
                                "risks": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "title",
                                "motivation",
                                "novelty_rationale",
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
                                "hypothesis": {"type": "string"},
                                "setup": {"type": "string"},
                                "metrics": {"type": "array", "items": {"type": "string"}},
                                "expected_signal": {"type": "string"},
                                "failure_conditions": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "title",
                                "hypothesis",
                                "setup",
                                "metrics",
                                "expected_signal",
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
