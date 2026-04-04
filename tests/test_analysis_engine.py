import json
from pathlib import Path

from gemma_ra.analysis.engine import AnalysisEngine
from gemma_ra.core.schemas import PaperDocument, PaperMetadata, ResearchContext, TaskType
from gemma_ra.core.tasks import get_task_spec


class FakeClient:
    def chat(self, messages, tools=None) -> dict:
        return {"message": {"role": "assistant", "content": "Ready", "tool_calls": []}}

    def generate_structured(self, prompt: str, schema: dict) -> dict:
        assert "Graph neural nets" in prompt
        return {
            "problem": "Predict molecular properties.",
            "inputs": ["Molecular graphs"],
            "processing": "Encode graph structure and aggregate message passing features.",
            "outputs": ["Property predictions"],
            "key_ideas": ["Use message passing"],
            "contributions": ["Strong benchmark results"],
            "limitations": ["Needs labeled data"],
        }


def test_analysis_engine_validates_analyze_paper_output() -> None:
    engine = AnalysisEngine(FakeClient())
    paper = PaperDocument(
        metadata=PaperMetadata(
            paper_id="paper-1",
            title="Graph neural nets",
            authors=["Jane Doe"],
            source="local",
        ),
        content="Graph neural nets for molecules.",
        sections=[],
    )
    context = ResearchContext(task=TaskType.ANALYZE_PAPER, papers=[paper])

    result = engine.run(TaskType.ANALYZE_PAPER, get_task_spec(TaskType.ANALYZE_PAPER), context)

    assert result.content["problem"] == "Predict molecular properties."
    assert result.content["paper"]["title"] == "Graph neural nets"


def test_find_papers_returns_helpful_note_when_context_is_empty() -> None:
    engine = AnalysisEngine(FakeClient())
    context = ResearchContext(
        task=TaskType.FIND_PAPERS,
        topic="generative ai",
        professors=["Ali mahdavi amiri"],
        discovery_notes=['No arXiv papers matched professor "Ali mahdavi amiri" with topic "generative ai".'],
    )

    result = engine.run(TaskType.FIND_PAPERS, get_task_spec(TaskType.FIND_PAPERS), context)

    assert result.content["results"] == []
    assert "No papers were found" in result.content["note"]


def test_map_research_opportunities_validates_composite_output() -> None:
    class CompositeClient:
        def chat(self, messages, tools=None) -> dict:
            return {"message": {"role": "assistant", "content": "Ready", "tool_calls": []}}

        def generate_structured(self, prompt: str, schema: dict) -> dict:
            assert "Topic: agentic retrieval" in prompt
            return {
                "topic": "agentic retrieval",
                "suggested_professors": ["Percy Liang", "Yejin Choi"],
                "coauthor_leads": ["Tatsunori Hashimoto"],
                "field_summary": "The field combines retrieval, planning, and grounded generation.",
                "hot_topics": ["retrieval planning", "tool-using agents"],
                "open_problems": ["robustness to noisy corpora"],
                "opportunities": [
                    {
                        "title": "Adaptive retrieval budgets",
                        "motivation": "Current systems over-retrieve.",
                        "novelty_rationale": "Budget control is underexplored for small agents.",
                        "related_papers": ["Paper A"],
                        "proposed_method": "Learn a stopping policy for retrieval hops.",
                        "risks": ["May reduce recall"],
                    }
                ],
                "experiments": [
                    {
                        "title": "Budget ablation",
                        "hypothesis": "Adaptive retrieval can match quality with fewer calls.",
                        "setup": "Compare fixed and adaptive budgets on QA tasks.",
                        "metrics": ["accuracy", "tool calls"],
                        "expected_signal": "Same accuracy with fewer retrieval steps.",
                        "failure_conditions": ["Accuracy drops sharply"],
                    }
                ],
                "search_strategy": ["Started from two seed professors."],
            }

    engine = AnalysisEngine(CompositeClient())
    paper = PaperDocument(
        metadata=PaperMetadata(
            paper_id="paper-1",
            title="Agentic Retrieval Systems",
            authors=["Jane Doe"],
            source="arxiv",
        ),
        content="Research on agentic retrieval systems.",
        sections=[],
    )
    context = ResearchContext(
        task=TaskType.MAP_RESEARCH_OPPORTUNITIES,
        topic="agentic retrieval",
        papers=[paper],
        discovery_notes=["Searched seed professor list."],
    )

    result = engine.run(
        TaskType.MAP_RESEARCH_OPPORTUNITIES,
        get_task_spec(TaskType.MAP_RESEARCH_OPPORTUNITIES),
        context,
    )

    assert result.content["opportunities"][0]["title"] == "Adaptive retrieval budgets"
    assert "Searched seed professor list." in result.content["search_strategy"]


def test_run_instructions_validates_instruction_output() -> None:
    class InstructionClient:
        def chat(self, messages, tools=None) -> dict:
            return {"message": {"role": "assistant", "content": "Ready", "tool_calls": []}}

        def generate_structured(self, prompt: str, schema: dict) -> dict:
            assert "READ PAPERS FROM ANDREAS TAGLIASACCHI" in prompt
            return {
                "request": "READ PAPERS FROM ANDREAS TAGLIASACCHI RELATED TO GAUSSIAN SPLATTING AND SUMMARIZE THEM FOR ME",
                "summary": "The papers center on efficient 3D representation and rendering.",
                "actions_taken": ["searched arXiv for relevant professor names"],
                "key_findings": ["Gaussian splatting is being optimized for quality-speed tradeoffs."],
                "hot_topics": ["3D Gaussian splatting", "real-time rendering"],
                "research_opportunities": [
                    {
                        "title": "Adaptive splat pruning",
                        "motivation": "Rendering cost remains high in dense scenes.",
                        "novelty_rationale": "Dynamic pruning policies appear underexplored.",
                        "related_papers": ["Paper A"],
                        "proposed_method": "Learn scene-aware pruning during rendering.",
                        "risks": ["Visual artifacts"],
                    }
                ],
                "experiment_suggestions": [
                    {
                        "title": "Pruning pilot",
                        "hypothesis": "Adaptive pruning reduces render cost with limited quality loss.",
                        "setup": "Compare static and adaptive pruning on benchmark scenes.",
                        "metrics": ["FPS", "PSNR"],
                        "expected_signal": "Higher FPS with similar PSNR.",
                        "failure_conditions": ["Sharp PSNR drop"],
                    }
                ],
                "next_steps": ["Fetch the top 5 most relevant papers and compare their splat control strategies."],
            }

    engine = AnalysisEngine(InstructionClient())
    context = ResearchContext(
        task=TaskType.RUN_INSTRUCTIONS,
        instructions="READ PAPERS FROM ANDREAS TAGLIASACCHI RELATED TO GAUSSIAN SPLATTING AND SUMMARIZE THEM FOR ME",
        discovery_notes=["searched arXiv for related authors"],
    )

    result = engine.run(TaskType.RUN_INSTRUCTIONS, get_task_spec(TaskType.RUN_INSTRUCTIONS), context)

    assert result.content["summary"].startswith("The papers center")
    assert result.content["research_opportunities"][0]["title"] == "Adaptive splat pruning"


def test_run_instructions_reprompts_when_model_narrates_without_tool_call() -> None:
    class RecoveryClient:
        def __init__(self) -> None:
            self.chat_calls = 0

        def chat(self, messages, tools=None) -> dict:
            self.chat_calls += 1
            if self.chat_calls == 1:
                return {"message": {"role": "assistant", "content": "I will update config.json now.", "tool_calls": []}}
            if self.chat_calls == 2:
                return {"message": {"role": "assistant", "content": "", "tool_calls": []}}
            return {"message": {"role": "assistant", "content": "FINISHED", "tool_calls": []}}

        def generate_structured(self, prompt: str, schema: dict) -> dict:
            return {
                "request": "Tune the config until success.",
                "summary": "Stopped after the recovery prompt.",
                "actions_taken": [],
                "key_findings": [],
                "hot_topics": [],
                "research_opportunities": [],
                "experiment_suggestions": [],
                "next_steps": [],
            }

    client = RecoveryClient()
    engine = AnalysisEngine(client, max_iterations=3)
    context = ResearchContext(task=TaskType.RUN_INSTRUCTIONS, instructions="Tune config.")

    engine.run(TaskType.RUN_INSTRUCTIONS, get_task_spec(TaskType.RUN_INSTRUCTIONS), context)

    assert client.chat_calls == 2


def test_run_instructions_reprompts_when_only_thinking_is_present() -> None:
    class ThinkingOnlyClient:
        def __init__(self) -> None:
            self.chat_calls = 0

        def chat(self, messages, tools=None) -> dict:
            self.chat_calls += 1
            if self.chat_calls == 1:
                return {"message": {"role": "assistant", "content": "", "thinking": "I should update config.json.", "tool_calls": []}}
            return {"message": {"role": "assistant", "content": "FINISHED", "tool_calls": []}}

        def generate_structured(self, prompt: str, schema: dict) -> dict:
            return {
                "request": "Tune config until success.",
                "summary": "Recovered after thinking-only response.",
                "actions_taken": [],
                "key_findings": [],
                "hot_topics": [],
                "research_opportunities": [],
                "experiment_suggestions": [],
                "next_steps": [],
            }

    client = ThinkingOnlyClient()
    engine = AnalysisEngine(client, max_iterations=2)
    context = ResearchContext(task=TaskType.RUN_INSTRUCTIONS, instructions="Tune config.")

    engine.run(TaskType.RUN_INSTRUCTIONS, get_task_spec(TaskType.RUN_INSTRUCTIONS), context)

    assert client.chat_calls == 2


def test_inspect_loaded_papers_exposes_ids_and_locations() -> None:
    reports: list[tuple[str, str]] = []

    class InspectClient:
        def chat(self, messages, tools=None) -> dict:
            tool_call = {
                "function": {
                    "name": "inspect_loaded_papers",
                    "arguments": {},
                }
            }
            return {"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}

        def generate_structured(self, prompt: str, schema: dict) -> dict:
            return {
                "request": "Inspect the currently loaded papers before summarizing them.",
                "summary": "Inspected the loaded papers.",
                "actions_taken": ["inspected currently loaded papers"],
                "key_findings": [],
                "hot_topics": [],
                "research_opportunities": [],
                "experiment_suggestions": [],
                "next_steps": [],
            }

    def reporter(kind: str, text: str, end: str = "\n") -> None:
        reports.append((kind, text))

    engine = AnalysisEngine(InspectClient(), max_iterations=1, reporter=reporter)
    paper = PaperDocument(
        metadata=PaperMetadata(
            paper_id="paper-1",
            title="Graph neural nets",
            authors=["Jane Doe"],
            pdf_url="https://arxiv.org/pdf/paper-1",
            local_path=Path("/tmp/paper-1.pdf"),
            source="arxiv",
        ),
        content="Graph neural nets for molecules.",
        sections=[],
    )
    context = ResearchContext(
        task=TaskType.RUN_INSTRUCTIONS,
        papers=[paper],
        instructions="Inspect the currently loaded papers before summarizing them.",
    )

    result = engine.run(TaskType.RUN_INSTRUCTIONS, get_task_spec(TaskType.RUN_INSTRUCTIONS), context)

    assert result.content["request"] == "Inspect the currently loaded papers before summarizing them."
    tool_results = [text for kind, text in reports if kind == "tool_result"]
    assert any("paper-1" in text for text in tool_results)
    assert any("https://arxiv.org/pdf/paper-1" in text for text in tool_results)
    assert any("/tmp/paper-1.pdf" in text for text in tool_results)


def test_fetch_arxiv_full_text_accepts_title_lookup() -> None:
    fetched: list[str] = []

    class FetchClient:
        def chat(self, messages, tools=None) -> dict:
            tool_call = {
                "function": {
                    "name": "fetch_arxiv_full_text",
                    "arguments": {"title": "Sample Paper"},
                }
            }
            return {"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}

        def generate_structured(self, prompt: str, schema: dict) -> dict:
            return {
                "request": "Fetch the paper PDF before summarizing it.",
                "summary": "Fetched the full PDF before summarizing.",
                "actions_taken": ["fetched arXiv full text"],
                "key_findings": ["The full PDF is now loaded in context."],
                "hot_topics": [],
                "research_opportunities": [],
                "experiment_suggestions": [],
                "next_steps": [],
            }

    def fake_fetch(metadata, download_dir):
        fetched.append(metadata.paper_id)
        return PaperDocument(
            metadata=PaperMetadata(
                paper_id=metadata.paper_id,
                title=metadata.title,
                authors=metadata.authors,
                pdf_url=metadata.pdf_url,
                local_path=download_dir / f"{metadata.paper_id}.pdf",
                source="arxiv_pdf",
            ),
            content="Full paper text",
            sections=[],
        )

    class WorkspaceStub:
        @staticmethod
        def resolve_path(path: str) -> Path:
            return Path("/tmp")

    engine = AnalysisEngine(
        FetchClient(),
        arxiv_fetch=fake_fetch,
        workspace=WorkspaceStub(),
        max_iterations=1,
    )
    context = ResearchContext(
        task=TaskType.RUN_INSTRUCTIONS,
        instructions="Fetch the paper PDF before summarizing it.",
        papers=[
            PaperDocument(
                metadata=PaperMetadata(
                    paper_id="1234.5678v1",
                    title="Sample Paper",
                    authors=["Jane Doe"],
                    pdf_url="https://arxiv.org/pdf/1234.5678v1",
                    source="arxiv",
                ),
                content="Short abstract.",
                sections=[],
            )
        ],
    )

    engine.run(TaskType.RUN_INSTRUCTIONS, get_task_spec(TaskType.RUN_INSTRUCTIONS), context)

    assert fetched == ["1234.5678v1"]
    assert context.papers[0].metadata.source == "arxiv_pdf"


def test_read_loaded_paper_content_exposes_parsed_text_to_tool_loop() -> None:
    reports: list[tuple[str, str]] = []

    class ReadClient:
        def chat(self, messages, tools=None) -> dict:
            tool_call = {
                "function": {
                    "name": "read_loaded_paper_content",
                    "arguments": {"paper_id": "paper-1", "max_chars": 80},
                }
            }
            return {"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}

        def generate_structured(self, prompt: str, schema: dict) -> dict:
            return {
                "problem": "Predict molecular properties.",
                "inputs": ["Molecular graphs"],
                "processing": "Encode graph structure and aggregate message passing features.",
                "outputs": ["Property predictions"],
                "key_ideas": ["Use message passing"],
                "contributions": ["Strong benchmark results"],
                "limitations": ["Needs labeled data"],
            }

    def reporter(kind: str, text: str, end: str = "\n") -> None:
        reports.append((kind, text))

    engine = AnalysisEngine(ReadClient(), max_iterations=1, reporter=reporter)
    paper = PaperDocument(
        metadata=PaperMetadata(
            paper_id="paper-1",
            title="Graph neural nets",
            authors=["Jane Doe"],
            source="local",
        ),
        content="This paper introduces message passing for molecular graphs.",
        sections=[],
    )
    context = ResearchContext(task=TaskType.ANALYZE_PAPER, papers=[paper])

    engine.run(TaskType.ANALYZE_PAPER, get_task_spec(TaskType.ANALYZE_PAPER), context)

    tool_results = [text for kind, text in reports if kind == "tool_result"]
    assert any("message passing for molecular graphs" in text for text in tool_results)


def test_non_instruction_tasks_report_internal_tool_loop_completion() -> None:
    reports: list[tuple[str, str]] = []

    class ReadyClient:
        def chat(self, messages, tools=None) -> dict:
            return {"message": {"role": "assistant", "content": "READY", "tool_calls": []}}

        def generate_structured(self, prompt: str, schema: dict) -> dict:
            return {
                "topic": "computer graphics",
                "results": [
                    {
                        "title": "Sample Paper",
                        "authors": ["Jane Doe"],
                        "why_relevant": "Matches the topic.",
                    }
                ],
            }

    def reporter(kind: str, text: str, end: str = "\n") -> None:
        reports.append((kind, text))

    engine = AnalysisEngine(ReadyClient(), max_iterations=1, reporter=reporter)
    context = ResearchContext(
        task=TaskType.FIND_PAPERS,
        topic="computer graphics",
        professors=["Ali Mahdavi Amiri"],
        papers=[
            PaperDocument(
                metadata=PaperMetadata(
                    paper_id="paper-1",
                    title="Sample Paper",
                    authors=["Jane Doe"],
                    source="arxiv",
                ),
                content="Short abstract.",
                sections=[],
            )
        ],
    )

    engine.run(TaskType.FIND_PAPERS, get_task_spec(TaskType.FIND_PAPERS), context)

    assert ("agent", "tool loop complete; proceeding to structured synthesis") in reports


def test_tool_loop_accepts_interactive_guidance_between_iterations() -> None:
    reports: list[tuple[str, str]] = []
    pending_guidance = ["Stop hyperparameter tuning and change the architecture in train.py."]

    class GuidanceClient:
        def __init__(self) -> None:
            self.chat_calls = 0

        def chat(self, messages, tools=None) -> dict:
            self.chat_calls += 1
            if self.chat_calls == 1:
                tool_call = {
                    "function": {
                        "name": "list_workspace_files",
                        "arguments": {"directory": ".", "pattern": "*"},
                    }
                }
                return {"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}
            assert any("change the architecture" in message.get("content", "") for message in messages)
            return {"message": {"role": "assistant", "content": "FINISHED", "tool_calls": []}}

        def generate_structured(self, prompt: str, schema: dict) -> dict:
            return {
                "request": "Tune the project until success.",
                "summary": "Processed guidance from the terminal.",
                "actions_taken": [],
                "key_findings": [],
                "hot_topics": [],
                "research_opportunities": [],
                "experiment_suggestions": [],
                "next_steps": [],
            }

    class WorkspaceStub:
        @staticmethod
        def list_files(directory: str = ".", pattern: str = "*") -> list[str]:
            return ["train.py", "config.json"]

    def reporter(kind: str, text: str, end: str = "\n") -> None:
        reports.append((kind, text))

    def interactive_guidance() -> list[str]:
        nonlocal pending_guidance
        guidance, pending_guidance = pending_guidance, []
        return guidance

    engine = AnalysisEngine(
        GuidanceClient(),
        workspace=WorkspaceStub(),
        max_iterations=2,
        reporter=reporter,
        interactive_guidance=interactive_guidance,
    )
    context = ResearchContext(task=TaskType.RUN_INSTRUCTIONS, instructions="Tune the project until success.")

    engine.run(TaskType.RUN_INSTRUCTIONS, get_task_spec(TaskType.RUN_INSTRUCTIONS), context)

    assert any("received user guidance" in text for kind, text in reports if kind == "agent")


def test_run_instructions_reprompt_restates_success_criteria() -> None:
    class CriteriaClient:
        def __init__(self) -> None:
            self.chat_calls = 0
            self.second_messages = []

        def chat(self, messages, tools=None) -> dict:
            self.chat_calls += 1
            if self.chat_calls == 1:
                return {"message": {"role": "assistant", "content": "I should inspect the logs next.", "tool_calls": []}}
            self.second_messages = messages
            return {"message": {"role": "assistant", "content": "FINISHED", "tool_calls": []}}

        def generate_structured(self, prompt: str, schema: dict) -> dict:
            return {
                "request": "Tune until success.",
                "summary": "Recovered after explicit criteria restatement.",
                "actions_taken": [],
                "key_findings": [],
                "hot_topics": [],
                "research_opportunities": [],
                "experiment_suggestions": [],
                "next_steps": [],
            }

    client = CriteriaClient()
    engine = AnalysisEngine(client, max_iterations=2)
    context = ResearchContext(
        task=TaskType.RUN_INSTRUCTIONS,
        instructions=(
            "Success criteria:\n"
            "- logs/latest.json must show success true\n"
            "- final_loss must be less than or equal to 0.005\n"
        ),
    )

    engine.run(TaskType.RUN_INSTRUCTIONS, get_task_spec(TaskType.RUN_INSTRUCTIONS), context)

    assert client.chat_calls == 2
    recovery_prompt = next(
        message["content"]
        for message in reversed(client.second_messages)
        if message.get("role") == "user" and "Active success criteria:" in message.get("content", "")
    )
    assert "logs/latest.json must show success true" in recovery_prompt
    assert "final_loss must be less than or equal to 0.005" in recovery_prompt


def test_instruction_state_update_summarizes_log_json() -> None:
    result = AnalysisEngine._instruction_state_update(
        tool_name="read_workspace_file",
        arguments={"path": "logs/latest.json"},
        result=json.dumps(
            {
                "success": False,
                "validation_accuracy": 0.7166,
                "target_accuracy": 0.88,
            }
        ),
    )

    assert "Read logs/latest.json." in result
    assert "success=False" in result
    assert "validation_accuracy=0.7166" in result
    assert "target_accuracy=0.88" in result


def test_anti_loop_instruction_blocks_repeated_log_reads() -> None:
    prompt = AnalysisEngine._anti_loop_instruction(
        tool_name="read_workspace_file",
        arguments={"path": "logs/latest.json"},
        repeat_count=2,
    )

    assert "Do not read the same log file again" in prompt
    assert "update_json_field" in prompt
    assert "run_uv_python" in prompt


def test_next_action_instruction_prefers_config_edits_after_failed_metrics() -> None:
    prompt = AnalysisEngine._next_action_instruction(
        tool_name="read_workspace_file",
        arguments={"path": "logs/latest.json"},
        result=json.dumps(
            {
                "success": False,
                "learning_rate": 0.003,
                "epochs": 100,
                "hidden_size": 24,
                "depth": 1,
                "dropout": 0.2,
                "weight_decay": 0.0,
                "validation_accuracy": 0.7166,
                "target_accuracy": 0.88,
            }
        ),
    )

    assert "Do not write a prose analysis" in prompt
    assert "Prefer `config.json` before `train.py`" in prompt
    assert "adjust `learning_rate`" in prompt
    assert "increase `epochs`" in prompt
    assert "run_uv_python" in prompt
