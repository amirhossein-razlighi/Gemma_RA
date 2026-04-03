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

    assert client.chat_calls == 3
