"""Customer Agent — AgentExecutor bridge between A2A SDK and LangGraph."""

from __future__ import annotations

import logging
import os
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart

from customer_agent.graph import build_graph

logger = logging.getLogger(__name__)


class CustomerAgentExecutor(AgentExecutor):
    """Bridges A2A RequestContext to the Customer LangGraph agent."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        question = self._extract_question(context)
        context_id = context.context_id or str(uuid4())
        task_id = context.task_id or str(uuid4())

        # Propagate or generate trace metadata
        metadata = context.message.metadata or {} if context.message else {}
        trace_id = metadata.get("trace_id", str(uuid4()))
        depth = int(metadata.get("delegation_depth", 0))

        logger.info(
            "CustomerAgent executing | task=%s context=%s trace=%s depth=%d",
            task_id, context_id, trace_id, depth,
        )

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.submit()
        await updater.start_work()

        try:
            if self._should_delegate_directly():
                answer = await self._delegate_directly(question, context_id, trace_id, depth)
            else:
                answer = await self._run_customer_agent(question, context_id, trace_id, depth)

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=answer))],
                name="legal_response",
            )
            await updater.complete()

        except Exception as exc:
            logger.exception("CustomerAgent execution error: %s", exc)
            await updater.failed(
                updater.new_agent_message(
                    parts=[Part(root=TextPart(text=f"Request failed: {exc}"))]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()

    async def _run_customer_agent(
        self,
        question: str,
        context_id: str,
        trace_id: str,
        depth: int,
    ) -> str:
        """Run the original Customer ReAct agent from the codelab."""
        graph = build_graph(
            trace_id=trace_id,
            context_id=context_id,
            depth=depth,
        )

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": context_id}},
        )

        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content

        for msg in reversed(result.get("messages", [])):
            content = getattr(msg, "content", "")
            if content and not isinstance(msg, HumanMessage):
                return content

        return "I was unable to process your legal question at this time."

    async def _delegate_directly(
        self,
        question: str,
        context_id: str,
        trace_id: str,
        depth: int,
    ) -> str:
        """Bypass Customer LLM tool-calling for free models that skip tool use."""
        from common.a2a_client import delegate
        from common.registry_client import discover

        logger.info(
            "CustomerAgent delegating directly | trace=%s context=%s depth=%d",
            trace_id, context_id, depth,
        )
        endpoint = await discover("legal_question")
        answer = await delegate(
            endpoint=endpoint,
            question=question,
            context_id=context_id,
            trace_id=trace_id,
            depth=depth + 1,
        )
        return answer or "The Law Agent returned an empty response. Please try again."

    @staticmethod
    def _should_delegate_directly() -> bool:
        direct_flag = os.getenv("CUSTOMER_DIRECT_DELEGATE", "").lower()
        if direct_flag in {"1", "true", "yes", "on"}:
            return True
        if direct_flag in {"0", "false", "no", "off"}:
            return False
        return os.getenv("OPENROUTER_MODEL", "").lower() == "openrouter/free"

    @staticmethod
    def _extract_question(context: RequestContext) -> str:
        if context.message and context.message.parts:
            parts = []
            for part in context.message.parts:
                inner = getattr(part, "root", part)
                text = getattr(inner, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return ""
