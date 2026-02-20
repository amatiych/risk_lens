"""SSE streaming chat endpoint."""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.schemas import ChatRequest
from backend.api.dependencies import store, get_provider
from backend.llm.tools import TOOLS, execute_tool, set_portfolio_context
from backend.llm.providers import LLMConfig, get_provider as get_llm_provider
from backend.llm.guardrails import GuardrailsReport
from backend.llm.guardrails.input_guards import InputGuardrails
from backend.llm.guardrails.output_guards import OutputGuardrails
from backend.llm.guardrails.process_guards import ProcessGuardrails
from frontend.services.chat_service import get_system_prompt, get_guardrails_summary

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _stream_chat(messages, report, provider_name):
    """Stream chat response using the LLM provider, independent of Streamlit."""
    set_portfolio_context(report.portfolio)
    config = LLMConfig(provider=provider_name)
    provider = get_llm_provider(config)
    system_prompt = get_system_prompt(report)

    conversation = list(messages)
    max_tool_iterations = 5

    for _ in range(max_tool_iterations):
        response = provider.create_message(
            system_prompt=system_prompt,
            messages=conversation,
            tools=TOOLS,
            max_tokens=2000,
        )

        if response.stop_reason == "tool_use":
            tool_results = []
            tool_names = []
            for tool_call in response.tool_calls:
                tool_names.append(tool_call.name)
                try:
                    result = execute_tool(tool_call.name, tool_call.arguments)
                    tool_results.append(
                        provider.format_tool_result(tool_call.id, json.dumps(result))
                    )
                except Exception as e:
                    tool_results.append(
                        provider.format_tool_result(
                            tool_call.id, json.dumps({"error": str(e)}), is_error=True
                        )
                    )

            yield f"[Looking up: {', '.join(tool_names)}...]\n\n"

            conversation.append(
                provider.format_assistant_message(response.content, response.tool_calls)
            )
            conversation.append({"role": "user", "content": tool_results})
        else:
            if response.content:
                yield response.content
            break


@router.post("/{portfolio_id}")
async def chat_stream(portfolio_id: str, request: ChatRequest):
    entry = store.get(portfolio_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if not entry.report:
        raise HTTPException(status_code=400, detail="Portfolio not yet analyzed")

    provider_name = get_provider()

    def generate():
        guardrails_report = GuardrailsReport()

        input_guards = InputGuardrails()
        input_results = input_guards.run_all_checks(request.message)
        for r in input_results:
            guardrails_report.add_input_check(r)

        if guardrails_report.blocked:
            yield _sse_event({"type": "text", "content": "I cannot process this request due to safety guardrails."})
            summary = get_guardrails_summary(guardrails_report)
            yield _sse_event({"type": "done", "guardrails": summary})
            return

        process_guards = ProcessGuardrails()
        process_results = process_guards.pre_request_checks("api")
        for r in process_results:
            guardrails_report.add_process_check(r)

        if guardrails_report.blocked:
            yield _sse_event({"type": "text", "content": "Rate limit exceeded. Please try again later."})
            summary = get_guardrails_summary(guardrails_report)
            yield _sse_event({"type": "done", "guardrails": summary})
            return

        full_messages = request.history + [{"role": "user", "content": request.message}]
        full_response = ""

        try:
            for chunk in _stream_chat(full_messages, entry.report, provider_name):
                full_response += chunk
                yield _sse_event({"type": "text", "content": chunk})
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield _sse_event({"type": "text", "content": f"\n\nError: {type(e).__name__}: {str(e)}"})

        output_guards = OutputGuardrails()
        source_data = {
            "portfolio": entry.report.holdings_json,
            "var": entry.report.var_json,
            "factors": entry.report.factor_data,
        }
        _, output_results = output_guards.run_all_checks(full_response, source_data)
        for r in output_results:
            guardrails_report.add_output_check(r)

        summary = get_guardrails_summary(guardrails_report)
        yield _sse_event({"type": "done", "guardrails": summary})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
