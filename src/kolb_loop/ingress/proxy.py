"""OpenAI-compatible reverse proxy with experience logging and concept injection."""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from kolb_loop.adapter.openai_adapter import LLMAdapter
from kolb_loop.logger.experience_logger import ExperienceLogger

router = APIRouter()


def _extract_text_from_response(resp_json: dict[str, Any]) -> str:
    try:
        return str(resp_json["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError):
        return ""


def _build_response_message(resp_json: dict[str, Any]) -> dict[str, Any]:
    try:
        return dict(resp_json["choices"][0]["message"])
    except (KeyError, IndexError):
        return {}


class ProxyHandler:
    def __init__(
        self,
        adapter: LLMAdapter,
        logger: ExperienceLogger,
        fail_open: bool = True,
    ) -> None:
        self._adapter = adapter
        self._logger = logger
        self._fail_open = fail_open

    async def handle(self, request: Request) -> Response:
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            return Response(content='{"error":"invalid json"}', status_code=400)

        session_id = request.headers.get("x-session-id", str(uuid.uuid4()))
        messages: list[dict[str, Any]] = body.get("messages", [])
        model: str = body.get("model", "unknown")
        is_stream: bool = body.get("stream", False)

        if is_stream:
            return await self._handle_stream(body, session_id, messages, model)
        return await self._handle_non_stream(body, session_id, messages, model)

    async def _handle_non_stream(
        self,
        body: dict[str, Any],
        session_id: str,
        messages: list[dict[str, Any]],
        model: str,
    ) -> Response:
        try:
            resp_json = await self._adapter.chat_completions(body)
            self._logger.log(
                session_id=session_id,
                request_messages=messages,
                model=model,
                response_message=_build_response_message(resp_json),
                usage=resp_json.get("usage", {}),
            )
            return Response(
                content=json.dumps(resp_json),
                media_type="application/json",
            )
        except Exception as exc:
            self._logger.log(
                session_id=session_id,
                request_messages=messages,
                model=model,
                error=str(exc),
            )
            if self._fail_open:
                return Response(
                    content=json.dumps({"error": str(exc)}),
                    status_code=502,
                    media_type="application/json",
                )
            raise

    async def _handle_stream(
        self,
        body: dict[str, Any],
        session_id: str,
        messages: list[dict[str, Any]],
        model: str,
    ) -> Response:
        collected_chunks: list[bytes] = []

        async def _stream_generator() -> Any:
            try:
                async for chunk in self._adapter.chat_completions_stream(body):
                    collected_chunks.append(chunk)
                    yield chunk
                self._log_stream_completion(session_id, messages, model, collected_chunks)
            except Exception as exc:
                self._logger.log(
                    session_id=session_id,
                    request_messages=messages,
                    model=model,
                    error=str(exc),
                )
                if not self._fail_open:
                    raise

        return StreamingResponse(_stream_generator(), media_type="text/event-stream")

    def _log_stream_completion(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        model: str,
        chunks: list[bytes],
    ) -> None:
        # Reassemble the last complete JSON chunk for usage stats
        full_text = b"".join(chunks).decode("utf-8", errors="replace")
        # SSE lines start with "data: "; find the last non-[DONE] one
        lines = [line[6:] for line in full_text.splitlines() if line.startswith("data: ")]
        usage: dict[str, int] = {}
        last_content = ""
        for line in reversed(lines):
            if line.strip() == "[DONE]":
                continue
            try:
                obj: dict[str, Any] = json.loads(line)
                if "usage" in obj:
                    usage = obj["usage"]
                delta = obj.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    last_content = delta["content"]
                break
            except json.JSONDecodeError:
                continue

        self._logger.log(
            session_id=session_id,
            request_messages=messages,
            model=model,
            response_message={"role": "assistant", "content": last_content},
            usage=usage,
        )
