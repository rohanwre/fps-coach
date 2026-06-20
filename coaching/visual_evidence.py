"""Extract structured coaching evidence from player-perspective frames."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import pathlib
import urllib.error
import urllib.request
from typing import Any, Protocol


SYSTEM_PROMPT = """You are an FPS coaching evidence extractor.
Analyze only what is visibly supported by the player-perspective images.
Report all boolean judgments for the final image. Earlier images provide context
only. The yellow plus sign is the crosshair and the red capsule is the enemy.
Return one JSON object with:
target_visible (boolean), crosshair_near_target (boolean or null),
player_exposed (boolean or null), tags (array of short snake_case strings),
summary (one factual sentence), recommended_focus (one short coaching focus),
confidence (number from 0 to 1).
Do not invent hidden enemies, intent, or game state."""

VISUAL_EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "target_visible": {"type": "boolean"},
        "crosshair_near_target": {"type": ["boolean", "null"]},
        "player_exposed": {"type": ["boolean", "null"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "recommended_focus": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "target_visible",
        "crosshair_near_target",
        "player_exposed",
        "tags",
        "summary",
        "recommended_focus",
        "confidence",
    ],
    "additionalProperties": False,
}


class VisualEvidenceProvider(Protocol):
    def analyze(self, frame_paths: list[str], event_window: dict[str, Any]) -> dict[str, Any]:
        ...


class StructuredEventFallback:
    """Deterministic evidence used when no VLM endpoint is configured."""

    def analyze(self, frame_paths: list[str], event_window: dict[str, Any]) -> dict[str, Any]:
        events = event_window.get("events", [])
        center_event_type = event_window.get("center_event_type", "")
        player_damage_events = [
            event
            for event in events
            if event.get("event_type") == "actor_damaged"
            and event.get("payload", {}).get("actor_id") == "player"
        ]
        player_misses = [
            event
            for event in events
            if event.get("event_type") == "shot_missed"
            and event.get("payload", {}).get("shooter_id") == "player"
        ]

        tags: list[str] = []
        if player_damage_events:
            tags.append("player_took_damage")
        if player_misses:
            tags.append("player_missed_shots")
        if center_event_type == "round_ended":
            tags.append("round_end_context")
        tags.append("frames_available" if frame_paths else "no_frames_available")

        return {
            "evidence_source": "structured_event_fallback",
            "frame_count": len(frame_paths),
            "event_context_included": False,
            "target_visible": None,
            "crosshair_near_target": None,
            "player_exposed": None,
            "tags": tags,
            "summary": _summarize_tags(tags),
            "recommended_focus": _focus_from_tags(tags),
            "confidence": 0.45 if tags else 0.1,
        }


class OpenAICompatibleVLM:
    """Minimal dependency-free client for OpenAI-compatible vision endpoints."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout_sec: float = 60.0,
        include_event_context: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.include_event_context = include_event_context

    @classmethod
    def from_environment(cls, include_event_context: bool = False) -> "OpenAICompatibleVLM":
        endpoint = os.environ["COACH_VLM_ENDPOINT"]
        model = os.environ["COACH_VLM_MODEL"]
        return cls(
            endpoint,
            model,
            os.environ.get("COACH_VLM_API_KEY"),
            include_event_context=include_event_context,
        )

    def analyze(self, frame_paths: list[str], event_window: dict[str, Any]) -> dict[str, Any]:
        if not frame_paths:
            return StructuredEventFallback().analyze(frame_paths, event_window)

        selected_frame_paths = frame_paths[-3:]
        prompt = "Analyze these player-perspective frames."
        if self.include_event_context:
            prompt += f" The center gameplay event is {event_window.get('center_event_type', 'unknown')}."
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame_path in selected_frame_paths:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(frame_path)}})

        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "fps_visual_evidence",
                    "strict": True,
                    "schema": VISUAL_EVIDENCE_SCHEMA,
                },
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM endpoint returned HTTP {exc.code}: {detail}") from exc

        message = result["choices"][0]["message"]["content"]
        evidence = json.loads(message) if isinstance(message, str) else message
        evidence["evidence_source"] = f"vlm:{self.model}"
        evidence["frame_count"] = len(selected_frame_paths)
        evidence["analyzed_frame_path"] = selected_frame_paths[-1]
        evidence["event_context_included"] = self.include_event_context
        return evidence


def extract_visual_evidence(
    frame_paths: list[str],
    event_window: dict[str, Any],
    provider: VisualEvidenceProvider | None = None,
) -> dict[str, Any]:
    selected_provider = provider or _provider_from_environment()
    return selected_provider.analyze(frame_paths, event_window)


def _provider_from_environment() -> VisualEvidenceProvider:
    if os.environ.get("COACH_VLM_ENDPOINT") and os.environ.get("COACH_VLM_MODEL"):
        return OpenAICompatibleVLM.from_environment()
    return StructuredEventFallback()


def _image_data_url(frame_path: str) -> str:
    path = pathlib.Path(frame_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _summarize_tags(tags: list[str]) -> str:
    if "player_took_damage" in tags and "player_missed_shots" in tags:
        return "Player took damage near missed shots; review positioning and shot setup."
    if "player_took_damage" in tags:
        return "Player took damage in this window; review exposure, cover, and timing."
    if "player_missed_shots" in tags:
        return "Player missed shots in this window; review crosshair placement and pacing."
    return "No strong coaching evidence extracted from structured events yet."


def _focus_from_tags(tags: list[str]) -> str:
    if "player_took_damage" in tags:
        return "Use cover and reduce exposure before committing to the duel."
    if "player_missed_shots" in tags:
        return "Stabilize crosshair placement before firing."
    return "Review positioning and decision timing."
