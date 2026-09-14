"""Controlled action schema for LLM-triggered portfolio navigation.

The LLM may only ever select a value from the closed `SectionId` enum via
the `navigate_to_section` tool. This module is the server-side authority
that re-validates any tool call the LLM returns — the LLM's own
schema-following is never trusted as enforcement.
"""

from typing import Any, Literal

from pydantic import BaseModel

SectionId = Literal[
    "home",
    "about",
    "experience",
    "ai_projects",
    "other_projects",
    "skills",
    "education",
    "certifications",
    "resume",
    "github",
    "contact",
]

VALID_SECTION_IDS: frozenset[str] = frozenset(SectionId.__args__)  # type: ignore[attr-defined]

NAVIGATE_TOOL_NAME = "navigate_to_section"

# Controlled, explicit synonym map. This is deliberately NOT fuzzy matching:
# an unmapped value (e.g. a hallucinated "skin") must fall through and be
# rejected by resolve_section_id, never guessed at.
SECTION_ALIASES: dict[str, str] = {
    "project": "ai_projects",
    "projects": "ai_projects",
    "ai project": "ai_projects",
    "ai_project": "ai_projects",
    "ai projects": "ai_projects",
    "other project": "other_projects",
    "other_project": "other_projects",
    "side project": "other_projects",
    "side_project": "other_projects",
    "side projects": "other_projects",
    "side_projects": "other_projects",
    "skill": "skills",
    "cv": "resume",
    "résumé": "resume",
    "resumes": "resume",
    "github profile": "github",
    "github_profile": "github",
    "repo": "github",
    "repos": "github",
    "repository": "github",
    "repositories": "github",
    "certification": "certifications",
    "certificate": "certifications",
    "certificates": "certifications",
    "education background": "education",
    "education_background": "education",
    "educations": "education",
    "background": "about",
    "bio": "about",
    "about me": "about",
    "about_me": "about",
    "contact me": "contact",
    "contact_me": "contact",
    "get in touch": "contact",
    "get_in_touch": "contact",
    "home page": "home",
    "home_page": "home",
    "homepage": "home",
    "landing page": "home",
    "landing_page": "home",
}


def resolve_section_id(raw: object) -> SectionId | None:
    """Map free-text (from an LLM tool call or a leaked JSON blob) to a
    canonical SectionId, or None if it can't be trusted.

    Accepts an already-canonical value ("ai_projects") case/whitespace
    variations of one, or a known synonym ("projects", "cv"). Anything else
    — including near-misses like "skin" — is rejected outright rather than
    guessed at, so an unrecognized value never reaches the frontend.
    """
    if not isinstance(raw, str):
        return None

    cleaned = raw.strip().lower()
    if not cleaned:
        return None

    underscored = cleaned.replace("-", "_").replace(" ", "_")
    if underscored in VALID_SECTION_IDS:
        return underscored  # type: ignore[return-value]

    if cleaned in SECTION_ALIASES:
        return SECTION_ALIASES[cleaned]  # type: ignore[return-value]
    if underscored in SECTION_ALIASES:
        return SECTION_ALIASES[underscored]  # type: ignore[return-value]

    return None


class NavigateAction(BaseModel):
    type: Literal["navigate_to_section"] = "navigate_to_section"
    target: SectionId


TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAVIGATE_TOOL_NAME,
        "description": (
            "Scroll the portfolio page to a specific section. Call this only "
            "when the user explicitly asks to see, go to, or show a section."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": sorted(VALID_SECTION_IDS),
                }
            },
            "required": ["target"],
        },
    },
}


def validate_action(tool_call: dict[str, Any] | None) -> NavigateAction | None:
    """Validate a raw tool call from the LLM against the closed action schema.

    Returns a `NavigateAction` only if the function name is recognized and
    the target resolves (directly or via a known synonym) to a canonical
    SectionId. Any mismatch (unknown function, unresolvable target, wrong
    types, missing fields) returns None so the caller can silently drop the
    action rather than forwarding untrusted data to the client.
    """
    if not tool_call:
        return None

    name = tool_call.get("name")
    if name != NAVIGATE_TOOL_NAME:
        return None

    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        return None

    target = resolve_section_id(arguments.get("target"))
    if target is None:
        return None

    return NavigateAction(target=target)  # type: ignore[arg-type]
