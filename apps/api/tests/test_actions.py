from core.actions import NAVIGATE_TOOL_NAME, NavigateAction, resolve_section_id, validate_action


def _tool_call(name: str, arguments: object) -> dict:
    return {"name": name, "arguments": arguments}


def test_valid_target_is_accepted():
    result = validate_action(_tool_call(NAVIGATE_TOOL_NAME, {"target": "experience"}))
    assert result == NavigateAction(target="experience")


def test_none_tool_call_returns_none():
    assert validate_action(None) is None


def test_unknown_function_name_is_rejected():
    assert validate_action(_tool_call("delete_everything", {"target": "home"})) is None


def test_unknown_target_is_rejected():
    assert validate_action(_tool_call(NAVIGATE_TOOL_NAME, {"target": "not-a-section"})) is None


def test_path_traversal_target_is_rejected():
    assert (
        validate_action(_tool_call(NAVIGATE_TOOL_NAME, {"target": "../etc/passwd"})) is None
    )


def test_script_injection_target_is_rejected():
    assert (
        validate_action(
            _tool_call(NAVIGATE_TOOL_NAME, {"target": "<script>alert(1)</script>"})
        )
        is None
    )


def test_javascript_uri_target_is_rejected():
    assert (
        validate_action(_tool_call(NAVIGATE_TOOL_NAME, {"target": "javascript:alert(1)"}))
        is None
    )


def test_empty_string_target_is_rejected():
    assert validate_action(_tool_call(NAVIGATE_TOOL_NAME, {"target": ""})) is None


def test_missing_target_key_is_rejected():
    assert validate_action(_tool_call(NAVIGATE_TOOL_NAME, {})) is None


def test_non_string_target_is_rejected():
    assert validate_action(_tool_call(NAVIGATE_TOOL_NAME, {"target": 123})) is None


def test_non_dict_arguments_is_rejected():
    assert validate_action(_tool_call(NAVIGATE_TOOL_NAME, ["home"])) is None
    assert validate_action(_tool_call(NAVIGATE_TOOL_NAME, "home")) is None
    assert validate_action(_tool_call(NAVIGATE_TOOL_NAME, None)) is None


def test_known_synonym_maps_to_canonical_section():
    result = validate_action(_tool_call(NAVIGATE_TOOL_NAME, {"target": "projects"}))
    assert result == NavigateAction(target="ai_projects")


def test_synonym_resolution_is_case_and_whitespace_insensitive():
    assert resolve_section_id("Projects") == "ai_projects"
    assert resolve_section_id("  projects  ") == "ai_projects"
    assert resolve_section_id("AI-Projects") == "ai_projects"
    assert resolve_section_id("CV") == "resume"
    assert resolve_section_id("repos") == "github"


def test_unmapped_near_miss_is_rejected_not_guessed():
    # "skin" must never be silently corrected to "skills" - only exact
    # canonical values or explicit, known synonyms resolve.
    assert resolve_section_id("skin") is None
    assert validate_action(_tool_call(NAVIGATE_TOOL_NAME, {"target": "skin"})) is None


def test_every_declared_section_id_is_accepted():
    sections = [
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
    for section in sections:
        result = validate_action(_tool_call(NAVIGATE_TOOL_NAME, {"target": section}))
        assert result is not None
        assert result.target == section
