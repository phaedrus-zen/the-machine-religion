"""Isolation for ms4_gateway tests.

The gateway holds a few process-global caches: the grounding cache
(``machine_spirit_4.gateway.context``), the foreground model picker
cache (``machine_spirit_4.double_agent.model_picker``), etc. Without
explicit cleanup, tests that touch ``build_grounded_user_message`` or
``choose_foreground_model`` leak state into the next test in the file
or in a sibling file — observed live as ``test_tools_question_routes_
through_tools_grounding`` failing only when run after the grounding
cache suite because the suite leaves the tools entry warm.

Clear those caches autouse before every test so behavior is hermetic.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_gateway_caches():
    from machine_spirit_4.gateway import context as ctx_module
    from machine_spirit_4.double_agent import model_picker as picker_module

    ctx_module.clear_grounding_cache()
    if hasattr(picker_module, "_clear_cache_for_tests"):
        picker_module._clear_cache_for_tests()
    yield
    ctx_module.clear_grounding_cache()
    if hasattr(picker_module, "_clear_cache_for_tests"):
        picker_module._clear_cache_for_tests()
