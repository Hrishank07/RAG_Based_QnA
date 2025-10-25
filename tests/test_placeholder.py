"""Basic smoke tests for the Codex repository."""

from codex.infra.app import create_app


def test_cdk_synthesizes():
    app = create_app()
    assembly = app.synth()
    assert any(stack.stack_name == "CodexStack" for stack in assembly.stacks)
