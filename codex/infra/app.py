"""Entry point for the Codex CDK application."""

from aws_cdk import App, Environment

from codex.infra.codex_stack import CodexStack


def create_app() -> App:
    """Create and return the CDK application."""
    app = App()

    CodexStack(
        app,
        "CodexStack",
        env=Environment(
            account=app.node.try_get_context("account"),
            region=app.node.try_get_context("region"),
        ),
    )
    return app


def main() -> None:
    """Synthesize the CDK app when executed as a script."""
    app = create_app()
    app.synth()


if __name__ == "__main__":
    main()
