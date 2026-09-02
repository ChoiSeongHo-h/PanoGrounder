from panogrounder.cli import build_parser


def test_cli_does_not_expose_model_hyperparameters() -> None:
    help_text = build_parser().format_help()
    assert "lora" not in help_text.lower()
    assert "visibility_power" not in help_text.lower()
    assert "loss" not in help_text.lower()
