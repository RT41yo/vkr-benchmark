import json
from pathlib import Path

from vkr_benchmark.config import LocalModelConfig


def test_model_config_resolves_local_path_from_project_root(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "models"
    config_dir.mkdir(parents=True)
    path = config_dir / "model.json"
    path.write_text(
        json.dumps(
            {
                "id": "example/model",
                "revision": "abc123",
                "local_path": "models/example",
                "dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )

    config = LocalModelConfig.from_json(path)
    assert config.model_id == "example/model"
    assert config.local_path == (tmp_path / "models" / "example").resolve()
    assert config.prompt_add_special_tokens is False
