"""Keep shared data directories and host-native binaries out of containers."""

from pathlib import Path

import pytest


@pytest.mark.parametrize('pattern', [
    'work/', 'sourcedata/', '**/*.so', '**/*.pyd', '**/*.dylib',
])
def test_container_excludes_shared_generated_material(pattern: str) -> None:
    rules = [line.strip() for line in
             (Path(__file__).resolve().parents[2] / '.dockerignore').read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    assert pattern in rules
    assert not any(rule.startswith('!') for rule in rules)
