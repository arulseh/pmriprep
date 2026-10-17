"""Keep shared data directories and host-native binaries out of containers."""

from pathlib import Path

def test_container_excludes_shared_generated_material() -> None:
    rules = [line.strip() for line in
             (Path(__file__).resolve().parents[2] / '.dockerignore').read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    assert {'work/', 'sourcedata/', '**/*.so', '**/*.pyd', '**/*.dylib'} <= set(rules)
    assert not any(rule.startswith('!') for rule in rules)
