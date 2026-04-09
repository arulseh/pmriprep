"""Shared constants for configuration modules."""

# Named dose presets resolved to float multipliers
# Used by DCE, DSC configs, loader, and CLI
DOSE_PRESETS: dict[str, float] = {"half": 0.5, "full": 1.0, "double": 2.0}
