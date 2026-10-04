"""NAB Sentry: offline, CPU-only natural-language search over surveillance footage.

The offline environment is set first, before any other import, so that no model
library (torch, open_clip, huggingface_hub) can be imported with network access
enabled (Requirement 13.3). Keep this module free of heavy imports.
"""

from nab_sentry import startup as _startup  # stdlib-only module

_startup.enable_offline_mode()

__version__ = "0.1.0"
