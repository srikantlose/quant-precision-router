"""Keep the Windows machine awake for the duration of a long-running
benchmark or analysis job, without touching any persistent power settings.

Uses the Win32 SetThreadExecutionState API: ES_CONTINUOUS tells Windows this
execution-state override should persist until explicitly cleared;
ES_SYSTEM_REQUIRED additionally resets the system idle timer, preventing
sleep. On exit, only ES_CONTINUOUS is set (clearing ES_SYSTEM_REQUIRED),
which releases the override -- the machine can sleep normally again as soon
as the `with` block exits, and nothing is left changed in Windows' own power
settings.
"""

from __future__ import annotations

import ctypes

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class keepawake:
    def __enter__(self) -> "keepawake":
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        return False
