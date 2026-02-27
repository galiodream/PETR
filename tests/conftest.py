from pathlib import Path

import _pytest._code.code as pytest_code
import _pytest.pathlib as pytest_pathlib


_original_bestrelpath = pytest_pathlib.bestrelpath


def _safe_bestrelpath(base: Path, dest: Path):
    try:
        return _original_bestrelpath(base, dest)
    except ValueError:
        return str(dest)


pytest_pathlib.bestrelpath = _safe_bestrelpath
pytest_code.bestrelpath = _safe_bestrelpath
