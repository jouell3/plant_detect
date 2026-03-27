import os
import time
from pathlib import Path

from loguru import logger

_CACHE_MAX_AGE_SECONDS = int(os.getenv("GCS_CACHE_MAX_AGE_SECONDS", str(3 * 3600)))


def is_cache_valid(local_dir: Path, filenames: list[str], max_age: int = _CACHE_MAX_AGE_SECONDS) -> bool:
    """Return True if all named files exist in local_dir and are fresher than max_age seconds."""
    if not local_dir.exists():
        return False
    now = time.time()
    for name in filenames:
        p = local_dir / name
        if not p.exists():
            logger.debug("Cache miss: {} not found.", p)
            return False
        age = now - p.stat().st_mtime
        if age > max_age:
            logger.debug("Cache expired: {} is {:.0f}s old (max {}s).", p, age, max_age)
            return False
    logger.info("GCS cache hit for {} — skipping download.", local_dir)
    return True


def is_cache_valid_by_patterns(
    local_dir: Path, patterns: list[str], max_age: int = _CACHE_MAX_AGE_SECONDS
) -> bool:
    """Return True if for each glob pattern at least one matching file exists and is fresh."""
    if not local_dir.exists():
        return False
    now = time.time()
    for pattern in patterns:
        matches = sorted(local_dir.glob(pattern))
        if not matches:
            logger.debug("Cache miss: no file matching '{}' in {}.", pattern, local_dir)
            return False
        newest = matches[-1]
        age = now - newest.stat().st_mtime
        if age > max_age:
            logger.debug("Cache expired: {} is {:.0f}s old (max {}s).", newest, age, max_age)
            return False
    logger.info("GCS cache hit for {} — skipping download.", local_dir)
    return True
