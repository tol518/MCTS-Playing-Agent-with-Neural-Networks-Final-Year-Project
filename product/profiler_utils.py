import cProfile
import os
import pstats
import sys
import time
from typing import Callable, Optional, Any


def _profile_enabled() -> bool:
    env_flag = os.getenv("GO_PROFILE", "").strip().lower()
    if env_flag in {"1", "true", "yes", "on"}:
        return True
    return "--profile" in sys.argv


def _profile_dir() -> str:
    env_dir = os.getenv("GO_PROFILE_DIR", "").strip()
    if env_dir:
        return env_dir
    for arg in sys.argv:
        if arg.startswith("--profile-dir="):
            return arg.split("=", 1)[1]
    return os.path.join(os.path.dirname(__file__), "profiles")


def _safe_label(label: str) -> str:
    cleaned = []
    for char in label.strip().replace(" ", "_"):
        if char.isalnum() or char in {"_", "-"}:
            cleaned.append(char)
    return "".join(cleaned) or "profile"


def run_with_profiler(label: str, func: Callable[..., Any], *args, **kwargs) -> Any:
    if not _profile_enabled():
        return func(*args, **kwargs)

    profile_dir = _profile_dir()
    os.makedirs(profile_dir, exist_ok=True)

    safe_label = _safe_label(label)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    profile_path = os.path.join(profile_dir, f"{safe_label}_{timestamp}.prof")
    summary_path = os.path.join(profile_dir, f"{safe_label}_{timestamp}.txt")

    print(f"[Profiler] Enabled. Writing results to: {profile_dir}")

    profiler = cProfile.Profile()
    try:
        result = profiler.runcall(func, *args, **kwargs)
    finally:
        profiler.dump_stats(profile_path)
        with open(summary_path, "w", encoding="utf-8") as summary:
            stats = pstats.Stats(profiler, stream=summary)
            stats.strip_dirs().sort_stats("cumulative")
            stats.print_stats(40)
    return result
