"""macOS integrations: nightly launchd job, Photos album for cleanup, git commits."""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from shotvault.config import Config

LABEL = "com.shotvault.nightly"
PLIST = Path("~/Library/LaunchAgents").expanduser() / f"{LABEL}.plist"


def shotvault_command() -> list[str]:
    exe = shutil.which("shotvault")
    return [exe] if exe else [sys.executable, "-m", "shotvault"]


def install_schedule(cfg: Config, config_file: Path, hour: int, minute: int) -> Path:
    """Run `shotvault run` daily. `caffeinate -i` keeps the Mac awake while a big
    backlog is processed; a run missed while asleep starts on the next wake."""
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    path_env = os.pathsep.join(["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"])
    plist = {
        "Label": LABEL,
        "ProgramArguments": ["/usr/bin/caffeinate", "-i", *shotvault_command(), "run", "--quiet"],
        "StartCalendarInterval": {"Hour": hour, "Minute": minute},
        "EnvironmentVariables": {"PATH": path_env, "SHOTVAULT_CONFIG": str(config_file.resolve())},
        "StandardOutPath": str(cfg.logs_dir / "nightly.log"),
        "StandardErrorPath": str(cfg.logs_dir / "nightly.log"),
        "ProcessType": "Background",
        "LowPriorityIO": True,
    }
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    with PLIST.open("wb") as handle:
        plistlib.dump(plist, handle)
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", domain, str(PLIST)], capture_output=True)
    subprocess.run(["launchctl", "bootstrap", domain, str(PLIST)], check=True)
    return PLIST


def uninstall_schedule() -> bool:
    if not PLIST.exists():
        return False
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(PLIST)], capture_output=True)
    PLIST.unlink()
    return True


def add_to_album(uuids: list[str], album_name: str) -> int:
    """Put Photos items in an album (it syncs to the iPhone via iCloud) so they
    can be reviewed and deleted in one go. Nothing is deleted here."""
    import photoscript

    library = photoscript.PhotosLibrary()
    album = library.album(album_name) or library.create_album(album_name)
    added = 0
    for start in range(0, len(uuids), 200):
        added += len(album.add([photoscript.Photo(uuid) for uuid in uuids[start : start + 200]]))
    return added


def git_commit(vault: Path, folder: str, message: str, push: bool = False) -> bool:
    if not (vault / ".git").exists():
        return False

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(vault), *args], capture_output=True, text=True)

    git("add", "--", folder)
    if git("diff", "--cached", "--quiet", "--", folder).returncode == 0:
        return False
    # Path-limited commit: never sweeps in other changes you have staged.
    if git("commit", "-m", message, "--", folder).returncode != 0:
        return False
    if push:
        git("push")
    return True
