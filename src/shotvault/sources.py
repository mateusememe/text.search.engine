"""Where screenshots come from. Every source is read-only: nothing is ever
deleted or modified in Photos, on the iPhone or in watched folders."""

from __future__ import annotations

import asyncio
import posixpath
import queue
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from shotvault.config import Config, FolderSource, IPhoneSource, PhotosSource
from shotvault.imaging import IMAGE_EXTENSIONS, has_screenshot_marker, looks_like_screenshot

Seen = Callable[[str, str], bool]


class SourceUnavailable(RuntimeError):
    """The source can't run here (not a Mac, no device plugged in, missing extra)."""


@dataclass
class Candidate:
    source: str
    ref: str  # stable id inside the source: Photos uuid, device path, file path
    name: str
    path: Path | None = None
    taken_at: datetime | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    temporary: bool = False  # delete `path` once ingested
    rejected: bool = False  # looked at, not a screenshot; remember and skip next time


class Folders:
    name = "folder"

    def __init__(self, cfg: FolderSource):
        self.cfg = cfg

    def scan(self, seen: Seen) -> Iterator[Candidate]:
        for raw in self.cfg.paths:
            root = Path(raw).expanduser()
            if not root.is_dir():
                continue
            files = root.rglob("*") if self.cfg.recursive else root.iterdir()
            for path in sorted(files):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS or path.name.startswith("."):
                    continue
                stat = path.stat()
                ref = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
                if seen(self.name, ref):
                    continue
                if self.cfg.only_screenshots and not looks_like_screenshot(path.name):
                    with path.open("rb") as handle:
                        head = handle.read(512 * 1024)
                    if not has_screenshot_marker(head):
                        yield Candidate(self.name, ref, path.name, rejected=True)
                        continue
                yield Candidate(
                    self.name,
                    ref,
                    path.name,
                    path=path,
                    taken_at=datetime.fromtimestamp(stat.st_mtime),
                    meta={"folder": str(root)},
                )


class Photos:
    """Apple Photos via osxphotos. Covers iCloud Photos: originals that were
    offloaded to iCloud are downloaded through Photos itself."""

    name = "photos"

    def __init__(self, cfg: PhotosSource, staging: Path):
        self.cfg = cfg
        self.staging = staging / "photos"

    def scan(self, seen: Seen) -> Iterator[Candidate]:
        if sys.platform != "darwin":
            raise SourceUnavailable("Apple Photos is only available on macOS")
        try:
            import osxphotos
        except ImportError as exc:
            raise SourceUnavailable("install the mac extra: pip install 'shotvault[mac]'") from exc

        since = datetime.fromisoformat(self.cfg.since) if self.cfg.since else None
        db = osxphotos.PhotosDB(dbfile=self.cfg.library) if self.cfg.library else osxphotos.PhotosDB()
        missing: dict[str, Any] = {}
        for photo in db.photos(images=True, movies=False, from_date=since):
            if not photo.screenshot or (photo.hidden and not self.cfg.include_hidden):
                continue
            if seen(self.name, photo.uuid):
                continue
            path = photo.path_edited if photo.hasadjustments and photo.path_edited else photo.path
            if path and Path(path).exists():
                yield self._candidate(photo, Path(path))
            else:
                missing[photo.uuid] = photo

        if missing and self.cfg.download_missing:
            yield from self._download(missing)

    def _candidate(self, photo: Any, path: Path, temporary: bool = False) -> Candidate:
        meta: dict[str, Any] = {"albums": list(photo.albums or [])}
        if photo.place and getattr(photo.place, "name", None):
            meta["place"] = photo.place.name
        lat, lon = photo.location
        if lat is not None:
            meta["location"] = [round(lat, 5), round(lon, 5)]
        for key in ("title", "description"):
            if getattr(photo, key, None):
                meta[key] = getattr(photo, key)
        if photo.keywords:
            meta["keywords"] = list(photo.keywords)
        if photo.favorite:
            meta["favorite"] = True
        taken = photo.date.astimezone().replace(tzinfo=None) if photo.date.tzinfo else photo.date
        return Candidate(self.name, photo.uuid, photo.original_filename, path, taken, meta, temporary)

    def _download(self, missing: dict[str, Any]) -> Iterator[Candidate]:
        """Ask Photos to export iCloud-only originals, 100 at a time."""
        self.staging.mkdir(parents=True, exist_ok=True)
        uuids = list(missing)
        for start in range(0, len(uuids), 100):
            batch = uuids[start : start + 100]
            uuid_file = self.staging / "uuids.txt"
            uuid_file.write_text("\n".join(batch))
            cmd = [
                sys.executable, "-m", "osxphotos", "export", str(self.staging),
                "--uuid-from-file", str(uuid_file), "--download-missing",
                "--filename", "{uuid}", "--skip-original-if-edited", "--skip-live",
                "--skip-bursts", "--no-progress", "--overwrite",
            ]  # fmt: skip
            if self.cfg.library:
                cmd += ["--library", self.cfg.library]
            if self.cfg.use_photokit:
                cmd.append("--use-photokit")
            subprocess.run(cmd, check=False, capture_output=True, text=True)
            uuid_file.unlink(missing_ok=True)
            for exported in sorted(self.staging.iterdir()):
                photo = missing.pop(exported.stem.split("_")[0], None)
                if photo is not None and exported.suffix.lower() in IMAGE_EXTENSIONS:
                    yield self._candidate(photo, exported, temporary=True)


class IPhone:
    """Reads DCIM over USB with pymobiledevice3 (unlock the phone and tap "Trust").

    iOS saves screenshots as PNG with a "Screenshot" marker in their metadata,
    so camera photos are never downloaded. With "Optimize iPhone Storage" on,
    only originals still on the device are visible; the Photos source covers the rest.
    """

    name = "iphone"

    def __init__(self, cfg: IPhoneSource, staging: Path):
        self.cfg = cfg
        self.staging = staging / "iphone"

    def scan(self, seen: Seen) -> Iterator[Candidate]:
        try:
            import pymobiledevice3  # noqa: F401
        except ImportError as exc:
            raise SourceUnavailable("install the iphone extra: pip install 'shotvault[iphone]'") from exc

        # pymobiledevice3 is async; run it in a thread and stream files through a
        # small queue so thousands of screenshots never sit in memory at once.
        items: queue.Queue = queue.Queue(maxsize=8)
        done = object()

        def worker() -> None:
            try:
                asyncio.run(self._pull(seen, items.put))
            except BaseException as exc:  # forwarded to the consumer
                items.put(exc)
            finally:
                items.put(done)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while (item := items.get()) is not done:
            if isinstance(item, BaseException):
                raise item
            yield item
        thread.join()

    async def _pull(self, seen: Seen, emit: Callable[[Candidate], None]) -> None:
        from pymobiledevice3.lockdown import create_using_usbmux
        from pymobiledevice3.services.afc import AfcService
        from pymobiledevice3.usbmux import list_devices

        devices = [d for d in await list_devices() if d.is_usb]
        if self.cfg.serial:
            devices = [d for d in devices if d.serial == self.cfg.serial]
        if not devices:
            raise SourceUnavailable("no iPhone connected over USB")
        serial = devices[0].serial
        self.staging.mkdir(parents=True, exist_ok=True)

        try:
            lockdown = await create_using_usbmux(serial=serial)
        except Exception as exc:  # not trusted yet, locked, pairing dialog pending...
            raise SourceUnavailable(
                f"couldn't open the iPhone ({type(exc).__name__}); unlock it, tap Trust, and run again"
            ) from exc
        async with lockdown:
            device_name = getattr(lockdown, "display_name", None) or "iPhone"
            async with AfcService(lockdown) as afc:
                for folder in sorted(await afc.listdir("/DCIM")):
                    if folder.startswith("."):
                        continue
                    base = posixpath.join("/DCIM", folder)
                    try:
                        names = sorted(await afc.listdir(base))
                    except Exception:
                        continue
                    for name in names:
                        if not name.lower().endswith(".png"):
                            continue
                        remote = posixpath.join(base, name)
                        info = await afc.stat(remote)
                        ref = f"{serial}:{remote}:{info['st_size']}"
                        if seen(self.name, ref):
                            continue
                        data = await afc.get_file_contents(remote)
                        if not (self.cfg.accept_all_png or has_screenshot_marker(data)):
                            emit(Candidate(self.name, ref, name, rejected=True))
                            continue
                        local = self.staging / f"{serial[-8:]}_{folder}_{name}"
                        local.write_bytes(data)
                        mtime = info.get("st_birthtime") or info.get("st_mtime")
                        emit(
                            Candidate(
                                self.name,
                                ref,
                                name,
                                path=local,
                                taken_at=mtime if isinstance(mtime, datetime) else None,
                                meta={"device": device_name},
                                temporary=True,
                            )
                        )


def iphone_connected() -> bool:
    try:
        from pymobiledevice3.usbmux import list_devices
    except ImportError:
        return False
    try:
        return any(d.is_usb for d in asyncio.run(list_devices()))
    except Exception:
        return False


def make_sources(cfg: Config, only: list[str] | None = None) -> list[Folders | Photos | IPhone]:
    wanted = set(only or ["photos", "iphone", "folder"])
    sources: list[Folders | Photos | IPhone] = []
    if "photos" in wanted and (cfg.sources.photos.enabled or only):
        sources.append(Photos(cfg.sources.photos, cfg.staging_dir))
    if "iphone" in wanted and (cfg.sources.iphone.enabled or only):
        sources.append(IPhone(cfg.sources.iphone, cfg.staging_dir))
    if "folder" in wanted and (cfg.sources.folders.enabled or only):
        sources.append(Folders(cfg.sources.folders))
    return sources


def clear_staging(cfg: Config) -> None:
    shutil.rmtree(cfg.staging_dir, ignore_errors=True)
