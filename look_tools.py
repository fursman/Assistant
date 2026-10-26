"""The `look` tool: a screenshot, a webcam frame or an image file, made small
enough to send to a vision model.

voice_assistant.py offers `look` to the native tool loop only when the model
answering advertises vision (llama-server's GET /props, "modalities"), and
`assistant --image FILE` uses the same file check and downscale. Everything
here is capture and conversion; deciding when to offer it lives there.

Nothing is kept on disk: a capture goes to a private temporary directory that
is gone by the time the image has been encoded, and only the JPEG's bytes
leave this module. Errors are LookError, whose message is written for the
model to read out ("there is no camera on this machine"), not a traceback.

Only the standard library; the resizing is ffmpeg's (ImageMagick when ffmpeg
is missing), so no imaging package is pulled into the venv.
"""
import base64
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

# Longest side of the image sent, in pixels. Prompt tokens grow with the area:
# a 1280x720 screenshot was ~1000 prompt tokens and 28 s on the network
# Nemotron, 768x432 ~400 and 14 s -- but 1280 is what still reads a 1080p
# screen's small text.
LOOK_MAX_SIDE = max(64, int(os.getenv("VOICE_ASSISTANT_LOOK_MAX_SIDE", "1280")))
# ffmpeg's JPEG quality scale: 2 (best) .. 31 (worst). 4 is visually clean at
# about a third of the size of 2.
LOOK_JPEG_QUALITY = 4
# "focused" = the monitor with the focused window (Hyprland, Sway); "all" =
# every output side by side, which on a two-monitor desk halves the detail.
LOOK_SCREEN = os.getenv("VOICE_ASSISTANT_LOOK_SCREEN", "focused").strip().lower()
# Which camera: empty = the first capture device; a /dev/video* path; or a
# shell command that writes one frame to {file} (e.g. "cam look {file}").
LOOK_CAMERA = os.getenv("VOICE_ASSISTANT_LOOK_CAMERA", "").strip()
# Image files bigger than this are refused rather than decoded.
LOOK_MAX_FILE_MB = float(os.getenv("VOICE_ASSISTANT_LOOK_MAX_FILE_MB", "40"))
# One capture or conversion may take this long.
LOOK_TIMEOUT = float(os.getenv("VOICE_ASSISTANT_LOOK_TIMEOUT", "20"))
# Frames the camera runs before the one kept: the first few after opening are
# dark while auto-exposure settles (about a third of a second at 30 fps).
CAMERA_WARMUP_FRAMES = 10
CAMERA_SIZE = "1280x720"

SOURCES = ("screen", "camera", "file")

_LOG = logging.getLogger("voice-assistant.look")

# What a file has to start with to be an image ffmpeg will read. The name is
# not trusted: "photo.png" that is a text file is refused here, not by a
# decoder error the model cannot explain.
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"BM", "BMP"),
    (b"II*\x00", "TIFF"),
    (b"MM\x00*", "TIFF"),
)
FORMATS = "PNG, JPEG, GIF, WebP, BMP or TIFF"

# The xdg-desktop-portal screenshot, for GNOME (and any desktop with a
# portal but no grim). Run by the system python, which has PyGObject; the
# venv does not. The portal answers on a Request object's Response signal, so
# a one-shot gdbus call cannot wait for it. With interactive=false GNOME asks
# once for permission and remembers it; until then nothing comes back.
_PORTAL_SCRIPT = r"""
import sys, uuid
from gi.repository import Gio, GLib
bus = Gio.bus_get_sync(Gio.BusType.SESSION)
token = "va" + uuid.uuid4().hex[:8]
sender = bus.get_unique_name()[1:].replace(".", "_")
path = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"
loop, out = GLib.MainLoop(), {}
def done(_c, _s, _p, _i, _n, params):
    code, res = params.unpack()
    out["code"], out["uri"] = code, (res.get("uri", "") if code == 0 else "")
    loop.quit()
bus.signal_subscribe("org.freedesktop.portal.Desktop", "org.freedesktop.portal.Request",
                     "Response", path, None, Gio.DBusSignalFlags.NO_MATCH_RULE, done)
bus.call_sync("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop",
              "org.freedesktop.portal.Screenshot", "Screenshot",
              GLib.Variant("(sa{sv})", ("", {"handle_token": GLib.Variant("s", token),
                                             "interactive": GLib.Variant("b", False)})),
              None, Gio.DBusCallFlags.NONE, 10000, None)
GLib.timeout_add_seconds(int(sys.argv[1]), loop.quit)
loop.run()
if not out.get("uri"):
    sys.exit("response %s" % out.get("code", "timeout"))
print(out["uri"])
"""


class LookError(Exception):
    """Why there is no image, in words the model can pass on."""


def _run(cmd, timeout=None, **kw):
    """subprocess.run with output captured; a missing binary or a timeout is a
    LookError. Returns the CompletedProcess whatever its exit code."""
    timeout = timeout or LOOK_TIMEOUT
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, **kw)
    except FileNotFoundError:
        raise LookError(f"{cmd[0]} is not installed")
    except subprocess.TimeoutExpired:
        raise LookError(f"{Path(cmd[0]).name} took longer than {timeout:.0f} seconds")


def _err(proc) -> str:
    """The last line of a failed command's stderr, for the message."""
    lines = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
    return lines[-1][:200] if lines else f"exit code {proc.returncode}"


# --- screen ---------------------------------------------------------------

def _focused_output(desktop: str) -> str:
    """The name of the monitor with the focused window, or "" (then grim
    takes every output)."""
    try:
        if "Hyprland" in desktop:
            proc = _run(["hyprctl", "-j", "monitors"], timeout=3)
            outputs = json.loads(proc.stdout or b"[]")
        elif "Sway" in desktop:
            proc = _run(["swaymsg", "-t", "get_outputs", "-r"], timeout=3)
            outputs = json.loads(proc.stdout or b"[]")
        else:
            return ""
        return next((str(o.get("name") or "") for o in outputs if o.get("focused")), "")
    except (LookError, ValueError, AttributeError, TypeError):
        return ""


def screen_command(desktop: str, out: str) -> list:
    """How to take the screenshot on this desktop: grim on Hyprland, Sway and
    any other wlroots session; the screenshot portal on GNOME and elsewhere."""
    wlroots = "Hyprland" in desktop or "Sway" in desktop
    if wlroots or ("GNOME" not in desktop and os.environ.get("WAYLAND_DISPLAY")
                   and shutil.which("grim")):
        output = _focused_output(desktop) if LOOK_SCREEN != "all" else ""
        return ["grim", "-t", "png"] + (["-o", output] if output else []) + [out]
    return ["/usr/bin/python3", "-c", _PORTAL_SCRIPT, str(int(max(5, LOOK_TIMEOUT - 2)))]


def capture_screen(desktop: str, out: str) -> str:
    """Screenshot to `out` (PNG). Returns a word for which screen, for the label."""
    cmd = screen_command(desktop, out)
    proc = _run(cmd)
    if cmd[0] == "grim":
        if proc.returncode != 0:
            raise LookError(f"the screenshot failed: {_err(proc)}")
        return "the screen"
    # The portal saved it where it saves screenshots (~/Pictures on GNOME);
    # it is moved here so it does not stay behind in the user's files.
    uri = (proc.stdout or b"").decode("utf-8", "replace").strip()
    if proc.returncode != 0 or not uri.startswith("file://"):
        why = _err(proc)
        if "GNOME" in desktop:
            raise LookError("the screenshot portal gave no image (" + why + "). On GNOME the "
                            "first screenshot asks for permission on screen; allow it once")
        raise LookError(f"taking a screenshot is not supported on this desktop ({why})")
    src = Path(urllib.parse.unquote(uri[len("file://"):]))
    try:
        shutil.move(str(src), out)
    except OSError as e:
        raise LookError(f"the screenshot could not be read: {e}")
    return "the screen"


# --- camera ---------------------------------------------------------------

def camera_device() -> str:
    """The first video capture device, or "". A webcam usually exposes two
    nodes; the capture one is index 0 (the other carries metadata)."""
    if LOOK_CAMERA.startswith("/dev/"):
        return LOOK_CAMERA if Path(LOOK_CAMERA).exists() else ""
    nodes = []
    for d in Path("/sys/class/video4linux").glob("video*"):
        try:
            index = (d / "index").read_text().strip()
            num = int(d.name[len("video"):])
        except (OSError, ValueError):
            continue
        nodes.append((index != "0", num))
    for _not_capture, num in sorted(nodes):
        dev = f"/dev/video{num}"
        if Path(dev).exists():
            return dev
    return ""


def camera_command(out: str, plain: bool = False) -> list:
    """One frame from the camera to `out`. A configured command wins; else
    ffmpeg on the first capture device, as MJPEG at 1280x720 (what nearly
    every webcam offers) unless `plain`, which takes the driver's defaults."""
    if LOOK_CAMERA and not LOOK_CAMERA.startswith("/dev/"):
        return ["bash", "-c", LOOK_CAMERA.replace("{file}", shlex.quote(out))]
    dev = camera_device()
    if not dev:
        raise LookError("there is no camera on this machine")
    fmt = [] if plain else ["-input_format", "mjpeg", "-video_size", CAMERA_SIZE]
    return (["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "v4l2"]
            + fmt + ["-i", dev, "-frames:v", str(CAMERA_WARMUP_FRAMES), "-update", "1",
                     "-y", out])


def capture_camera(out: str) -> str:
    """One webcam frame to `out` (JPEG). Returns the label's words."""
    cmd = camera_command(out)
    proc = _run(cmd)
    if proc.returncode != 0 and cmd[0] == "ffmpeg":
        why = _err(proc)
        if "busy" in why.lower():
            raise LookError("the camera is busy: another program is using it")
        _LOG.info(f"look: camera refused {CAMERA_SIZE} MJPEG ({why}); trying its defaults")
        proc = _run(camera_command(out, plain=True))
    if proc.returncode != 0 or not Path(out).is_file() or Path(out).stat().st_size == 0:
        why = _err(proc)
        if "busy" in why.lower():
            raise LookError("the camera is busy: another program is using it")
        raise LookError(f"the camera gave no picture ({why})")
    return "the camera"


# --- files ----------------------------------------------------------------

def check_image_file(path: str) -> Path:
    """The image file `path` names, resolved, or LookError. ~ is expanded, and
    a relative path is taken from the home directory -- what "the picture in
    Downloads" means when said out loud."""
    raw = (path or "").strip().strip("\"'")
    if not raw:
        raise LookError("no file path was given")
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = Path.home() / p
    if not p.exists():
        raise LookError(f"there is no file at {p}")
    if p.is_dir():
        raise LookError(f"{p} is a folder, not an image")
    size = p.stat().st_size
    if size > LOOK_MAX_FILE_MB * 1024 * 1024:
        raise LookError(f"{p.name} is {size / 1048576:.0f} MB, more than the "
                        f"{LOOK_MAX_FILE_MB:.0f} MB limit for images")
    try:
        with open(p, "rb") as f:
            head = f.read(16)
    except OSError as e:
        raise LookError(f"{p} cannot be read: {e}")
    if not image_format(head):
        raise LookError(f"{p.name} is not an image ({FORMATS} only)")
    return p


def image_format(head: bytes) -> str:
    """'PNG', 'JPEG', ... from a file's first bytes, or ""."""
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WebP"
    return next((name for magic, name in _MAGIC if head.startswith(magic)), "")


# --- conversion -----------------------------------------------------------

def downscale(src: str, max_side: int = None) -> bytes:
    """`src` as JPEG bytes with its longest side at most `max_side` (never
    enlarged). ffmpeg, else ImageMagick."""
    m = int(max_side or LOOK_MAX_SIDE)
    if shutil.which("ffmpeg"):
        proc = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", src,
                     "-frames:v", "1", "-vf",
                     f"scale='min({m},iw)':'min({m},ih)':force_original_aspect_ratio=decrease",
                     "-pix_fmt", "yuvj420p", "-q:v", str(LOOK_JPEG_QUALITY),
                     "-f", "image2pipe", "-c:v", "mjpeg", "-"])
    else:
        magick = shutil.which("magick") or shutil.which("convert")
        if not magick:
            raise LookError("images cannot be resized here: install ffmpeg")
        proc = _run([magick, src + "[0]", "-auto-orient", "-resize", f"{m}x{m}>",
                     "-quality", "85", "jpeg:-"])
    if proc.returncode != 0 or not proc.stdout:
        raise LookError(f"the image could not be converted ({_err(proc)})")
    return proc.stdout


def data_uri(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _short_path(p: Path) -> str:
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return str(p)


def look(source: str, path: str = "", desktop: str = "", logger=None):
    """Capture or read one image. Returns (label, data_uri): the label is what
    stands in for the image once the turn is over, e.g. "image of the screen,
    20:15". Raises LookError."""
    log = logger or _LOG
    source = (source or "").strip().lower()
    if source not in SOURCES:
        raise LookError(f"unknown source {source!r}: use screen, camera or file")
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="va-look-") as tmp:
        if source == "file":
            src = check_image_file(path)
            what = f"image file {_short_path(src)}"
        elif source == "screen":
            src = Path(tmp) / "screen.png"
            what = "image of " + capture_screen(desktop, str(src))
        else:
            src = Path(tmp) / "camera.jpg"
            what = "image from " + capture_camera(str(src))
        jpeg = downscale(str(src))
    label = f"{what}, {time.strftime('%H:%M')}"
    log.info(f"look: {label} ({len(jpeg) // 1024} KB, {time.monotonic() - t0:.2f}s)")
    return label, data_uri(jpeg)


if __name__ == "__main__":
    # look_tools.py screen|camera|file [PATH] -- writes the JPEG that would be sent.
    logging.basicConfig(level=logging.INFO)
    try:
        label, uri = look(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "",
                          os.getenv("XDG_CURRENT_DESKTOP", ""))
    except LookError as e:
        sys.exit(f"look: {e}")
    out = Path(tempfile.gettempdir()) / "look.jpg"
    out.write_bytes(base64.b64decode(uri.split(",", 1)[1]))
    print(label, out)
