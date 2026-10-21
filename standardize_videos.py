# ./Experiment in Cognitive Science 02455/standardize_videos.py
import re
import shutil
import subprocess
from pathlib import Path
import cv2
import config

# --- STANDARDIZATION SETTINGS ---
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 30
TARGET_DURATION_S = 4.0                     # Exact target length in seconds for ALL videos (e.g. 4.0s)
TARGET_FRAMES = int(round(TARGET_DURATION_S * TARGET_FPS))  # Exactly 120 frames @ 30fps

# Choose how to handle videos that are longer/shorter than TARGET_DURATION_S:
#   "center_crop" : Cuts equal time from start & end to isolate the middle 4.0s action (stretches if < 4.0s)
#   "speed_match" : Smoothly adjusts playback speed so the entire clip fits into 4.0s without cutting frames
#   "start_trim"  : Takes the first 4.0s from 0.0s (pads/stretches if < 4.0s)
DURATION_MODE = "start_trim"

# Supported input formats (now including .wmv, .mpg, .mpeg, .flv, .m4v)
VALID_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".mpg", ".mpeg", ".m4v")


def get_ffmpeg_exe() -> str:
    """Finds the ffmpeg executable (either from imageio_ffmpeg inside venv or system PATH)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    raise RuntimeError(
        "Could not find ffmpeg! Run this in your terminal inside venv_ExpCogSc:\n"
        "  pip install imageio-ffmpeg"
    )


def get_video_duration(video_path: Path, ffmpeg_exe: str) -> float:
    """
    Reliably determines the duration (in seconds) of any video format, including .wmv.
    1. First parses the container 'Duration: HH:MM:SS.xx' header via ffmpeg -i.
    2. Falls back to OpenCV metadata or frame counting if needed.
    """
    # 1. Try FFmpeg header probe (most reliable for .wmv / ASF containers)
    cmd = [ffmpeg_exe, "-i", str(video_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", res.stderr)
    if match:
        hrs, mins, secs = float(match.group(1)), float(match.group(2)), float(match.group(3))
        total_s = hrs * 3600.0 + mins * 60.0 + secs
        if total_s > 0:
            return total_s

    # 2. Fallback to OpenCV property check
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return TARGET_DURATION_S
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps and fps > 0 and frame_count and frame_count > 0:
        cap.release()
        return float(frame_count) / float(fps)

    # 3. Last resort for headerless .wmv streams: count decoded frames directly
    counted_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        counted_frames += 1
    cap.release()

    effective_fps = fps if (fps and 1.0 < fps < 240.0) else TARGET_FPS
    if counted_frames > 0:
        return float(counted_frames) / float(effective_fps)

    return TARGET_DURATION_S


def standardize_folder(folder_path: Path, backup_root: Path, ffmpeg_exe: str, mode: str = DURATION_MODE):
    backup_dir = backup_root / folder_path.name
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 1. Move any newly placed raw files in folder_path into backup_dir first
    #    (unless a backup with the exact same stem already exists, so we never overwrite untouched originals)
    existing_backup_stems = {b.stem.lower(): b for b in backup_dir.iterdir() if b.is_file() and b.suffix.lower() in VALID_EXTS}

    for f in list(folder_path.iterdir()):
        if not f.is_file() or f.suffix.lower() not in VALID_EXTS or f.name.startswith("_temp_"):
            continue
        stem_lower = f.stem.lower()
        if stem_lower not in existing_backup_stems:
            backup_target = backup_dir / f.name
            shutil.copy2(str(f), str(backup_target))
            existing_backup_stems[stem_lower] = backup_target

    # 2. Clean out any non-.mp4 files (e.g. raw .wmv / .mov) from the active stimuli folder
    #    since they are safely backed up in backup_dir and will be replaced by standardized .mp4 files
    for f in list(folder_path.iterdir()):
        if f.is_file() and f.suffix.lower() in VALID_EXTS and f.suffix.lower() != ".mp4":
            f.unlink(missing_ok=True)

    source_files = sorted([b for b in backup_dir.iterdir() if b.is_file() and b.suffix.lower() in VALID_EXTS])
    if not source_files:
        print(f"[SKIP] No video files found in {folder_path}")
        return

    for src_path in source_files:
        orig_dur = get_video_duration(src_path, ffmpeg_exe)
        temp_out = folder_path / f"_temp_h264_{src_path.stem}.mp4"
        final_out = folder_path / f"{src_path.stem}.mp4"

        print(
            f"Standardizing: {src_path.name:<20} | Orig: {orig_dur:5.2f}s -> "
            f"Target: {TARGET_DURATION_S:.2f}s ({TARGET_FRAMES} frames, mode='{mode}')"
        )

        # Base spatial & color-space filter (1080p, BT.709 accurate sRGB, neutral grey letterbox padding)
        scale_pad_filter = (
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease:"
            f"out_color_matrix=bt709:out_range=tv,"
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x808080,"
            f"setsar=1"
        )

        ss_args = []
        time_filter = ""

        if mode == "speed_match" or orig_dur < TARGET_DURATION_S:
            speed_factor = TARGET_DURATION_S / max(orig_dur, 0.1)
            time_filter = f"setpts={speed_factor:.6f}*PTS,"
        elif mode == "center_crop" and orig_dur > TARGET_DURATION_S:
            start_offset = (orig_dur - TARGET_DURATION_S) / 2.0
            # Put -ss after -i for .wmv files because WMV keyframes can be sparse
            ss_args = ["-ss", f"{start_offset:.4f}"]
        elif mode == "start_trim":
            ss_args = ["-ss", "0.0"]

        vf_chain = (
            f"{time_filter}{scale_pad_filter},"
            f"fps={TARGET_FPS},"
            f"tpad=stop_mode=clone:stop_duration=2.0,"
            f"format=yuv420p"
        )

        # Note: Putting *ss_args AFTER -i ensures frame-accurate seeking on .wmv files!
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(src_path),
            *ss_args,
            "-vf", vf_chain,
            "-frames:v", str(TARGET_FRAMES),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-an",
            "-movflags", "+faststart",
            str(temp_out)
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 or not temp_out.exists():
            print(f"  [ERROR] Failed to convert {src_path.name}:\n{result.stderr[-400:]}")
            continue

        final_out.unlink(missing_ok=True)
        shutil.move(str(temp_out), str(final_out))

        new_dur = get_video_duration(final_out, ffmpeg_exe)
        print(f"  [OK] Saved {final_out.name} -> Verified Duration: {new_dur:.3f}s ({TARGET_FRAMES} frames)")


def main():
    ffmpeg_exe = get_ffmpeg_exe()
    print(f"Using FFmpeg binary: {ffmpeg_exe}")
    print(f"Target Spec: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {TARGET_FPS}fps | Exact Duration: {TARGET_DURATION_S}s ({TARGET_FRAMES} frames)\n")

    backup_root = config.STIM_DIR / "raw_originals_backup"
    standardize_folder(config.CLIMBING_DIR, backup_root, ffmpeg_exe, mode=DURATION_MODE)
    standardize_folder(config.EVERYDAY_DIR, backup_root, ffmpeg_exe, mode=DURATION_MODE)

    print(f"\n[DONE] Every video is now a standardized {TARGET_DURATION_S:.2f}s H.264 .mp4!")


if __name__ == "__main__":
    main()