# ./Experiment in Cognitive Science 02455/run_experiment.py
import json
import uuid
import time
import random
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from psychopy import visual, core, event, gui, logging
import config

logging.console.setLevel(logging.ERROR)

# Guard against PsychoPy MovieStim cleanup bug on exit
if hasattr(visual, "MovieStim") and not hasattr(visual.MovieStim, "_freePlayer"):
    visual.MovieStim._freePlayer = lambda self: None


class TriggerManager:
    """
    Handles sending hardware/software triggers in 'Live Mode' (Serial, LSL, or Parallel TTL)
    while remaining completely safe if run in 'none' (Offline SD Squeeze-Synced Mode).
    """
    def __init__(self, mode="none", port_name=config.DEFAULT_SERIAL_PORT, baudrate=config.SERIAL_BAUDRATE):
        self.mode = mode.lower()
        self.port_name = port_name
        self.baudrate = baudrate
        self.interface = None

        if self.mode == "serial":
            try:
                import serial
                self.interface = serial.Serial(self.port_name, self.baudrate, timeout=0.01)
                print(f"[TRIGGER] Connected to Serial Port {self.port_name} at {self.baudrate} baud.")
            except Exception as e:
                print(f"[WARNING] Failed to open Serial Port '{self.port_name}': {e}. Falling back to 'none'.")
                self.mode = "none"
        elif self.mode == "lsl":
            try:
                from pylsl import StreamInfo, StreamOutlet
                info = StreamInfo("PsychoPy_EMG_Markers", "Markers", 1, 0, "int32", "dtu_02455_emg_exp")
                self.interface = StreamOutlet(info)
                print("[TRIGGER] LabStreamingLayer (LSL) Marker Outlet initialized.")
            except Exception as e:
                print(f"[WARNING] Failed to initialize LSL: {e}. Falling back to 'none'.")
                self.mode = "none"
        elif self.mode == "parallel":
            try:
                from psychopy import parallel
                self.interface = parallel.ParallelPort(address=config.PARALLEL_PORT_ADDRESS)
                self.interface.setData(0)
                print(f"[TRIGGER] Parallel Port initialized at {hex(config.PARALLEL_PORT_ADDRESS)}.")
            except Exception as e:
                print(f"[WARNING] Failed to initialize Parallel Port: {e}. Falling back to 'none'.")
                self.mode = "none"
        else:
            print("[TRIGGER] Running in Continuous Squeeze-Synced Mode (no live triggers emitted).")

    def send(self, code):
        if self.mode == "none" or self.interface is None:
            return
        try:
            if self.mode == "serial":
                self.interface.write(bytes([int(code)]))
            elif self.mode == "lsl":
                self.interface.push_sample([int(code)])
            elif self.mode == "parallel":
                self.interface.setData(int(code))
        except Exception as e:
            print(f"[WARNING] Trigger send error (code {code}): {e}")

    def close(self):
        if self.mode == "serial" and self.interface is not None:
            self.interface.close()
        elif self.mode == "parallel" and self.interface is not None:
            self.interface.setData(0)


def fmt_sec(val: float) -> str:
    """Formats a duration from config.py cleanly (e.g. 30.0 -> '30', 1.5 -> '1.5')."""
    return f"{int(val)}" if float(val).is_integer() else f"{val:g}"


def get_participant_info():
    anon_id = f"sub-{uuid.uuid4().hex[:6].upper()}"
    default_seed = random.randint(10000, 99999)

    dlg_data = {
        "Anonymized_ID": anon_id,
        "Expertise_Group": ["climber", "non_climber"],
        "Climbing_Freq_Per_Week": 0.0,
        "Climbing_Years": 0.0,
        "Dominant_Hand": ["right", "left"],
        "Shimmer_Sampling_Rate_Hz": config.DEFAULT_SAMPLING_RATE,
        "Random_Seed": default_seed,
        "Trigger_Mode": ["none", "serial", "lsl", "parallel"],
        "Serial_Port": config.DEFAULT_SERIAL_PORT
    }

    dlg = gui.DlgFromDict(
        dictionary=dlg_data,
        title="02455 sEMG Motor Resonance - Setup",
        order=[
            "Anonymized_ID", "Expertise_Group", "Dominant_Hand",
            "Climbing_Freq_Per_Week", "Climbing_Years",
            "Shimmer_Sampling_Rate_Hz", "Random_Seed", "Trigger_Mode", "Serial_Port"
        ]
    )

    if not dlg.OK:
        print("Experiment cancelled by user.")
        core.quit()

    return dlg_data


def load_and_randomize_stimuli(seed):
    valid_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv")

    climb_files = sorted([f for f in config.CLIMBING_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    everyday_files = sorted([f for f in config.EVERYDAY_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])

    if len(climb_files) == 0 or len(everyday_files) == 0:
        raise FileNotFoundError(
            f"No video files found in '{config.CLIMBING_DIR}' (found {len(climb_files)}) "
            f"or '{config.EVERYDAY_DIR}' (found {len(everyday_files)}).\n"
            f"Place your clips in stimuli/climbing/ and stimuli/everyday/ and run standardize_videos.py."
        )

    n_climb = min(len(climb_files), config.N_CLIPS_PER_CONDITION)
    n_every = min(len(everyday_files), config.N_CLIPS_PER_CONDITION)
    if n_climb < config.N_CLIPS_PER_CONDITION or n_every < config.N_CLIPS_PER_CONDITION:
        print(
            f"[TEST MODE NOTICE] Found {len(climb_files)} climbing and {len(everyday_files)} everyday clips "
            f"(config.N_CLIPS_PER_CONDITION is {config.N_CLIPS_PER_CONDITION}). "
            f"Running with {n_climb + n_every} total trials."
        )

    trials = []
    for f in climb_files[:n_climb]:
        trials.append({"condition": "climbing", "video_name": f.name, "video_path": str(f)})
    for f in everyday_files[:n_every]:
        trials.append({"condition": "everyday", "video_name": f.name, "video_path": str(f)})

    rng = random.Random(int(seed))
    rng.shuffle(trials)

    for idx, trial in enumerate(trials, start=1):
        trial["trial_num"] = idx
        trial["isi_duration_target"] = round(rng.uniform(config.ISI_MIN, config.ISI_MAX), 4)

    return trials


def preload_video_frames_cv2(video_path: str, max_width: int = 1280):
    """
    Decodes a video clip into RAM using OpenCV and wraps each frame as an 8-bit sRGB PIL.Image.
    Using PIL.Image bypasses PsychoPy's [-1, 1] float color-space conversion, preserving exact
    BT.709 colors, contrast, and luminosity with zero frame-drop lag.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video file: {video_path}. Run standardize_videos.py first!")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0

    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        h, w = frame_bgr.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"0 frames decoded from '{video_path}'. Run standardize_videos.py first!")

    duration_s = len(frames) / float(fps)
    return {"frames": frames, "fps": float(fps), "duration_s": duration_s}


def on_flip_event(container, key_prefix, sync_anchor_ref, trigger_mgr=None, trigger_code=None):
    """
    Executed on the exact hardware VSync flip:
      - Locks `t0_psy` on the very first Start Squeeze onset (`sync_start`)
      - Computes `rel_sync_s` = exact elapsed seconds since `t = 0.000s` Start Squeeze
    """
    t_unix = time.time()
    t_psy = core.getTime()

    if trigger_mgr is not None and trigger_code is not None:
        trigger_mgr.send(trigger_code)

    if sync_anchor_ref["t0_psy"] is None:
        sync_anchor_ref["t0_psy"] = t_psy

    rel_s = t_psy - sync_anchor_ref["t0_psy"]

    container[f"{key_prefix}_unix_s"] = t_unix
    container[f"{key_prefix}_unix_ms"] = int(round(t_unix * 1000))
    container[f"{key_prefix}_psychopy_s"] = t_psy
    container[f"{key_prefix}_rel_sync_s"] = round(rel_s, 5)
    container[f"{key_prefix}_trigger_code"] = trigger_code if trigger_code is not None else 0


def check_abort(win, event_log, log_filepath, trigger_mgr):
    if "escape" in event.getKeys(keyList=["escape"]):
        print("\n[INFO] ESCAPE pressed! Saving partial log and exiting...")
        if event_log:
            pd.DataFrame(event_log).to_csv(log_filepath, index=False)
        trigger_mgr.close()
        win.close()
        core.quit()


def play_preloaded_video(win, img_stim, preloaded_clip, onset_trig_code, sync_anchor_ref, trigger_mgr, event_log, csv_log_path):
    frames = preloaded_clip["frames"]
    fps = preloaded_clip["fps"]
    duration_s = preloaded_clip["duration_s"]
    n_frames = len(frames)

    vid_times = {}

    img_stim.image = frames[0]
    img_stim.draw()
    win.callOnFlip(on_flip_event, vid_times, "onset", sync_anchor_ref, trigger_mgr, onset_trig_code)
    win.flip()

    start_t = vid_times["onset_psychopy_s"]
    last_frame_idx = 0

    while True:
        elapsed = core.getTime() - start_t
        if elapsed >= duration_s:
            break

        frame_idx = int(elapsed * fps)
        if frame_idx >= n_frames:
            break

        if frame_idx != last_frame_idx:
            img_stim.image = frames[frame_idx]
            last_frame_idx = frame_idx

        img_stim.draw()
        win.flip()
        check_abort(win, event_log, csv_log_path, trigger_mgr)

    return vid_times, duration_s


def build_log_row(p_info, seed, trig_mode, ev_type, trial_num, cond, stim_file, times_dict, target_dur):
    return {
        "participant_id": p_info["Anonymized_ID"],
        "expertise_group": p_info["Expertise_Group"],
        "dominant_hand": p_info["Dominant_Hand"],
        "shimmer_fs_configured_hz": float(p_info["Shimmer_Sampling_Rate_Hz"]),
        "seed": seed,
        "trigger_mode": trig_mode,
        "event_type": ev_type,
        "trial_num": trial_num,
        "condition": cond,
        "stimulus_file": stim_file,
        # Primary timing columns for offline Shimmer alignment (relative to t=0 Start Squeeze):
        "rel_from_sync_onset_s": times_dict["onset_rel_sync_s"],
        "rel_from_sync_offset_s": times_dict["offset_rel_sync_s"],
        "target_duration_s": round(float(target_dur), 4),
        "actual_duration_s": round(times_dict["offset_psychopy_s"] - times_dict["onset_psychopy_s"], 4),
        # Secondary clocks & trigger codes:
        "onset_psychopy_s": round(times_dict["onset_psychopy_s"], 5),
        "offset_psychopy_s": round(times_dict["offset_psychopy_s"], 5),
        "onset_unix_ms": times_dict["onset_unix_ms"],
        "offset_unix_ms": times_dict["offset_unix_ms"],
        "onset_trigger_code": times_dict["onset_trigger_code"],
        "offset_trigger_code": times_dict["offset_trigger_code"],
    }


def run_experiment():
    p_info = get_participant_info()
    sub_id = p_info["Anonymized_ID"]
    seed = int(p_info["Random_Seed"])
    trig_mode = p_info["Trigger_Mode"]
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_filename = f"{sub_id}_{p_info['Expertise_Group']}_{session_timestamp}"
    csv_log_path = config.LOG_DIR / f"{base_filename}_events.csv"
    json_meta_path = config.LOG_DIR / f"{base_filename}_metadata.json"

    trigger_mgr = TriggerManager(mode=trig_mode, port_name=p_info["Serial_Port"])
    trials = load_and_randomize_stimuli(seed)
    n_total = len(trials)

    # Save session metadata with all active config parameters
    metadata = {
        **p_info,
        "effective_trigger_mode": trigger_mgr.mode,
        "session_datetime_iso": datetime.now().isoformat(),
        "sync_countdown_duration_s": config.SYNC_COUNTDOWN_DURATION,
        "sync_squeeze_duration_s": config.SYNC_SQUEEZE_DURATION,
        "post_sync_settle_s": config.POST_SYNC_SETTLE,
        "baseline_duration_s": config.BASELINE_DURATION,
        "isi_range_s": [config.ISI_MIN, config.ISI_MAX],
        "n_clips_per_condition_config": config.N_CLIPS_PER_CONDITION,
        "n_trials_actual": n_total,
        "emg_channel_map": config.EMG_CHANNEL_MAP,
        "trial_sequence": [
            {"trial": t["trial_num"], "cond": t["condition"], "video": t["video_name"], "isi": t["isi_duration_target"]}
            for t in trials
        ]
    }
    with open(json_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    win = visual.Window(
        size=config.WINDOW_SIZE,
        fullscr=config.FULLSCREEN,
        screen=config.SCREEN_NUM,
        color=config.BG_COLOR,
        units="height",
        allowGUI=False
    )
    win.mouseVisible = False

    fixation = visual.ShapeStim(
        win=win,
        vertices=((0, -0.04), (0, 0.04), (0, 0), (-0.04, 0), (0.04, 0)),
        lineWidth=4,
        closeShape=False,
        lineColor=config.TEXT_COLOR
    )

    instr_text = visual.TextStim(win=win, text="", color=config.TEXT_COLOR, height=0.032, wrapWidth=1.28)
    video_img_stim = visual.ImageStim(win=win, size=config.VIDEO_SIZE, interpolate=True)

    event_log = []
    sync_anchor_ref = {"t0_psy": None}

    # Format config durations cleanly for on-screen instructions
    cd_str = fmt_sec(config.SYNC_COUNTDOWN_DURATION)
    sq_str = fmt_sec(config.SYNC_SQUEEZE_DURATION)
    settle_str = fmt_sec(config.POST_SYNC_SETTLE)
    base_str = fmt_sec(config.BASELINE_DURATION)
    isi_min_str = fmt_sec(config.ISI_MIN)
    isi_max_str = fmt_sec(config.ISI_MAX)

    # --- SINGLE STARTUP INSTRUCTION SCREEN (ALL NUMBERS PULLED FROM config.py) ---
    instr_text.text = (
        "WELCOME TO THE ACTION OBSERVATION EXPERIMENT\n\n"
        f"Please rest your {p_info['Dominant_Hand'].upper()} forearm comfortably on the armrest.\n"
        "Once we begin, the entire session will run automatically without pausing:\n\n"
        f"  1. START SYNC: After a {cd_str}s countdown, clench your fist firmly for {sq_str}s\n"
        f"     when 'SQUEEZE HARD NOW!' appears, then relax for {settle_str}s.\n"
        f"  2. BASELINE ({base_str}s): Passively look at the central fixation cross while\n"
        "     keeping your hand and forearm completely relaxed.\n"
        f"  3. VIDEO TRIALS ({n_total} clips): Passively watch each hand action video\n"
        f"     (separated by {isi_min_str}–{isi_max_str}s fixation crosses). Do not mimic the movements.\n"
        f"  4. END SYNC: One final {sq_str}s fist squeeze at the very end.\n\n"
        "[Experimenter: Confirm Shimmer is recording -> Press SPACE to start continuous run]"
    )
    instr_text.draw()
    win.flip()
    event.waitKeys(keyList=["space", "escape"])
    check_abort(win, event_log, csv_log_path, trigger_mgr)

    # ==========================================================================
    # CONTINUOUS ZERO-PAUSE EXECUTION BLOCK (LOCKED TO t = 0.000s START SQUEEZE)
    # ==========================================================================

    # --- 1. COUNTDOWN TO START SQUEEZE (config.SYNC_COUNTDOWN_DURATION) ---
    cd_timer = core.CountdownTimer(config.SYNC_COUNTDOWN_DURATION)
    while cd_timer.getTime() > 0:
        rem = int(np.ceil(cd_timer.getTime()))
        instr_text.text = (
            f"Get ready to clench your {p_info['Dominant_Hand']} fist once for {sq_str}s...\n\n"
            f"Starting in {rem}"
        )
        instr_text.color = config.TEXT_COLOR
        instr_text.draw()
        win.flip()
        check_abort(win, event_log, csv_log_path, trigger_mgr)

    # --- 2. START SQUEEZE: t = 0.000s ANCHOR (config.SYNC_SQUEEZE_DURATION) ---
    sync_start_times = {}
    instr_text.text = f"SQUEEZE HARD NOW! ({sq_str}s)"
    instr_text.color = [1, -0.4, -0.4]
    instr_text.draw()
    win.callOnFlip(
        on_flip_event, sync_start_times, "onset",
        sync_anchor_ref, trigger_mgr, config.TRIGGER_CODES["sync_start_onset"]
    )
    win.flip()

    sq_timer = core.CountdownTimer(config.SYNC_SQUEEZE_DURATION)
    while sq_timer.getTime() > 0:
        instr_text.draw()
        win.flip()
        check_abort(win, event_log, csv_log_path, trigger_mgr)

    # --- 3. POST-SQUEEZE MUSCLE RELAXATION SETTLE (config.POST_SYNC_SETTLE) ---
    settle_times = {}
    instr_text.color = config.TEXT_COLOR
    instr_text.text = (
        f"RELAX YOUR ARM COMPLETELY.\n\n"
        f"{base_str}s resting baseline starting in {int(np.ceil(config.POST_SYNC_SETTLE))}s..."
    )
    instr_text.draw()
    win.callOnFlip(
        on_flip_event, sync_start_times, "offset",
        sync_anchor_ref, trigger_mgr, config.TRIGGER_CODES["sync_start_offset"]
    )
    win.callOnFlip(on_flip_event, settle_times, "onset", sync_anchor_ref, None, 0)
    win.flip()

    settle_timer = core.CountdownTimer(config.POST_SYNC_SETTLE)
    while settle_timer.getTime() > 0:
        rem_settle = int(np.ceil(settle_timer.getTime()))
        instr_text.text = (
            f"RELAX YOUR ARM COMPLETELY.\n\n"
            f"{base_str}s resting baseline starting in {rem_settle}s..."
        )
        instr_text.draw()
        win.flip()
        check_abort(win, event_log, csv_log_path, trigger_mgr)

    event_log.append(build_log_row(
        p_info, seed, trigger_mgr.mode, "sync_start", 0, "calibration_start",
        "fist_squeeze_start", sync_start_times, config.SYNC_SQUEEZE_DURATION
    ))

    # --- 4. RESTING BASELINE (config.BASELINE_DURATION) + PRELOAD TRIAL 1 ---
    base_times = {}
    fixation.draw()
    win.callOnFlip(on_flip_event, settle_times, "offset", sync_anchor_ref, None, 0)
    win.callOnFlip(
        on_flip_event, base_times, "onset",
        sync_anchor_ref, trigger_mgr, config.TRIGGER_CODES["baseline_onset"]
    )
    win.flip()

    base_timer = core.CountdownTimer(config.BASELINE_DURATION)
    next_clip_ram = None

    while base_timer.getTime() > 0:
        fixation.draw()
        win.flip()
        if next_clip_ram is None:
            next_clip_ram = preload_video_frames_cv2(trials[0]["video_path"])
        check_abort(win, event_log, csv_log_path, trigger_mgr)

    fixation.draw()
    win.callOnFlip(
        on_flip_event, base_times, "offset",
        sync_anchor_ref, trigger_mgr, config.TRIGGER_CODES["baseline_offset"]
    )
    win.flip()

    event_log.append(build_log_row(
        p_info, seed, trigger_mgr.mode, "post_sync_settle", 0, "muscle_settle",
        "relax_screen", settle_times, config.POST_SYNC_SETTLE
    ))
    event_log.append(build_log_row(
        p_info, seed, trigger_mgr.mode, "baseline", 0, "resting_baseline",
        "fixation_cross", base_times, config.BASELINE_DURATION
    ))
    pd.DataFrame(event_log).to_csv(csv_log_path, index=False)

    # --- 5. MAIN VIDEO BLOCK (n_total TRIALS + JITTERED ISI FROM config.ISI_MIN..ISI_MAX) ---
    for idx, trial in enumerate(trials):
        t_num = trial["trial_num"]
        cond = trial["condition"]
        vid_name = trial["video_name"]
        isi_target = trial["isi_duration_target"]

        current_clip_ram = next_clip_ram
        next_clip_ram = None

        onset_trig_code = (
            config.TRIGGER_CODES["climbing_onset"]
            if cond == "climbing"
            else config.TRIGGER_CODES["everyday_onset"]
        )

        vid_times, vid_target_dur = play_preloaded_video(
            win=win,
            img_stim=video_img_stim,
            preloaded_clip=current_clip_ram,
            onset_trig_code=onset_trig_code,
            sync_anchor_ref=sync_anchor_ref,
            trigger_mgr=trigger_mgr,
            event_log=event_log,
            csv_log_path=csv_log_path
        )

        # Flip immediately to Fixation Cross: marks Video Offset AND ISI Onset simultaneously
        isi_times = {}
        fixation.draw()
        win.callOnFlip(
            on_flip_event, vid_times, "offset",
            sync_anchor_ref, trigger_mgr, config.TRIGGER_CODES["video_offset_isi_onset"]
        )
        win.callOnFlip(
            on_flip_event, isi_times, "onset",
            sync_anchor_ref, None, config.TRIGGER_CODES["video_offset_isi_onset"]
        )
        win.flip()

        isi_timer = core.CountdownTimer(isi_target)

        del current_clip_ram
        while isi_timer.getTime() > 0:
            fixation.draw()
            win.flip()
            if next_clip_ram is None and (idx + 1 < n_total):
                next_clip_ram = preload_video_frames_cv2(trials[idx + 1]["video_path"])
            check_abort(win, event_log, csv_log_path, trigger_mgr)

        fixation.draw()
        win.callOnFlip(
            on_flip_event, isi_times, "offset",
            sync_anchor_ref, trigger_mgr, config.TRIGGER_CODES["isi_offset"]
        )
        win.flip()

        event_log.append(build_log_row(
            p_info, seed, trigger_mgr.mode, "trial_video", t_num, cond,
            vid_name, vid_times, vid_target_dur
        ))
        event_log.append(build_log_row(
            p_info, seed, trigger_mgr.mode, "isi", t_num, f"post_{cond}_isi",
            "fixation_cross", isi_times, isi_target
        ))
        pd.DataFrame(event_log).to_csv(csv_log_path, index=False)

    # --- 6. AUTOMATIC END-OF-SESSION BOOKEND SQUEEZE (config.SYNC_SQUEEZE_DURATION) ---
    sync_end_times = {}
    instr_text.text = f"FINAL CALIBRATION:\nSQUEEZE HARD NOW! ({sq_str}s)"
    instr_text.color = [1, -0.4, -0.4]
    instr_text.draw()
    win.callOnFlip(
        on_flip_event, sync_end_times, "onset",
        sync_anchor_ref, trigger_mgr, config.TRIGGER_CODES["sync_end_onset"]
    )
    win.flip()

    end_sq_timer = core.CountdownTimer(config.SYNC_SQUEEZE_DURATION)
    while end_sq_timer.getTime() > 0:
        instr_text.draw()
        win.flip()
        check_abort(win, event_log, csv_log_path, trigger_mgr)

    instr_text.text = "RELAX.\nSaving session logs..."
    instr_text.color = config.TEXT_COLOR
    instr_text.draw()
    win.callOnFlip(
        on_flip_event, sync_end_times, "offset",
        sync_anchor_ref, trigger_mgr, config.TRIGGER_CODES["sync_end_offset"]
    )
    win.flip()
    core.wait(2.0)

    event_log.append(build_log_row(
        p_info, seed, trigger_mgr.mode, "sync_end", n_total + 1, "calibration_end",
        "fist_squeeze_end", sync_end_times, config.SYNC_SQUEEZE_DURATION
    ))
    pd.DataFrame(event_log).to_csv(csv_log_path, index=False)

    # --- 7. END SCREEN ---
    trigger_mgr.send(config.TRIGGER_CODES["experiment_end"])
    instr_text.text = (
        "The experiment is now complete!\n\n"
        "Thank you for your participation.\n"
        "[Experimenter: Stop the Shimmer recording and press SPACE to exit.]"
    )
    instr_text.draw()
    win.flip()
    event.waitKeys(keyList=["space", "escape"])

    trigger_mgr.close()
    win.close()
    core.quit()


if __name__ == "__main__":
    run_experiment()