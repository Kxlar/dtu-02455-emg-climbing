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


class TriggerManager:
    """
    Handles sending hardware/software triggers in 'Live Mode' (Serial, LSL, or Parallel TTL)
    while remaining completely safe if run in 'none' (Offline SD mode).
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
                info = StreamInfo(
                    name="PsychoPy_EMG_Markers",
                    type="Markers",
                    channel_count=1,
                    nominal_srate=0,
                    channel_format="int32",
                    source_id="dtu_02455_emg_exp"
                )
                self.interface = StreamOutlet(info)
                print("[TRIGGER] LabStreamingLayer (LSL) Marker Outlet initialized.")
            except Exception as e:
                print(f"[WARNING] Failed to initialize LSL (is pylsl installed?): {e}. Falling back to 'none'.")
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
            print("[TRIGGER] Running in Offline Timestamp Mode (no live triggers emitted).")

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


def get_participant_info():
    anon_id = f"sub-{uuid.uuid4().hex[:6].upper()}"
    default_seed = random.randint(10000, 99999)

    dlg_data = {
        "Anonymized_ID": anon_id,
        "Expertise_Group": ["climber", "non_climber"],
        "Climbing_Freq_Per_Week": 0.0,
        "Climbing_Years": 0.0,
        "Dominant_Hand": ["right", "left"],
        "Random_Seed": default_seed,
        "Trigger_Mode": ["none", "serial", "lsl", "parallel"],
        "Serial_Port": config.DEFAULT_SERIAL_PORT,
        "Include_Sync_Squeeze": True
    }

    dlg = gui.DlgFromDict(
        dictionary=dlg_data,
        title="02455 sEMG Motor Resonance - Setup",
        order=[
            "Anonymized_ID", "Expertise_Group", "Dominant_Hand",
            "Climbing_Freq_Per_Week", "Climbing_Years",
            "Random_Seed", "Trigger_Mode", "Serial_Port", "Include_Sync_Squeeze"
        ]
    )

    if not dlg.OK:
        print("Experiment cancelled by user.")
        core.quit()

    return dlg_data


def load_and_randomize_stimuli(seed):
    valid_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")

    climb_files = sorted([f for f in config.CLIMBING_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    everyday_files = sorted([f for f in config.EVERYDAY_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])

    if len(climb_files) == 0 or len(everyday_files) == 0:
        raise FileNotFoundError(
            f"No video files found in '{config.CLIMBING_DIR}' (found {len(climb_files)}) "
            f"or '{config.EVERYDAY_DIR}' (found {len(everyday_files)})."
        )

    n_climb = min(len(climb_files), config.N_CLIPS_PER_CONDITION)
    n_every = min(len(everyday_files), config.N_CLIPS_PER_CONDITION)
    if n_climb < config.N_CLIPS_PER_CONDITION or n_every < config.N_CLIPS_PER_CONDITION:
        print(
            f"[TEST MODE NOTICE] Found {len(climb_files)} climbing and {len(everyday_files)} everyday clips "
            f"(target is {config.N_CLIPS_PER_CONDITION} each). Running with {n_climb + n_every} total trials."
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
    Decodes a 5-second video clip into RAM using OpenCV and wraps each frame as an
    exact 8-bit RGB PIL.Image.
    Why PIL.Image?
      - PsychoPy's ImageStim natively accepts PIL.Image objects in true 0-255 sRGB space,
        completely bypassing the [-1, 1] float contrast clipping and red/dark color shift!
      - Also avoids needing np.flipud() and uses 4x less RAM than float32 arrays.
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

        # Convert OpenCV BGR (uint8 0..255) -> true sRGB PIL Image (no float rescaling!)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(
            f"0 frames decoded from '{video_path}'. "
            f"Please run 'python standardize_videos.py' to convert it to H.264!"
        )

    duration_s = len(frames) / float(fps)
    return {"frames": frames, "fps": float(fps), "duration_s": duration_s}


def on_flip_event(container, key_prefix, trigger_mgr=None, trigger_code=None):
    t_unix = time.time()
    t_psy = core.getTime()

    if trigger_mgr is not None and trigger_code is not None:
        trigger_mgr.send(trigger_code)

    container[f"{key_prefix}_unix_s"] = t_unix
    container[f"{key_prefix}_unix_ms"] = int(round(t_unix * 1000))
    container[f"{key_prefix}_psychopy_s"] = t_psy
    container[f"{key_prefix}_trigger_code"] = trigger_code if trigger_code is not None else 0


def check_abort(win, event_log, log_filepath, trigger_mgr):
    if "escape" in event.getKeys(keyList=["escape"]):
        print("\n[INFO] ESCAPE pressed! Saving partial log and exiting...")
        if event_log:
            pd.DataFrame(event_log).to_csv(log_filepath, index=False)
        trigger_mgr.close()
        win.close()
        core.quit()


def play_preloaded_video(win, img_stim, preloaded_clip, onset_trig_code, trigger_mgr, event_log, csv_log_path):
    """
    Plays the RAM-preloaded video frames at exact wall-clock timing via ImageStim.
    Guarantees 0ms onset lag and works on every Windows GPU without ffpyplayer issues.
    """
    frames = preloaded_clip["frames"]
    fps = preloaded_clip["fps"]
    duration_s = preloaded_clip["duration_s"]
    n_frames = len(frames)

    vid_times = {}

    # Draw Frame 0 and capture exact hardware VSync onset timestamp
    img_stim.image = frames[0]
    img_stim.draw()
    win.callOnFlip(on_flip_event, vid_times, "onset", trigger_mgr, onset_trig_code)
    win.flip()

    start_t = vid_times["onset_psychopy_s"]
    last_frame_idx = 0

    # Time-locked playback loop (selects exact frame corresponding to elapsed wall-clock time)
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

    metadata = {
        **p_info,
        "effective_trigger_mode": trigger_mgr.mode,
        "session_datetime_iso": datetime.now().isoformat(),
        "baseline_duration_s": config.BASELINE_DURATION,
        "isi_range_s": [config.ISI_MIN, config.ISI_MAX],
        "n_trials_total": len(trials),
        "trigger_code_map": config.TRIGGER_CODES,
        "trial_sequence": [
            {"trial": t["trial_num"], "cond": t["condition"], "video": t["video_name"], "isi": t["isi_duration_target"]}
            for t in trials
        ]
    }
    with open(json_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    win_size = getattr(config, "WINDOW_SIZE", [1920, 1200])
    win = visual.Window(
        size=win_size,
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

    instr_text = visual.TextStim(
        win=win,
        text="",
        color=config.TEXT_COLOR,
        height=0.035,
        wrapWidth=1.2
    )

    # Single reusable ImageStim for rendering video frames
    video_img_stim = visual.ImageStim(
        win=win,
        size=config.VIDEO_SIZE,
        interpolate=True
    )

    event_log = []

    # --- STEP A: WELCOME SCREEN ---
    instr_text.text = (
        "Welcome to the Action Observation Experiment.\n\n"
        "Please rest your dominant arm completely relaxed on the armrest.\n"
        "Avoid moving or mimicking the actions you see on screen.\n\n"
        f"[Experimenter: Shimmer recording active | Trigger Mode: {trigger_mgr.mode.upper()}]\n"
        "Press SPACE to continue."
    )
    instr_text.draw()
    win.flip()
    event.waitKeys(keyList=["space", "escape"])
    check_abort(win, event_log, csv_log_path, trigger_mgr)

    # --- STEP B: OPTIONAL PHYSIOLOGICAL SYNC MARKER ---
    if p_info["Include_Sync_Squeeze"]:
        instr_text.text = (
            "CALIBRATION SYNC:\n\n"
            "When the screen says 'SQUEEZE NOW', briefly clench your fist once "
            "for 1 second, then completely relax your arm again.\n\n"
            "Press SPACE when ready."
        )
        instr_text.draw()
        win.flip()
        event.waitKeys(keyList=["space", "escape"])
        check_abort(win, event_log, csv_log_path, trigger_mgr)

        sync_times = {}
        instr_text.text = "SQUEEZE NOW (1 sec)!"
        instr_text.color = [1, -0.5, -0.5]
        instr_text.draw()
        win.callOnFlip(on_flip_event, sync_times, "onset", trigger_mgr, config.TRIGGER_CODES["sync_onset"])
        win.flip()
        core.wait(1.5)

        instr_text.text = "RELAX COMPLETELY.\n\nWaiting 5 seconds for muscle tone to settle..."
        instr_text.color = config.TEXT_COLOR
        instr_text.draw()
        win.callOnFlip(on_flip_event, sync_times, "offset", trigger_mgr, config.TRIGGER_CODES["sync_offset"])
        win.flip()
        core.wait(5.0)

        event_log.append({
            "participant_id": sub_id,
            "expertise_group": p_info["Expertise_Group"],
            "dominant_hand": p_info["Dominant_Hand"],
            "seed": seed,
            "trigger_mode": trigger_mgr.mode,
            "event_type": "sync_marker",
            "trial_num": 0,
            "condition": "calibration",
            "stimulus_file": "fist_squeeze",
            "onset_trigger_code": sync_times["onset_trigger_code"],
            "offset_trigger_code": sync_times["offset_trigger_code"],
            "onset_unix_s": sync_times["onset_unix_s"],
            "onset_unix_ms": sync_times["onset_unix_ms"],
            "offset_unix_s": sync_times["offset_unix_s"],
            "offset_unix_ms": sync_times["offset_unix_ms"],
            "onset_psychopy_s": sync_times["onset_psychopy_s"],
            "offset_psychopy_s": sync_times["offset_psychopy_s"],
            "target_duration_s": 1.5,
            "actual_duration_s": round(sync_times["offset_psychopy_s"] - sync_times["onset_psychopy_s"], 4)
        })

    # --- STEP C: 30-SECOND RESTING BASELINE (+ PRE-DECODE TRIAL 1 IN RAM) ---
    instr_text.text = (
        f"PHASE 1: RESTING BASELINE ({config.BASELINE_DURATION} seconds)\n\n"
        "Please keep your arm completely still and relaxed, "
        "and look passively at the central fixation cross.\n\n"
        "Press SPACE to start the baseline recording."
    )
    instr_text.draw()
    win.flip()
    event.waitKeys(keyList=["space", "escape"])
    check_abort(win, event_log, csv_log_path, trigger_mgr)

    base_times = {}
    fixation.draw()
    win.callOnFlip(on_flip_event, base_times, "onset", trigger_mgr, config.TRIGGER_CODES["baseline_onset"])
    win.flip()

    base_timer = core.CountdownTimer(config.BASELINE_DURATION)
    next_clip_ram = None

    while base_timer.getTime() > 0:
        fixation.draw()
        win.flip()
        # Decode Trial 1's frames into RAM while static fixation cross sits on screen
        if next_clip_ram is None:
            next_clip_ram = preload_video_frames_cv2(trials[0]["video_path"])
        check_abort(win, event_log, csv_log_path, trigger_mgr)

    fixation.draw()
    win.callOnFlip(on_flip_event, base_times, "offset", trigger_mgr, config.TRIGGER_CODES["baseline_offset"])
    win.flip()

    event_log.append({
        "participant_id": sub_id,
        "expertise_group": p_info["Expertise_Group"],
        "dominant_hand": p_info["Dominant_Hand"],
        "seed": seed,
        "trigger_mode": trigger_mgr.mode,
        "event_type": "baseline",
        "trial_num": 0,
        "condition": "resting_baseline",
        "stimulus_file": "fixation_cross",
        "onset_trigger_code": base_times["onset_trigger_code"],
        "offset_trigger_code": base_times["offset_trigger_code"],
        "onset_unix_s": base_times["onset_unix_s"],
        "onset_unix_ms": base_times["onset_unix_ms"],
        "offset_unix_s": base_times["offset_unix_s"],
        "offset_unix_ms": base_times["offset_unix_ms"],
        "onset_psychopy_s": base_times["onset_psychopy_s"],
        "offset_psychopy_s": base_times["offset_psychopy_s"],
        "target_duration_s": config.BASELINE_DURATION,
        "actual_duration_s": round(base_times["offset_psychopy_s"] - base_times["onset_psychopy_s"], 4)
    })
    pd.DataFrame(event_log).to_csv(csv_log_path, index=False)

    # --- STEP D: ZERO-LAG MAIN VIDEO BLOCK ---
    n_total = len(trials)
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

        # 1. Play Pre-decoded Video Immediately from RAM
        vid_times, vid_target_dur = play_preloaded_video(
            win=win,
            img_stim=video_img_stim,
            preloaded_clip=current_clip_ram,
            onset_trig_code=onset_trig_code,
            trigger_mgr=trigger_mgr,
            event_log=event_log,
            csv_log_path=csv_log_path
        )

        # 2. Flip immediately to Fixation Cross: marks Video Offset AND ISI Onset simultaneously
        isi_times = {}
        fixation.draw()
        win.callOnFlip(
            on_flip_event, vid_times, "offset",
            trigger_mgr, config.TRIGGER_CODES["video_offset_isi_onset"]
        )
        win.callOnFlip(
            on_flip_event, isi_times, "onset",
            None, config.TRIGGER_CODES["video_offset_isi_onset"]
        )
        win.flip()

        # Start ISI Countdown Timer immediately after flip
        isi_timer = core.CountdownTimer(isi_target)

        # 3. During the static 4-7s ISI Fixation Cross:
        #    Free finished clip RAM and pre-decode the NEXT trial's video frames into RAM
        del current_clip_ram
        while isi_timer.getTime() > 0:
            fixation.draw()
            win.flip()
            if next_clip_ram is None and (idx + 1 < n_total):
                next_clip_ram = preload_video_frames_cv2(trials[idx + 1]["video_path"])
            check_abort(win, event_log, csv_log_path, trigger_mgr)

        # Mark end of ISI
        fixation.draw()
        win.callOnFlip(
            on_flip_event, isi_times, "offset",
            trigger_mgr, config.TRIGGER_CODES["isi_offset"]
        )
        win.flip()

        # 4. Log both Video and ISI rows
        event_log.append({
            "participant_id": sub_id,
            "expertise_group": p_info["Expertise_Group"],
            "dominant_hand": p_info["Dominant_Hand"],
            "seed": seed,
            "trigger_mode": trigger_mgr.mode,
            "event_type": "trial_video",
            "trial_num": t_num,
            "condition": cond,
            "stimulus_file": vid_name,
            "onset_trigger_code": vid_times["onset_trigger_code"],
            "offset_trigger_code": vid_times["offset_trigger_code"],
            "onset_unix_s": vid_times["onset_unix_s"],
            "onset_unix_ms": vid_times["onset_unix_ms"],
            "offset_unix_s": vid_times["offset_unix_s"],
            "offset_unix_ms": vid_times["offset_unix_ms"],
            "onset_psychopy_s": vid_times["onset_psychopy_s"],
            "offset_psychopy_s": vid_times["offset_psychopy_s"],
            "target_duration_s": round(vid_target_dur, 4),
            "actual_duration_s": round(vid_times["offset_psychopy_s"] - vid_times["onset_psychopy_s"], 4)
        })

        event_log.append({
            "participant_id": sub_id,
            "expertise_group": p_info["Expertise_Group"],
            "dominant_hand": p_info["Dominant_Hand"],
            "seed": seed,
            "trigger_mode": trigger_mgr.mode,
            "event_type": "isi",
            "trial_num": t_num,
            "condition": f"post_{cond}_isi",
            "stimulus_file": "fixation_cross",
            "onset_trigger_code": isi_times["onset_trigger_code"],
            "offset_trigger_code": isi_times["offset_trigger_code"],
            "onset_unix_s": isi_times["onset_unix_s"],
            "onset_unix_ms": isi_times["onset_unix_ms"],
            "offset_unix_s": isi_times["offset_unix_s"],
            "offset_unix_ms": isi_times["offset_unix_ms"],
            "onset_psychopy_s": isi_times["onset_psychopy_s"],
            "offset_psychopy_s": isi_times["offset_psychopy_s"],
            "target_duration_s": isi_target,
            "actual_duration_s": round(isi_times["offset_psychopy_s"] - isi_times["onset_psychopy_s"], 4)
        })

        pd.DataFrame(event_log).to_csv(csv_log_path, index=False)

    # --- STEP E: END SCREEN ---
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