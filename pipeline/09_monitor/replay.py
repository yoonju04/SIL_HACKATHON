"""Failure-event clips for human replay only; never sent to the insight model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from monitor import view
from video_clip import find_video, make_clip

FAILURES = ("F1_no_or_late_switch", "F2_switch_transient", "F3_not_recovered")


def clip_window(record):
    r = view(record)
    onset = r.get('actual_fault_onset')
    if onset is None:
        onset = r.get('planned_fault_onset')
    onset = float(onset) if onset is not None else 0.0
    sr4 = (r.get('sr') or {}).get('SR4') or {}
    collision = (r.get('sr_measures') or {}).get('SR4_collision_t')
    event = r.get('switch_time') if r.get('outcome_category') == 'F2_switch_transient' else sr4.get('first_violation_t')
    if event is None:
        event = collision if collision is not None else r.get('switch_time')
    event = onset if event is None else float(event)
    start = max(0.0, min(onset, event) - 1.0)
    end = max(onset, event) + 3.0
    window = r.get('evaluation_window')
    if window and len(window) == 2:
        end = min(end, float(window[1]))
    return start, max(start + 0.05, end), event


def prepare_replays(records, screen):
    """Read source recording metadata and cache a clip per failed run, independent of API access."""
    screen = Path(screen)
    screen.mkdir(parents=True, exist_ok=True)
    manifest = screen / 'replays.json'
    previous = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
    output = {}
    for raw in records:
        r = view(raw)
        if r.get('outcome_category') not in FAILURES:
            continue
        rid = str(raw.get('run_id') or r.get('run_id') or '')
        start, end, event = clip_window(r)
        row = {'run_id': rid, 'clip': None, 'start_s': start, 'end_s': end, 'event_s': event,
               'source': None, 'exact_timing': False, 'error': None}
        folder = (r.get('run_dirs') or {}).get('fault')
        if not folder:
            row['error'] = '원본 녹화 경로가 없습니다. 녹화된 fault 런을 연결하면 실패 구간이 표시됩니다.'
            output[rid] = row
            continue
        try:
            src = find_video(folder)
            if not src:
                row['error'] = '이 런의 view.mp4 / front.mp4 녹화가 없습니다.'
            else:
                stat = src['video'].stat()
                stamp = hashlib.sha256(json.dumps([str(src['video'].resolve()), stat.st_mtime_ns,
                                                    stat.st_size, src['times'], start, end]).encode()).hexdigest()
                cached = previous.get(rid) or {}
                if cached.get('source_stamp') == stamp and cached.get('clip') and (screen / cached['clip']).is_file():
                    output[rid] = cached
                    continue
                name = hashlib.sha256(rid.encode()).hexdigest()[:20] + '.mp4'
                result = make_clip(folder, start, end, screen / 'clips' / name, n_frames=0)
                row.update(source=src['video'].name, source_stamp=stamp)
                if result:
                    row.update(clip='clips/' + name, exact_timing=result['exact_timing'])
                else:
                    row['error'] = '클립을 만들 수 없습니다. ffmpeg 설치와 원본 영상 형식을 확인하세요.'
        except (OSError, ValueError, TypeError, KeyError) as exc:
            row['error'] = f'리플레이 준비 실패: {type(exc).__name__}'
        output[rid] = row
    tmp = manifest.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, manifest)
    return output
