"""Failure clips stay independent of model inference: python pipeline/09_monitor/test_replay.py."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dashboard
import replay
from video_clip import ffmpeg_exe


def record(folder=None):
    r = {'run_id': 'clip-test', 'scenario_id': 'S1', 'sim_status': 'valid',
         'outcome_category': 'F2_switch_transient', 'actual_fault_onset': 10.0,
         'switch_time': 10.2, 'evaluation_window': [10, 20],
         'sr': {'SR4': {'first_violation_t': 12.0}}, 'sr_measures': {}}
    if folder:
        r['run_dirs'] = {'fault': str(folder)}
    return r


class ReplayTests(unittest.TestCase):
    def test_window_focuses_on_switch_for_transient_failure(self):
        self.assertEqual(replay.clip_window(record()), (9.0, 13.2, 10.2))
        r = record()
        r['outcome_category'] = 'F3_not_recovered'
        self.assertEqual(replay.clip_window(r), (9.0, 15.0, 12.0))
        r['actual_fault_onset'] = 0
        r['switch_time'] = None
        r['sr'] = {}
        self.assertEqual(replay.clip_window(r), (0, 3, 0))

    def test_missing_footage_is_explicit_and_passes_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            r = record()
            rows = replay.prepare_replays([r, {**r, 'run_id': 'pass', 'outcome_category': 'pass'}], d)
            self.assertEqual(list(rows), ['clip-test'])
            self.assertIsNone(rows['clip-test']['clip'])
            self.assertIn('녹화 경로', rows['clip-test']['error'])

    def test_codec_failure_does_not_stop_dashboard(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'source'
            source.mkdir()
            (source / 'view.mp4').write_bytes(b'not a video')
            with patch('replay.make_clip', return_value=None):
                rows = replay.prepare_replays([record(source)], Path(d) / 'screen')
            self.assertIn('ffmpeg', rows['clip-test']['error'])

    @unittest.skipUnless(ffmpeg_exe(), 'ffmpeg not available')
    def test_real_h264_clip_frame_mapping_and_cache(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'source'
            source.mkdir()
            subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'lavfi', '-i',
                            'testsrc=size=320x180:rate=20:duration=6', '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', str(source / 'view.mp4')], check=True)
            (source / 'view_frames.jsonl').write_text(''.join(json.dumps({'t': 9 + k / 20}) + '\n' for k in range(120)))
            screen = Path(d) / 'screen'
            rows = replay.prepare_replays([record(source)], screen)
            clip = rows['clip-test']
            self.assertTrue((screen / clip['clip']).is_file())
            self.assertTrue(clip['exact_timing'])
            with patch('replay.make_clip', side_effect=AssertionError('cache not used')):
                cached = replay.prepare_replays([record(source)], screen)
            self.assertEqual(rows, cached)
            # A source change must invalidate the clip cache.
            (source / 'view_frames.jsonl').write_text(json.dumps({'t': 9}) + '\n')
            with patch('replay.make_clip', return_value=None) as make:
                replay.prepare_replays([record(source)], screen)
                make.assert_called_once()

    def test_layout_keeps_replay_nemotron_and_results_sections(self):
        page = dashboard.render_page([record()], insights={'summary': 'demo'}, replays={
            'clip-test': {'clip': 'clips/sample.mp4', 'start_s': 9, 'end_s': 13.2, 'event_s': 10.2,
                          'source': 'view.mp4', 'exact_timing': True}})
        self.assertLess(page.index('<video'), page.index('id="insight-section"'))
        self.assertIn('취약 가설 · 통계 검증', page)
        self.assertNotIn('01 · 코드 분석', page)
        self.assertIn('Cosmos Reason 분석 결과', page)
        self.assertIn('제어기 전환 순간', page)
        self.assertLess(page.index('id="nemotron-section"'), page.index('id="results-column"'))
        self.assertLess(page.index('id="verdict-section"'), page.index('id="insight-section"'))
        self.assertEqual(page.count('<video'), 1)


if __name__ == '__main__':
    unittest.main()
