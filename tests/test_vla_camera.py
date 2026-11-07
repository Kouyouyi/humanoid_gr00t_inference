import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launcher', ROOT / 'run_real_vla.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PYTHON = os.environ.get('GROOT_TEST_PYTHON', sys.executable)


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gr00t-real-test-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def wait_for(self, function, timeout=10):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            result = function()
            if result:
                return result
            time.sleep(.03)
        self.fail('timeout')

    def test_preview_two_recordings_and_shutdown_real_video(self):
        log = (self.root / 'camera.log').open('wb')
        proc = subprocess.Popen([PYTHON, str(ROOT / 'vla_camera_view.py'),
            '--root', str(self.root), '--repo', str(m.DEFAULT_REPO), '--fixture'],
            stdout=log, stderr=subprocess.STDOUT)
        def command(action, **values):
            request_id = __import__('uuid').uuid4().hex
            m.write(self.root / 'camera-command.json', dict(id=request_id, action=action, **values))
            return self.wait_for(lambda: (s if (s := m.read(self.root / 'camera-state.json')).get('ack') == request_id else None))
        try:
            self.wait_for(lambda: m.read(self.root / 'camera-state.json').get('has_frame'))
            viewer = m.read(self.root / 'camera-viewer.json')['url']
            with urlopen(viewer) as response:
                self.assertIn('相机第一视角'.encode(), response.read())
            with self.assertRaises(HTTPError):urlopen(viewer.split('?')[0])
            image_url = viewer.replace('/?', '/frame.jpg?')
            with urlopen(image_url) as response:
                self.assertEqual(response.read()[:2], b'\xff\xd8')
            folder = self.root / 'recordings/first'
            state = command('start', folder=str(folder))
            self.assertIsNone(state['command_error'])
            self.assertTrue(state['recording'])
            start = time.monotonic()
            time.sleep(.7)
            state = command('stop', reason='operator_pause')
            self.assertIsNone(state['command_error'])
            result = m.read(folder / 'recording.json')
            self.assertEqual(result['status'], 'saved')
            self.assertTrue((folder / 'camera.mp4').is_file())
            self.assertGreater(result['frames'], 15)
            self.assertLess(abs(float(result['probe']['duration'])-result['wall_duration_s']), 1/30+.02)
            self.assertGreater(result['duplicate_frames'], 0)
            subprocess.run(['ffmpeg','-v','error','-i',str(folder / 'camera.mp4'),'-f','null','-'],check=True,capture_output=True)
            # Second P must create a separate clip; q finalizes that clip before exit.
            folder2 = self.root / 'recordings/second'
            command('start', folder=str(folder2))
            time.sleep(.2)
            command('shutdown', reason='operator_q')
            proc.wait(timeout=10)
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(m.read(folder2 / 'recording.json')['status'],'saved')
            self.assertFalse(m.read(self.root / 'camera-state.json')['online'])
            self.assertEqual(len(list((self.root / 'recordings').glob('*/camera.mp4'))),2)
        finally:
            if proc.poll() is None:
                proc.terminate();proc.wait(timeout=15)
            log.close()

    def test_start_records_before_command_and_failed_submit_stops_recording(self):
        c = m.Console.__new__(m.Console)
        c.pending = None;c.prepare_job = None;c.job = None
        c.native = Mock();c.native.alive.return_value=True
        events=[]
        c.start_recording=lambda:events.append('record')
        def fail(action):
            events.append('submit');raise RuntimeError('busy')
        c.submit=fail
        c.stop_recording=lambda reason:events.append('save')
        with self.assertRaises(RuntimeError):c.start_vla()
        self.assertEqual(events,['record','submit','save'])

    def test_vla_fault_finalizes_video_without_restarting(self):
        c = m.Console.__new__(m.Console)
        c.model=c.owner=None;c.pending=None;c.reported_fault=None;c.prepare_job=None
        c.run=self.root;c.job='job'
        (self.root / 'vla_results').mkdir()
        m.write(self.root / 'vla_results/job.json',dict(reason='owner_fault: example'))
        c.state=lambda:{}
        c.stop_recording=Mock()
        c.tick()
        c.stop_recording.assert_called_once_with('vla_result: owner_fault: example')
        self.assertIsNone(c.job)

    def test_q_stops_native_then_viewport_then_model(self):
        c=m.Console.__new__(m.Console)
        c.owner=None;c.native=None;c.job=None;c.prepare_job=None
        c.telemetry=None;c.model=Mock()
        events=[]
        c.stop_recording=lambda reason:events.append('save')
        c.stop_body=lambda:events.append('native')
        c.close_camera=lambda:events.append('camera')
        c.model.stop=lambda:events.append('model')
        c.close()
        self.assertEqual(events,['save','native','camera','model'])


if __name__ == '__main__':unittest.main()
