"""Lossless continuous camera recording, independent of RGB-D/UI sampling.

Persist every received H264 payload and its host arrival clock before decoding or
dropping preview frames. On close, remux (no encoding) using the recorded clock.
An interrupted raw capture can be finalized offline with this module's CLI.
This module never imports robot control or sends robot commands.
"""
import argparse
from collections import deque
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time


def write_json(path, value):
    temporary = path.with_name('.'+path.name)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


class PacketRecording:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        # A capture directory is owned by exactly one camera-service invocation.
        self.raw = (self.directory/'camera-stereo.h264').open('xb')
        try:
            self.index = (self.directory/'camera-packets.jsonl').open('x')
        except Exception:
            self.raw.close()
            raise
        self.digest = hashlib.sha256()
        self.count = self.size = 0
        self.first = self.last = None
        self.updated = time.monotonic()
        self.closed = False
        self.error = None
        self.status('recording')

    def status(self, state, **extra):
        write_json(self.directory/'recording.json', dict(
            schema='continuous_camera_recording_v1', state=state,
            source='received_original_h264_stereo', preview_sampling=False,
            clock='workstation_packet_receive_monotonic_not_exposure',
            packets=self.count, bytes=self.size,
            first_received_monotonic=self.first, last_received_monotonic=self.last,
            raw=str(self.directory/'camera-stereo.h264'),
            packet_index=str(self.directory/'camera-packets.jsonl'),
            updated_unix_s=time.time(), error=self.error, **extra))

    def append(self, data, received_monotonic, received_unix_s=None):
        if self.closed or self.error:
            return
        try:
            self.raw.write(data)
            self.index.write(json.dumps(dict(sequence=self.count, offset=self.size,
                size=len(data), received_monotonic=received_monotonic,
                received_unix_s=time.time() if received_unix_s is None else received_unix_s))+'\n')
            self.digest.update(data)
            self.first = received_monotonic if self.first is None else self.first
            self.last = received_monotonic
            self.count += 1
            self.size += len(data)
            if received_monotonic-self.updated >= 1:
                self.raw.flush()
                self.index.flush()
                os.fsync(self.raw.fileno())
                os.fsync(self.index.fileno())
                self.updated = received_monotonic
                self.status('recording')
        except Exception as exc:
            # Recording trouble is visible, but never becomes a motion gate.
            self.error = f'{type(exc).__name__}: {exc}'
            try:
                self.status('error')
            except OSError:
                pass

    def close(self, reason='camera_service_stopped'):
        if self.closed:
            return
        self.closed = True
        for stream in (self.raw, self.index):
            stream.close()
        self.status('sealed', stop_reason=reason, raw_sha256=self.digest.hexdigest())
        if self.error or not self.count:
            self.status('error', stop_reason=reason)
            return
        try:
            self.status('finalizing', stop_reason=reason, raw_sha256=self.digest.hexdigest())
            audit = finalize(self.directory)
            self.status('complete', stop_reason=reason, raw_sha256=self.digest.hexdigest(),
                        video=audit['video'], audit=str(self.directory/'video-audit.json'),
                        full_decode_passed=audit['full_decode_passed'])
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.status('error', stop_reason=reason, raw_sha256=self.digest.hexdigest(),
                        recovery='Raw H264 and arrival index retained; finalize offline.')


def finalize(directory):
    """Remux Annex B packets with their recorded arrival time; bounded memory."""
    import av

    directory = Path(directory)
    raw = directory/'camera-stereo.h264'
    destination = directory/'camera-stereo-walltime.mp4'
    temporary = directory/'camera-stereo-walltime.partial.mp4'
    time_base = Fraction(1, 1_000_000)
    packets = 0
    first = previous_tick = last = None
    intervals = deque(maxlen=120)
    # Both parsers visit byte offsets in increasing order. No full video/index
    # or decoded images are accumulated in memory.
    with (directory/'camera-packets.jsonl').open() as index, av.open(str(raw), format='h264') as source:
        entry = json.loads(next(index))
        input_stream = source.streams.video[0]
        rate = input_stream.average_rate or Fraction(30, 1)
        nominal_ticks = max(1, round(1_000_000/float(rate)))
        pending = None
        with av.open(str(temporary), 'w', format='mp4', options={'movflags':'+faststart'}) as output:
            stream = output.add_stream_from_template(input_stream)
            stream.time_base = time_base
            for packet in source.demux(input_stream):
                if not packet.size:
                    continue
                if packet.pos is None or packet.pos < 0:
                    raise ValueError('H264 demuxer did not expose source byte offsets')
                # Timestamp when the complete access unit had arrived, including
                # streams whose transport splits one frame into several packets.
                end = packet.pos+packet.size-1
                while end >= entry['offset']+entry['size']:
                    entry = json.loads(next(index))
                stamp = entry['received_monotonic']
                first = stamp if first is None else first
                tick = round((stamp-first)*1_000_000)
                tick = max(tick, previous_tick+1) if previous_tick is not None else tick
                packet.pts = packet.dts = tick
                packet.time_base = time_base
                packet.stream = stream
                if pending is not None:
                    pending.duration = tick-previous_tick
                    intervals.append(tick-previous_tick)
                    output.mux(pending)
                pending, previous_tick, last = packet, tick, stamp
                packets += 1
            if pending is None:
                raise ValueError('No complete camera frames in recorded H264')
            pending.duration = round(median(intervals)) if intervals else nominal_ticks
            output.mux(pending)
    decoded = 0
    with av.open(str(temporary)) as video:
        width, height = video.streams.video[0].width, video.streams.video[0].height
        duration = float(video.duration/av.time_base) if video.duration is not None else None
        for _ in video.decode(video=0):
            decoded += 1
    if decoded != packets:
        raise ValueError(f'Camera full-decode frame count differs: {decoded}/{packets}')
    temporary.replace(destination)
    audit = dict(video=str(destination), full_decode_passed=True,
        encoded_access_units=packets, decoded_frames=decoded, width=width, height=height,
        first_received_monotonic=first, last_received_monotonic=last,
        measured_receive_span_s=last-first, video_duration_s=duration,
        timeline_clock='measured_host_packet_receive_time',
        source='continuous_original_h264_before_preview_sampling',
        reencoded=False, preview_sampling=False, third_person_used=False)
    write_json(directory/'video-audit.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    audit=finalize(args.directory)
    status=args.directory/'recording.json'
    previous=json.loads(status.read_text()) if status.exists() else {}
    write_json(status,dict(previous,state='complete',error=None,recovered_offline=True,
        video=audit['video'],audit=str(args.directory/'video-audit.json'),full_decode_passed=True))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
