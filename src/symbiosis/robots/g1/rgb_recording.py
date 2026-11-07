"""Continuous shared RGB encoder for HEAD and external camera normalized video."""
from fractions import Fraction
import json
from pathlib import Path
from symbiosis.experience.execution_memory import atomic

class RGBRecording:
    def __init__(self,directory,width,height,*,profile=None,filename="third-person.mp4",source="external_realsense_full_received_color_stream"):
        import av
        from .camera_profile import load
        profile=profile or load();self.profile=profile;self.source=source
        frame=profile["frame"];encoding=profile["encoding"]
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/filename
        if self.path.exists():raise FileExistsError('Refusing to overwrite existing RGB recording: '+str(self.path))
        self.output=av.open(str(self.path),'w',options=dict(movflags='frag_keyframe+empty_moov+default_base_moof',flush_packets='1'))
        self.stream=self.output.add_stream(encoding['codec'],rate=frame['fps'])
        self.stream.width=width;self.stream.height=height;self.stream.pix_fmt=encoding['pixel_format']
        self.stream.time_base=Fraction(1,90000);self.stream.codec_context.time_base=Fraction(1,90000)
        self.stream.codec_context.thread_count=encoding['threads']
        self.stream.options=dict(preset=encoding['preset'],crf=str(encoding['crf']),tune='zerolatency',bf='0',g=str(encoding['keyframe_interval']))
        self.index=(self.directory/('first-frames.jsonl' if filename.startswith('first') else 'third-frames.jsonl')).open('x')
        self.first=None;self.last_pts=-1;self.count=0
        self.status('recording')
    def status(self,state,**extra):
        atomic(self.directory/'recording.json',dict(state=state,source=self.source,
            video=str(self.path),frames=self.count,preview_sampling=False,clock='host_frame_receive_monotonic',width=self.stream.width,height=self.stream.height,nominal_fps=self.profile['frame']['fps'],encoding=self.profile['encoding'],**extra))
    def append(self,pixels,received,unix,hardware_ms):
        import av
        if self.first is None:self.first=received
        pts=max(self.last_pts+1,round((received-self.first)*90000));self.last_pts=pts
        frame=av.VideoFrame.from_ndarray(pixels,format='rgb24');frame.pts=pts;frame.time_base=Fraction(1,90000)
        for packet in self.stream.encode(frame):self.output.mux(packet)
        self.index.write(json.dumps(dict(frame=self.count,pts=pts,time_base='1/90000',received_monotonic=received,
            received_unix_s=unix,hardware_timestamp_ms=hardware_ms))+'\n')
        self.count+=1
        if self.count%self.profile['frame']['fps']==0:self.index.flush();self.status('recording')
    def close(self,*,verify=True):
        for packet in self.stream.encode(None):self.output.mux(packet)
        self.output.close();self.index.close()
        import av
        decoded=None
        if verify:
            with av.open(str(self.path)) as video:decoded=sum(1 for _ in video.decode(video=0))
        audit=dict(video=str(self.path),frames=self.count,decoded_frames=decoded,
            full_decode_passed=(decoded==self.count if verify else None),verification_pending=not verify,continuous=True,preview_sampling=False,
            clock='host_frame_receive_monotonic',duration_s=self.last_pts/90000 if self.count else 0,
            reencoding_after_stop=False,width=self.stream.width,height=self.stream.height,nominal_fps=self.profile['frame']['fps'],encoding=self.profile['encoding'])
        atomic(self.directory/'video-audit.json',audit);self.status('complete',full_decode_passed=audit['full_decode_passed'])
        return audit
