#!/usr/bin/env python3
"""Rebuild the bundled pinned native source with system SDKs. Never starts a robot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def recipe(output):
    manifest=json.loads((ROOT/'native/bin/manifest.json').read_text())
    commands=manifest['commands']
    source=next(arg for arg in commands[0] if arg.endswith('/g1_deploy_onnx_ref.cpp'))
    old_build=source.split('/src/g1/')[0]
    old_sdk=next(arg for arg in commands[0] if arg.endswith('/thirdparty/unitree_sdk2/include'))
    old_deploy=old_sdk.rsplit('/thirdparty/',1)[0]
    def translated(arg):
        arg=arg.replace(old_build+'/src',str(ROOT/'native/source'))
        arg=arg.replace(old_build,str(output))
        arg=arg.replace(old_deploy+'/thirdparty',str(ROOT/'native/thirdparty'))
        arg=arg.replace('../../audio_thread/libaudio_thread.a',str(ROOT/'native/lib/libaudio_thread.a'))
        arg=arg.replace('../../TRTInference/libTRTInference.a',str(ROOT/'native/lib/libTRTInference.a'))
        arg=arg.replace('/home/kinetra-5090-01/TensorRT',os.environ.get('TensorRT_ROOT','/home/kinetra-5090-01/TensorRT'))
        arg=arg.replace('/opt/onnxruntime',os.environ.get('onnxruntime_ROOT','/opt/onnxruntime'))
        arg=arg.replace('/usr/local/cuda-12.8',os.environ.get('CUDAToolkit_ROOT','/usr/local/cuda-12.8'))
        return arg
    return manifest,[[translated(arg) for arg in command] for command in commands]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'native/build')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();output=args.output.resolve()
    manifest,commands=recipe(output)
    if args.dry_run:
        print(json.dumps(commands,indent=2));return
    output.mkdir(parents=True,exist_ok=False)
    for i,command in enumerate(commands):
        print(f'Building native component {i+1}/{len(commands)}',flush=True)
        with (output/f'build-{i}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=output/'g1_field'
    manifest.update(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        commands=commands,hardware_validated=False,standalone_build=True,link_cwd=str(ROOT))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Built '+str(binary)+'; existing deployed native/bin was not replaced.')

if __name__=='__main__':main()
