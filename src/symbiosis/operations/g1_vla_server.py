"""Serve the exact home RTC bundle on localhost; no robot imports or commands."""
from symbiosis.common.project import project_root
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=project_root()
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from symbiosis.robots.g1.vla import DEFAULT_MODEL
from symbiosis.robots.g1.vla import DEFAULT_RTC
from symbiosis.robots.g1.vla_runtime import identity


def prepare(bundle,rtc_repo,output):
    spec=importlib.util.spec_from_file_location('rax_model_launcher',rtc_repo/'scripts/local/pick_cola_rtc.py')
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    # exFAT results volumes cannot host the staging symlinks. This cache holds
    # small config files and links only; weights and all logs stay on data disk.
    launcher.REPO_ROOT=Path(os.environ.get('G1_MODEL_CACHE',str(Path.home()/'.cache/g1-sim/model-staging')))
    local=launcher.prepare_model(bundle.parent,bundle.name)
    output.mkdir(parents=True,exist_ok=True)
    (output/'staging.json').write_text(json.dumps(dict(model_bundle=str(bundle),staged_model=str(local)),indent=2))
    return local


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',type=Path,default=Path(DEFAULT_MODEL))
    p.add_argument('--rtc-repo',type=Path,default=Path(DEFAULT_RTC))
    p.add_argument('--output',type=Path,default=ROOT/'runtime/runs/vla-model')
    p.add_argument('--port',type=int,default=5550)
    p.add_argument('--rtc-delay-steps',type=int,choices=range(1,11),default=8,
                   help='Deployment prefix delay; the model also validates its trained maximum')
    p.add_argument('--cpu-threads',type=int,default=1,
                   help='PyTorch CPU intra-op threads; 1 avoids batch-one image preprocessing overhead')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if a.cpu_threads < 1:p.error('--cpu-threads must be positive')
    bundle=a.model_path.resolve()
    local=prepare(bundle,a.rtc_repo.resolve(),a.output.resolve())
    selected=identity(bundle)
    (a.output/'identity.json').write_text(json.dumps(selected,indent=2)+'\n')
    print(json.dumps(dict(model=selected,staged_path=str(local))),flush=True)
    if a.prepare_only:return
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
    sys.path.insert(0,str(a.rtc_repo.resolve()))
    import torch
    torch.set_num_threads(a.cpu_threads)
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer
    policy=Gr00tPolicy('UNITREE_G1_SONIC',str(local),device='cuda',sonic_token_mode='continuous',rtc_delay_steps=a.rtc_delay_steps)
    from symbiosis.operations.run_timing import Journal
    journal=Journal(Path(os.environ.get('G1_TIMING_ROOT',str(a.output/'timing')))/'vla-server.jsonl')
    original=policy.get_action
    def timed_action(*args,**kwargs):
        with journal.span('vla_server_get_action',measurement='server wall: preprocessing + inference + postprocessing'):
            return original(*args,**kwargs)
    with PolicyServer(policy,host='127.0.0.1',port=a.port) as server:
        server.register_endpoint('get_action',timed_action)
        server.register_endpoint('get_rax_identity',lambda:selected,requires_input=False)
        print(f'RAX home VLA ready: continuous, delay={a.rtc_delay_steps}, '
              f'cpu_threads={torch.get_num_threads()}, localhost:{a.port}',flush=True)
        server.run()


if __name__=='__main__':main()
