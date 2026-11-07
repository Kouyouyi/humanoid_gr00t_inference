#!/usr/bin/env python3
"""Read-only dependency/asset audit. Never create robot channels or load GPU weights."""
import argparse
import importlib
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime_config import bootstrap, SOURCE_PATHS
bootstrap()

class LocalApplicationOnly:
    """Block fallback to another editable application checkout."""
    names={'symbiosis','gear_sonic','gr00t','unitree_sdk2py'}
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] not in self.names:return None
        spec=PathFinder.find_spec(fullname,path)
        if spec is None:raise ImportError('Standalone source missing: '+fullname)
        locations=list(spec.submodule_search_locations or [])
        if spec.origin and spec.origin not in ('built-in','frozen'):locations.append(spec.origin)
        if any(not Path(p).resolve().is_relative_to(ROOT) for p in locations):
            raise ImportError('External application source forbidden: '+fullname+': '+str(locations))
        return spec


def worker(part,model):
    sys.meta_path.insert(0,LocalApplicationOnly())
    from symbiosis.common.deployment import load,external_path
    cfg=load()
    report={'part':part,'source_import_guard':True,'robot_channels_created':0,'gpu_weights_loaded':False,'checks':[]}
    for name in ('rtc_root','sonic_root','native_build','native_env','native_workdir','controller_assets','controller_planner','controller_reference','robot_model_assets'):
        path=external_path(name)
        assert path.resolve().is_relative_to(ROOT),(name,path)
        assert path.exists(),(name,path)
    report['checks'].append('all application/control paths are inside checkout')
    if part=='model':
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
        from gr00t.policy.server_client import PolicyServer
        from gr00t.policy.rtc_client import AsyncRTCController
        report['checks'].append('GR00T policy/model/RTC/server imports')
    else:
        from symbiosis.robots.g1.rtc_adapter import dependencies,robot_model,BraincoModelAdapter
        rtc=dependencies(external_path('rtc_root'))
        report['checks'].append('SONIC observation/camera/session imports')
        robot=robot_model()
        report['checks'].append('Pinocchio robot model and meshes load')
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        report['checks'].append('copied Unitree DDS SDK imports without initializing channels')
        from symbiosis.robots.g1.ankle_executor import run
        from symbiosis.operations.g1_prepare_native import prepare
        from symbiosis.task_runtime import default_bundle
        with tempfile.TemporaryDirectory(prefix='gr00t-standalone-audit-') as temp:
            runpath=Path(temp)/'run'
            prepare(runpath,vla=True,task_bundle=default_bundle())
            import run_real_vla as launcher
            launcher.adapt_snapshot(runpath,16551,'chunk')
            owner=json.loads((runpath/'manifest.json').read_text())
            assert str(ROOT/'native/bin/g1_field') in owner['controller']
            assert str(ROOT/'vendor/unitree') in owner['owner'][1]
            report['checks'].append('native scripts + guarded unlimited owner snapshot prepare without starting processes')
        if model:
            adapter=BraincoModelAdapter(model,None)
            report['checks'].append('selected checkpoint BrainCo/SONIC layouts validate')
    report['loaded_application_files']={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if name.split('.')[0] in LocalApplicationOnly.names and getattr(module,'__file__',None)}
    print('STANDALONE_AUDIT='+json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=('model','control'))
    parser.add_argument('--model',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'runtime/standalone-audit.json')
    args=parser.parse_args()
    if args.worker:return worker(args.worker,args.model)
    from symbiosis.common.deployment import external_path
    reports=[]
    for part,python in [('model',external_path('rtc_python')),('control',external_path('inference_python'))]:
        command=[str(python),str(Path(__file__).resolve()),'--worker',part]
        if args.model:command+=['--model',str(args.model)]
        env=os.environ|{'PYTHONPATH':os.pathsep.join(str(p) for p in [ROOT,*SOURCE_PATHS]),'PYTHONDONTWRITEBYTECODE':'1'}
        result=subprocess.run(command,env=env,cwd=ROOT,capture_output=True,text=True,timeout=180)
        if result.returncode:
            raise RuntimeError(part+' audit failed:\n'+result.stdout+result.stderr)
        line=next(x for x in result.stdout.splitlines() if x.startswith('STANDALONE_AUDIT='))
        reports.append(json.loads(line.split('=',1)[1]))
        print(part+': '+str(len(reports[-1]['checks']))+' checks passed',flush=True)
    native=subprocess.run(['bash','-c','source native/env.sh\nldd native/bin/g1_field'],cwd=ROOT,text=True,capture_output=True,check=True)
    if 'not found' in native.stdout:raise RuntimeError(native.stdout)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=True,reports=reports,native_libraries=native.stdout),indent=2)+'\n')
    print('All read-only standalone checks passed: '+str(args.output))

if __name__=='__main__':main()
