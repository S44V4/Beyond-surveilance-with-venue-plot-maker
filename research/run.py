"""Run the entire fixed evidence protocol, resuming after interruption."""
import ctypes
import json
import os
import shutil
import time
import traceback
from research.protocol import ROOT,STUDY,PROTOCOL,freeze,atomic_json,sha

def alive(pid):
    if os.name!='nt':
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return False
    code=ctypes.c_ulong();ok=kernel.GetExitCodeProcess(handle,ctypes.byref(code));kernel.CloseHandle(handle)
    return bool(ok and code.value==259)

def sources():
    files=[*ROOT.joinpath('research').glob('*.py'),*ROOT.joinpath('crowd_twin/models').glob('*.py'),ROOT/'crowd_twin/venue.py',ROOT/'crowd_twin/contracts.py',ROOT/'crowd_twin/data/strfe.py',ROOT/'backend/balanced_ddpf.py',ROOT/'configs/venues/dronecrowd_analysis_grid.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

def verify_sources():
    path=STUDY/'source-manifest.json';current=sources()
    if path.exists():
        if json.loads(path.read_text())!=current:raise ValueError('Research source changed since launch. Preserve the run and review an explicit protocol amendment before resuming.')
    else:
        atomic_json(path,current)
        for relative in current:
            target=STUDY/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)

def main():
    os.chdir(ROOT);STUDY.mkdir(parents=True,exist_ok=True);lock=STUDY/'run.lock'
    if lock.exists():
        previous=json.loads(lock.read_text())
        if alive(previous['pid']):raise RuntimeError(f"Study already running as PID {previous['pid']}")
        lock.unlink()
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as file:json.dump({'pid':os.getpid(),'started_unix':time.time()},file)
    try:
        freeze();verify_sources()
        if (STUDY/'EVALUATION_COMPLETE.json').exists():print('Evidence run already complete; see REPORT.md');return
        from research.cache import extract,calibrate
        from research.training import train_all
        from research.evaluation import evaluate
        from research.report import report
        extract();verify_sources();calibrate();train_all();verify_sources();evaluate();report()
        atomic_json(STUDY/'status.json',{'stage':'complete','updated_unix':time.time(),'decision':json.loads((STUDY/'EVALUATION_COMPLETE.json').read_text())['decision']})
    except BaseException as error:
        atomic_json(STUDY/'failure.json',{'error':str(error),'traceback':traceback.format_exc(),'updated_unix':time.time()});raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
