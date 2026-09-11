"""Create a self-contained receiving-PC bundle; never include raw/test data or credentials."""
from pathlib import Path
import argparse
import json
import zipfile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from forecast_v2.common import verify_data, sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    verify_data(a.data);files={}
    for folder in ('forecast_v2','rtx3050'):
        for f in (ROOT/folder).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py','.md','.ps1','.json'):
                files[f.relative_to(ROOT).as_posix()]=f
    for relative in ('configs/forecast_v2_rtx3050.json','configs/forecast_v2_long_horizon.json','publication/FORECAST_V2_RESEARCH.md','artifacts/forecast-v2-bias-diagnostic.json','artifacts/forecast-v2-verification.json','tests/test_forecast_v2.py'):
        path=ROOT/relative
        if not path.exists():raise FileNotFoundError(path)
        files[relative]=path
    for f in a.data.glob('*'):
        if f.is_file() and f.suffix in ('.npz','.json'):files[f'data/{f.name}']=f
    manifest={name:sha(f) for name,f in files.items()}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    temp=a.out.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for name,f in files.items():archive.write(f,name)
        archive.writestr('PACKAGE_SHA256.json',json.dumps(manifest,indent=2))
    temp.replace(a.out)
    with zipfile.ZipFile(a.out) as archive:
        if archive.testzip() is not None:raise ValueError('Archive integrity failed')
    print(json.dumps({'archive':str(a.out.resolve()),'bytes':a.out.stat().st_size,'sha256':sha(a.out),'files':len(files)},indent=2))


if __name__=='__main__':main()
