import argparse
import json
from pathlib import Path
from .common import read, write


def main():
    parser=argparse.ArgumentParser(description='Portable regional crowd forecasting development study')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--study',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('train');p.add_argument('--data',required=True);p.add_argument('--run',required=True);p.add_argument('--config',required=True);p.add_argument('--seed',type=int,default=23037);p.add_argument('--mode',choices=['transport','direct','state_only'],default='transport');p.add_argument('--max-steps',type=int);p.add_argument('--device')
    for name in ('pause','status'):
        p=sub.add_parser(name);p.add_argument('--run',required=True)
    p=sub.add_parser('evaluate');p.add_argument('--data',required=True);p.add_argument('--run',required=True);p.add_argument('--device')
    p=sub.add_parser('diagnose');p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('doctor');p.add_argument('--data',required=True);p.add_argument('--config',required=True)
    p=sub.add_parser('stress');p.add_argument('--data',required=True);p.add_argument('--run',required=True);p.add_argument('--device')
    args=parser.parse_args()
    if args.command=='prepare':
        from .data import prepare
        prepare(args.study,args.out)
    elif args.command=='train':
        from .train import train
        config=read(args.config);config.update(seed=args.seed,mode=args.mode)
        train(args.data,args.run,config,args.max_steps,args.device)
    elif args.command=='pause':
        run=Path(args.run)
        if not run.is_dir(): raise ValueError('Run does not exist')
        (run/'PAUSE').touch();print('Pause requested; wait for status=paused before switching off.')
    elif args.command=='status':
        print(json.dumps(read(Path(args.run)/'status.json'),indent=2))
    elif args.command=='evaluate':
        from .evaluate import evaluate
        evaluate(args.data,args.run,args.device)
    elif args.command=='diagnose':
        from .evaluate import diagnose
        diagnose(args.data,args.out)
    elif args.command=='stress':
        from .stress import stress
        stress(args.data,args.run,args.device)
    else:
        from .evaluate import doctor
        doctor(args.data,args.config)


if __name__=='__main__':main()
