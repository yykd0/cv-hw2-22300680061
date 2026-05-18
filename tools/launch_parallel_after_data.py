from __future__ import annotations
import os, subprocess, time
from pathlib import Path
import yaml
ROOT=Path('/root/autodl-tmp/cv_hw2_submit')
PYTHON='/root/miniconda3/bin/python'
LOG=ROOT/'runs/logs/parallel_launcher.log'
LOG.parent.mkdir(parents=True, exist_ok=True)

def log(s):
    with LOG.open('a', encoding='utf-8') as f:
        f.write(time.strftime('%F %T ') + s + '\n')
    print(s, flush=True)

def ready():
    base=ROOT/'data/oxford_iiit_pet/oxford-iiit-pet'
    return (base/'images').is_dir() and (base/'annotations').is_dir()

def load_yaml(p):
    return yaml.safe_load(Path(p).read_text(encoding='utf-8'))

def write_yaml(p,d):
    Path(p).write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return Path(p)

def make_cls(src,name,epochs=8,batch=128,pretrained=None):
    cfg=load_yaml(ROOT/'configs'/src); cfg['output_dir']=f'runs/{name}'; cfg['data']['root']='data/oxford_iiit_pet'; cfg['data']['download']=False; cfg['data']['num_workers']=8; cfg['train']['epochs']=epochs; cfg['train']['batch_size']=batch; cfg['logging']['backend']='csv'; cfg['logging']['run_name']=name
    if pretrained is not None: cfg['model']['pretrained']=pretrained
    return write_yaml(ROOT/'configs'/f'{name}.yaml', cfg)

def make_seg(src,name,epochs=10,batch=64):
    cfg=load_yaml(ROOT/'configs'/src); cfg['output_dir']=f'runs/{name}'; cfg['data']['root']='data/oxford_iiit_pet'; cfg['data']['download']=False; cfg['data']['num_workers']=8; cfg['train']['epochs']=epochs; cfg['train']['batch_size']=batch; cfg['logging']['backend']='csv'; cfg['logging']['run_name']=name
    return write_yaml(ROOT/'configs'/f'{name}.yaml', cfg)

log('waiting for Oxford dataset extraction')
while not ready():
    time.sleep(20)
log('Oxford dataset ready; stopping serial orchestrator if still alive')
subprocess.run(['pkill','-f','tools/run_full_hw2.py'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# Launch independent jobs. Keep batch moderate so several can share the large GPU.
jobs=[]
for src,name,pre in [
 ('task1_resnet18_imagenet.yaml','final_task1_resnet18_imagenet',None),
 ('task1_resnet18_scratch.yaml','final_task1_resnet18_scratch',None),
 ('task1_resnet18_se.yaml','final_task1_resnet18_se',None),
 ('task1_resnet18_cbam.yaml','final_task1_resnet18_cbam',None),
 ('task1_vit_tiny.yaml','final_task1_vit_tiny',False),
]:
    cfg=make_cls(src,name,pretrained=pre)
    if not (ROOT/'runs'/name/'metrics.csv').exists():
        log('launch '+name)
        p=subprocess.Popen([PYTHON,'scripts/train_pet_cls.py','--config',str(cfg)], cwd=ROOT, stdout=(ROOT/'runs/logs'/f'{name}.parallel.log').open('w'), stderr=subprocess.STDOUT)
        jobs.append((name,p))
for src,name in [
 ('task3_unet_ce.yaml','final_task3_unet_ce'),
 ('task3_unet_dice.yaml','final_task3_unet_dice'),
 ('task3_unet_ce_dice.yaml','final_task3_unet_ce_dice'),
]:
    cfg=make_seg(src,name)
    if not (ROOT/'runs'/name/'metrics.csv').exists():
        log('launch '+name)
        p=subprocess.Popen([PYTHON,'scripts/train_pet_seg.py','--config',str(cfg)], cwd=ROOT, stdout=(ROOT/'runs/logs'/f'{name}.parallel.log').open('w'), stderr=subprocess.STDOUT)
        jobs.append((name,p))
for name,p in jobs:
    rc=p.wait(); log(f'{name} finished rc={rc}')
log('all oxford jobs finished; finalize report can run after yolo/video decision')
