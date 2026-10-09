"""Small real-DNN CPU timing sanity check for the cloud-edge study.

This script intentionally does not download datasets or pretrained weights. It measures
forward-pass wall-clock time for standard torchvision architectures with random weights
and a fixed 1x3x224x224 tensor. It is an external execution sanity check, not a claim of
edge/cloud deployment performance and not an energy measurement.
"""
from __future__ import annotations
import json, os, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet50, mobilenet_v2


def bench(model, x, warmup=8, repeats=40):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        times=[]
        for _ in range(repeats):
            t0=time.perf_counter(); model(x); times.append((time.perf_counter()-t0)*1000.0)
    a=np.asarray(times)
    return dict(n=repeats, mean_ms=float(a.mean()), sd_ms=float(a.std(ddof=1)), median_ms=float(np.median(a)), p05_ms=float(np.quantile(a,.05)), p95_ms=float(np.quantile(a,.95)))


def main():
    root=Path(__file__).resolve().parents[1]; out=root/'results'; out.mkdir(exist_ok=True)
    torch.manual_seed(2026); torch.set_num_threads(1)
    x=torch.randn(1,3,224,224)
    rows=[]
    for name,model in [('ResNet-50',resnet50(weights=None)),('MobileNetV2',mobilenet_v2(weights=None))]:
        r=bench(model,x); rows.append({'model':name,'device':'CPU','threads':torch.get_num_threads(),'input_shape':'1x3x224x224','weights':'random/untrained',**r})
    pd.DataFrame(rows).to_csv(out/'hardware_microbenchmark.csv',index=False)
    
    cpu_model=''
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.lower().startswith('model name'):
                cpu_model=line.split(':',1)[1].strip(); break
    except Exception:
        cpu_model=platform.processor()
    info={'python':platform.python_version(),'torch':torch.__version__,'platform':platform.platform(),'processor':cpu_model or platform.processor(),'machine':platform.machine(),'torch_threads':torch.get_num_threads(),'note':'Timing-only sanity check; no dataset accuracy and no power/energy measurement.'}
    with open(out/'hardware_microbenchmark_system.json','w') as f: json.dump(info,f,indent=2)
    print(pd.DataFrame(rows).to_string(index=False)); print(info)
if __name__=='__main__': main()
