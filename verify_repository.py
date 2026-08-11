from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'code'
RESULTS = ROOT / 'results'
CHECKPOINTS = ROOT / 'checkpoints'
CONFIG = ROOT / 'config' / 'experiment_config.json'
sys.path.insert(0, str(CODE))
import revised_framework as rf  # noqa: E402

checks = []

def check(name: str, ok: bool, detail: str = '') -> None:
    checks.append((name, bool(ok), detail))
    if not ok:
        raise AssertionError(f'{name}: {detail}')

with CONFIG.open(encoding='utf-8') as f:
    config = json.load(f)

params_dict = dict(config['params'])
params = rf.SystemParams(**params_dict)
check('2 devices', params.num_devices == 2)
check('2 edge servers', params.num_edge_servers == 2)
check('cloud capacity 420 GFLOPS', math.isclose(params.cloud_cap_gflops, 420.0))
check('100 training episodes', config['training_episodes'] == 100)
check('30 ablation episodes', config['ablation_episodes'] == 30)
check('10 evaluation seeds', config['evaluation_seeds'] == list(range(1000, 1010)))

raw = pd.read_csv(RESULTS / 'results_raw.csv')
summary = pd.read_csv(RESULTS / 'results_summary.csv')
stats = pd.read_csv(RESULTS / 'paired_statistics.csv')
ab_raw = pd.read_csv(RESULTS / 'ablation_raw.csv')
ab_summary = pd.read_csv(RESULTS / 'ablation_summary.csv', header=[0,1], index_col=0)
history = pd.read_csv(RESULTS / 'training_history.csv')

check('240 raw rows', len(raw) == 240, str(len(raw)))
check('400 training-history rows', len(history) == 400, str(len(history)))
check('four scenarios in history', set(history['scenario']) == {'normal','burst','fluctuation','energy_constrained'})
check('10 seeds per ablation variant', bool((ab_raw.groupby('variant')['seed'].nunique() == 10).all()))

regen_summary = rf.summarize_results(raw).sort_values(['scenario','policy']).reset_index(drop=True)
stored_summary = summary.sort_values(['scenario','policy']).reset_index(drop=True)
check('summary columns match', list(regen_summary.columns) == list(stored_summary.columns))
num_cols = regen_summary.select_dtypes('number').columns
max_summary_diff = float(np.nanmax(np.abs(regen_summary[num_cols].to_numpy() - stored_summary[num_cols].to_numpy())))
check('summary regenerates from raw results', max_summary_diff < 1e-10, f'max diff={max_summary_diff}')

regen_stats = rf.paired_statistics(raw).sort_values(['baseline','scenario','metric']).reset_index(drop=True)
stored_stats = stats.sort_values(['baseline','scenario','metric']).reset_index(drop=True)
check('statistics keys match', regen_stats[['baseline','scenario','metric']].equals(stored_stats[['baseline','scenario','metric']]))
max_stats_diff = float(np.nanmax(np.abs(regen_stats[['statistic','p_value','p_holm']].to_numpy() - stored_stats[['statistic','p_value','p_holm']].to_numpy())))
check('Wilcoxon/Holm results regenerate', max_stats_diff < 1e-12, f'max diff={max_stats_diff}')

regen_ab = ab_raw.groupby('variant')[['latency_ms','device_energy_j','accuracy','violation_rate','utilization','depletion']].agg(['mean','std'])
ab_summary = ab_summary.loc[regen_ab.index]
max_ab_diff = float(np.nanmax(np.abs(regen_ab.to_numpy() - ab_summary.to_numpy(dtype=float))))
check('ablation summary regenerates', max_ab_diff < 1e-10, f'max diff={max_ab_diff}')

agents = {}
for scenario in ['normal','burst','fluctuation','energy_constrained']:
    env = rf.CloudEdgeEnv(params, scenario, 42)
    agent = rf.PPOAgent(env.state_dim, env.action_dim, seed=42)
    state = torch.load(CHECKPOINTS / f'ppo_{scenario}.pt', map_location='cpu', weights_only=True)
    agent.policy.load_state_dict(state)
    agents[scenario] = agent

reproduced = rf.run_multi_seed_evaluation(params, agents, ['normal','burst','fluctuation','energy_constrained'], list(range(1000,1010)))
keys = ['scenario','seed','policy']
metrics = ['tasks','latency_ms','device_energy_j','accuracy','violation_rate','utilization','depletion','mean_reward']
reproduced = reproduced.sort_values(keys).reset_index(drop=True)
stored = raw.sort_values(keys).reset_index(drop=True)
check('checkpoint result keys match', reproduced[keys].equals(stored[keys]))
max_raw_diff = float(np.nanmax(np.abs(reproduced[metrics].to_numpy() - stored[metrics].to_numpy())))
check('checkpoints reproduce stored raw results', max_raw_diff < 1e-10, f'max diff={max_raw_diff}')

print('REPOSITORY VERIFICATION PASSED')
for name, ok, detail in checks:
    suffix = f' | {detail}' if detail else ''
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{suffix}")
print(f'Checks passed: {sum(ok for _,ok,_ in checks)}/{len(checks)}')
