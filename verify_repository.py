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
FIGURES = ROOT / 'figures'
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
params = rf.SystemParams(**config['params'])

check('4 source devices', params.num_devices == 4)
check('3 edge servers', params.num_edge_servers == 3)
check('1 finite cloud node at 120 GFLOPS', math.isclose(params.cloud_cap_gflops, 120.0))
check('aggregate capacities remain comparable', max(sum(params.device_caps_gflops), sum(params.edge_caps_gflops), params.cloud_cap_gflops) / min(sum(params.device_caps_gflops), sum(params.edge_caps_gflops), params.cloud_cap_gflops) < 1.5)
check('45 s episode horizon', math.isclose(params.horizon_s, 45.0))
check('20 final matched evaluation seeds', config['evaluation_seeds'] == list(range(1000, 1020)))
check('three PPO training seeds', len(config['ppo_training_seeds']) == 3)
check('three DQN training seeds', len(config['dqn_training_seeds']) == 3)
check('9 actions', len(config['actions']) == 9)
check('17-state implementation', rf.CloudEdgeEnv.state_dim == 17)

expected = [
    RESULTS/'results_raw_all_replicates.csv', RESULTS/'results_paired_aggregated.csv',
    RESULTS/'results_summary.csv', RESULTS/'paired_statistics_global_holm.csv',
    RESULTS/'constant_action_validation.csv', RESULTS/'scenario_coverage.csv',
    RESULTS/'state_dependence_probe.csv', RESULTS/'sensitivity_raw.csv',
    RESULTS/'sensitivity_summary.csv', RESULTS/'ppo_training_history.csv',
    RESULTS/'dqn_training_history.csv', RESULTS/'hardware_microbenchmark.csv',
]
for p in expected:
    check(f'file exists: {p.name}', p.exists() and p.stat().st_size > 0)

for seed in config['ppo_training_seeds']:
    p = CHECKPOINTS/f'ppo_unified_seed_{seed}.pt'
    check(f'PPO checkpoint {seed}', p.exists() and p.stat().st_size > 0)
for seed in config['dqn_training_seeds']:
    p = CHECKPOINTS/f'dqn_unified_seed_{seed}.pt'
    check(f'DQN checkpoint {seed}', p.exists() and p.stat().st_size > 0)
for i in range(1, 7):
    for ext in ('svg', 'png'):
        matches = list(FIGURES.glob(f'figure_{i}_*.{ext}'))
        check(f'Figure {i} {ext}', len(matches) == 1, str(matches))

raw = pd.read_csv(RESULTS/'results_raw_all_replicates.csv')
agg = pd.read_csv(RESULTS/'results_paired_aggregated.csv')
summary = pd.read_csv(RESULTS/'results_summary.csv')
stats = pd.read_csv(RESULTS/'paired_statistics_global_holm.csv')
coverage = pd.read_csv(RESULTS/'scenario_coverage.csv')
probe = pd.read_csv(RESULTS/'state_dependence_probe.csv')
sens = pd.read_csv(RESULTS/'sensitivity_summary.csv')
hw = pd.read_csv(RESULTS/'hardware_microbenchmark.csv')

check('four scenarios in aggregated results', set(agg['scenario']) == set(rf.SCENARIOS))
check('20 evaluation seeds per scenario/policy', bool((agg.groupby(['scenario','policy'])['eval_seed'].nunique() == 20).all()))
check('three learned training replicas in raw results', raw[raw.policy=='PPO']['training_rep'].nunique() == 3 and raw[raw.policy=='DQN']['training_rep'].nunique() == 3)
check('deterministic and stochastic PPO are reported', {'PPO','PPO-stochastic'}.issubset(set(agg.policy)))
check('all comparator families are present', {'Local-only','Edge-only','Cloud-only','Static-rule','Greedy-myopic','Best-constant','DQN'}.issubset(set(agg.policy)))

regen_summary = rf.summarize_for_manuscript(agg).sort_values(['scenario','policy']).reset_index(drop=True)
stored_summary = summary.sort_values(['scenario','policy']).reset_index(drop=True)
check('summary columns match', list(regen_summary.columns) == list(stored_summary.columns))
num_cols = regen_summary.select_dtypes('number').columns
max_summary_diff = float(np.nanmax(np.abs(regen_summary[num_cols].to_numpy() - stored_summary[num_cols].to_numpy())))
check('summary regenerates from paired aggregate', max_summary_diff < 1e-10, f'max diff={max_summary_diff:.3g}')

regen_stats = rf.paired_statistics(agg).sort_values(['scenario','comparator','metric']).reset_index(drop=True)
stored_stats = stats.sort_values(['scenario','comparator','metric']).reset_index(drop=True)
key_cols = ['scenario','comparator','metric','n_pairs','testable']
check('statistical comparison keys match', regen_stats[key_cols].equals(stored_stats[key_cols]))
cols = ['mean_difference_ppo_minus_comparator','ci95_low','ci95_high','rank_biserial','statistic','p_value','p_holm_global']
a = regen_stats[cols].to_numpy(dtype=float); b = stored_stats[cols].to_numpy(dtype=float)
max_stats_diff = float(np.nanmax(np.abs(a-b)))
check('global-Holm statistics regenerate', max_stats_diff < 1e-10, f'max diff={max_stats_diff:.3g}')
check('single global test family has 84 rows', len(stats) == 4*7*3, str(len(stats)))

for seed in config['evaluation_seeds']:
    b = set(coverage[(coverage.scenario=='burst') & (coverage.seed==seed)].phase)
    f = set(coverage[(coverage.scenario=='fluctuation') & (coverage.seed==seed)].phase)
    check(f'burst/recovery coverage seed {seed}', b == {'burst','recovery'}, str(b))
    check(f'four fluctuation phases seed {seed}', f == {'phase_0','phase_1','phase_2','phase_3'}, str(f))

check('state-dependence probe includes all PPO seeds', set(probe.training_seed.astype(int)) == set(config['ppo_training_seeds']))
check('PPO argmax is state-dependent', bool((probe['distinct_argmax_actions'] >= 2).all()), probe[['training_seed','distinct_argmax_actions']].to_dict('records').__str__())
check('PPO probes use 20,000 states each', bool((probe['n_states'] == 20000).all()))
check('PPO probe entropy is below uniform maximum', bool((probe['mean_entropy'] < probe['max_entropy']).all()))

energy = summary[summary.scenario=='energy'].set_index('policy')
check('energy constraint is binding for local-only', float(energy.loc['Local-only','drop_rate_mean']) > 0.5, str(float(energy.loc['Local-only','drop_rate_mean'])))
check('PPO avoids energy-infeasible drops on average', float(energy.loc['PPO','drop_rate_mean']) < 1e-12, str(float(energy.loc['PPO','drop_rate_mean'])))

constant = summary.set_index(['scenario','policy'])
for sc in rf.SCENARIOS:
    check(f'PPO improves mean reward over best constant in {sc}', float(constant.loc[(sc,'PPO'),'mean_reward_mean']) > float(constant.loc[(sc,'Best-constant'),'mean_reward_mean']))

check('sensitivity covers three parameters', set(sens.parameter) == {'arrival_scale','bandwidth_scale','capacity_scale'})
check('sensitivity uses three levels each', bool((sens.groupby('parameter')['value'].nunique() == 3).all()))
check('hardware timing check includes ResNet-50 and MobileNetV2', set(hw.model) == {'ResNet-50','MobileNetV2'})
check('hardware timing uses 40 runs each', bool((hw.n == 40).all()))

# Structural checkpoint load check (fast; does not retrain).
for seed in config['ppo_training_seeds']:
    env = rf.CloudEdgeEnv(params, 'normal', 1000)
    agent = rf.PPOAgent(env.state_dim, env.action_dim, seed=seed)
    agent.policy.load_state_dict(torch.load(CHECKPOINTS/f'ppo_unified_seed_{seed}.pt', map_location='cpu', weights_only=True))
    state = env.reset(1000)
    action, _, _, entropy = agent.select_action(state, deterministic=True)
    check(f'PPO checkpoint {seed} loads and acts', 0 <= action < 9 and np.isfinite(entropy))

print('REPOSITORY VERIFICATION PASSED')
for name, ok, detail in checks:
    suffix = f' | {detail}' if detail else ''
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{suffix}")
print(f'Checks passed: {sum(ok for _,ok,_ in checks)}/{len(checks)}')
