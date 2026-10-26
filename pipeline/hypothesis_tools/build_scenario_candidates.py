#!/usr/bin/env python3
"""Deterministic, bounded exploratory ODD candidate generation.
Never claims to satisfy symbolic runtime arm constraints; non-executable arms are flagged.
Usage (repository root): python -m pipeline.hypothesis_tools.build_scenario_candidates -o candidates.yaml
Lateral demand above the P-02 threshold is tagged stress_test (G-03), never rejected.
"""
import argparse,hashlib,random,pathlib,re,yaml

from ..prior_tools.prior_tools import screen_lateral_demand
from .space import normalize_space

INEQUALITY=re.compile(r'\s*([<>])\s*([+-]?\d+(?:\.\d+)?)\s*')
NEAR=re.compile(r'\s*near\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*')

def draw(spec,rng):
    if 'choices' in spec:return rng.choice(spec['choices'])
    return round(rng.uniform(float(spec['min']),float(spec['max'])),6)

def bind(value,spec,rng):
    """Return (bound, value). Only scalars, choice subsets, '< x' / '> x' and near(x, tol) are bound."""
    if 'choices' in spec:
        if isinstance(value,(str,bool)) and value in spec['choices']:return True,value
        if isinstance(value,list) and value and all(v in spec['choices'] for v in value):return True,rng.choice(value)
        return False,None
    low,high=float(spec['min']),float(spec['max'])
    if isinstance(value,(int,float)) and not isinstance(value,bool):
        return (low<=value<=high),value
    if not isinstance(value,str):return False,None
    m=INEQUALITY.fullmatch(value)
    if m:
        threshold=float(m.group(2))
        if m.group(1)=='>':low=max(low,threshold+1e-6)
        else:high=min(high,threshold-1e-6)
    else:
        m=NEAR.fullmatch(value)
        if not m:return False,None
        center,tol=float(m.group(1)),float(m.group(2))
        low,high=max(low,center-tol),min(high,center+tol)
    if low>=high:return False,None
    return True,round(rng.uniform(low,high),6)

def make(h,space,n,seed,max_lateral_acc=None):
    dimensions=space['dimensions'];g=h.get('scenario_guidance',{})
    names=list(dict.fromkeys(g.get('odd_dimensions',[])+g.get('fault_dimensions',[])))
    rows=[]
    rng=random.Random(seed+int(hashlib.sha256(h['id'].encode()).hexdigest()[:8],16))
    # Non-scenario keys identical in both arms (profile facts such as hold_last) do not affect
    # arm comparability; they are carried as preconditions for the runner to verify.
    shared={k:v for k,v in h['condition'].items()
            if k not in dimensions and k not in ('fault_types','fault_params','co_injected_faults')
            and h['control'].get(k)==v}
    for arm in ('condition','control'):
        definitions=h[arm]
        for idx in range(n):
            params={name:draw(dimensions[name],rng) for name in names if name in dimensions}
            # Constraints expressed as formulae / labels remain unbound and require worker support.
            unbound=[]
            if isinstance(definitions.get('fault_types'),list):
                params['fault_type']=rng.choice(definitions['fault_types']) if definitions['fault_types'] else None
            if isinstance(definitions.get('co_injected_faults'),list):
                params['co_injected_faults']=list(definitions['co_injected_faults'])
            active={params.get('fault_type'),*params.get('co_injected_faults',[])}
            # Drop drawn parameters of faults this candidate does not inject.
            params={k:v for k,v in params.items() if '.' not in k or k.split('.')[0] in active}
            for key,value in definitions.items():
                if key in ('fault_types','co_injected_faults') or key in shared:
                    continue
                if key=='fault_params' and isinstance(value,dict):
                    for param,pvalue in value.items():
                        dim=f"{params.get('fault_type')}.{param}"
                        ok,bound=bind(pvalue,dimensions[dim],rng) if dim in dimensions else (False,None)
                        if ok:params[dim]=bound
                        else:unbound.append({dim:pvalue})
                elif key in dimensions:
                    ok,bound=bind(value,dimensions[key],rng)
                    if ok:params[key]=bound
                    else:unbound.append({key:value})
                else:
                    unbound.append({key:value})
            # Steady-state screening only (prior PK-001); stress candidates stay executable.
            screen=screen_lateral_demand(params,max_lateral_acc)
            if 'nominal_lateral_acc_mps2' in screen:params['nominal_lateral_acc_mps2']=round(screen['nominal_lateral_acc_mps2'],6)
            rows.append({'candidate_id':f"{h['id']}-{arm[0].upper()}-{idx+1:03d}",
                         'hypothesis_ids':[h['id']],'arm':arm,'metric':h['prediction']['metric'],
                         'parameters':params,'preconditions':shared,'unbound_conditions':unbound,
                         'lateral_screen':screen['tag'],
                         'executable':not unbound,
                         'note':'Exploratory proposal only: simulator adapters and matching checks still required'})
    return rows

def lateral_threshold(safety_spec):
    props=((safety_spec or {}).get('safety_specification') or {}).get('properties') or []
    return next((p.get('lateral_acc_threshold_mps2') for p in props if p.get('id')=='P-02'),None)

def main():
    root=pathlib.Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hypotheses',default=str(root/'02_hypotheses/hypotheses.yaml'))
    ap.add_argument('--scenario-space',default=str(root/'04_feasibility_expansion/scenario_space.yaml'))
    ap.add_argument('--safety-spec',default=str(root/'09_monitor/safety_spec.yaml'))
    ap.add_argument('--per-arm',type=int,default=8);ap.add_argument('--seed',type=int,default=7);ap.add_argument('-o','--out',default='candidates.yaml');a=ap.parse_args()
    if a.per_arm<1 or a.per_arm>10000:ap.error('per-arm must be 1..10000')
    def read(path):return yaml.safe_load(pathlib.Path(path).read_text(encoding='utf-8'))
    hy=read(a.hypotheses);sp=normalize_space(read(a.scenario_space));threshold=lateral_threshold(read(a.safety_spec))
    rows=[]
    for h in hy['hypotheses']:
        if h['status']=='proposed':rows.extend(make(h,sp,a.per_arm,a.seed,threshold))
    output={'meta':{'kind':'unverified_scenario_candidates','seed':a.seed,'per_arm':a.per_arm,
                    'lateral_acc_threshold_mps2':threshold,'ready_for_carla':False},'candidates':rows}
    pathlib.Path(a.out).write_text(yaml.safe_dump(output,allow_unicode=True,sort_keys=False,width=110),encoding='utf-8')
    print('generated',len(rows),'candidate specifications;',sum(x['executable'] for x in rows),'fully bound (not yet CARLA validated)')
if __name__=='__main__':main()
