#!/usr/bin/env python3
"""Deterministic, bounded exploratory ODD candidate generation.
Never claims to satisfy symbolic runtime arm constraints; non-executable arms are flagged.
Usage: python build_scenario_candidates.py hypotheses.generated.yaml scenario_space.example.yaml -o candidates.yaml
"""
import argparse,hashlib,random,pathlib,yaml,math,re

def draw(spec,rng):
    if 'choices' in spec:return rng.choice(spec['choices'])
    return round(rng.uniform(float(spec['min']),float(spec['max'])),6)

def make(h,space,n,seed):
    dimensions=space['dimensions'];g=h.get('scenario_guidance',{})
    names=list(dict.fromkeys(g.get('odd_dimensions',[])+g.get('fault_dimensions',[])))
    rows=[]
    rng=random.Random(seed+int(hashlib.sha256(h['id'].encode()).hexdigest()[:8],16))
    for arm in ('condition','control'):
        definitions=h[arm]
        for idx in range(n):
            params={name:draw(dimensions[name],rng) for name in names if name in dimensions}
            # Only explicit scalar numeric/string selectors are wired into scenario parameters.
            # Constraints expressed as formulae / labels remain unbound and require worker support.
            unbound=[]
            for key,value in definitions.items():
                keymap={'road_friction':'road_friction','vehicle_mass_scale':'vehicle_mass_scale',
                        'speed_mps':'speed_mps','maneuver':'maneuver'}
                if key in keymap and key in params and isinstance(value,(int,float)):
                    params[key]=value
                elif key in keymap and key in params and isinstance(value,str) and value in (dimensions[key].get('choices') or []):
                    params[key]=value
                elif key in ('speed_mps','road_curvature_per_m') and key in params and isinstance(value,str) and re.fullmatch(r'\s*[<>]\s*-?\d+(?:\.\d+)?\s*',value):
                    comparator=value.strip()[0];threshold=float(value.strip()[1:]);bound=dimensions[key]
                    low=max(float(bound['min']), threshold+0.000001) if comparator=='>' else float(bound['min'])
                    high=min(float(bound['max']), threshold-0.000001) if comparator=='<' else float(bound['max'])
                    if low>=high:unbound.append({'unreachable_bound':{key:value}})
                    else:params[key]=round(rng.uniform(low,high),6)
                elif key=='fault_types' and isinstance(value,list):
                    params['fault_type']=rng.choice(value) if value else None
                else:
                    unbound.append({key:value})
            # Necessary physical-screening condition, not a validated feasibility certificate.
            v=params.get('speed_mps');k=params.get('road_curvature_per_m');mu=params.get('road_friction')
            physical_screen='not_evaluated'
            if all(isinstance(x,(float,int)) for x in (v,k,mu)):
                ay=v*v*abs(k);params['lateral_accel_demand_mps2']=round(ay,6)
                physical_screen='pass_necessary_lateral_accel_screen' if ay<=mu*9.81 else 'reject_exceeds_mu_g'
            rows.append({'candidate_id':f"{h['id']}-{arm[0].upper()}-{idx+1:03d}",
                         'hypothesis_id':h['id'],'arm':arm,'parameters':params,
                         'unbound_conditions':unbound,'physical_screen':physical_screen,
                         'executable':not unbound and physical_screen!='reject_exceeds_mu_g',
                         'note':'Exploratory proposal only: simulator adapters and matching checks still required'})
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('hypotheses');ap.add_argument('scenario_space');ap.add_argument('--per-arm',type=int,default=8);ap.add_argument('--seed',type=int,default=7);ap.add_argument('-o','--out',default='candidates.yaml');a=ap.parse_args()
    if a.per_arm<1 or a.per_arm>10000:ap.error('per-arm must be 1..10000')
    hy=yaml.safe_load(pathlib.Path(a.hypotheses).read_text());sp=yaml.safe_load(pathlib.Path(a.scenario_space).read_text())
    rows=[]
    for h in hy['hypotheses']:
        if h['status']=='proposed':rows.extend(make(h,sp,a.per_arm,a.seed))
    output={'meta':{'kind':'unverified_scenario_candidates','seed':a.seed,'per_arm':a.per_arm,'ready_for_carla':False},'candidates':rows}
    pathlib.Path(a.out).write_text(yaml.safe_dump(output,allow_unicode=True,sort_keys=False,width=110))
    print('generated',len(rows),'candidate specifications;',sum(x['executable'] for x in rows),'fully bound (not yet CARLA validated)')
if __name__=='__main__':main()
