#!/usr/bin/env python3
"""Generate immutable, evidence-linked MPC/LQR hypotheses and scenario search specs.
Usage: python generate_hypotheses.py safety_profile.example.yaml vulnerability_patterns.yaml \
  --scenario-space scenario_space.example.yaml -o hypotheses.yaml
The generated YAML is a PROPOSAL, not executable CARLA scenarios or validated safety claims.
"""
from __future__ import annotations
import argparse, ast, copy, datetime as dt, json, math, pathlib, re
import yaml

FIELD = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_.]*)\}")
SINGLE = re.compile(r"^\{([a-zA-Z_][a-zA-Z0-9_.]*)\}$")

def get_path(obj, path):
    for key in path.split('.'):
        if not isinstance(obj,dict) or key not in obj:return None
        obj=obj[key]
    return obj

def missing(value):return value is None or value == [] or value == {}

def evidence_index(profile):
    out={}; ids=set()
    for ob in ((profile.get('code_evidence') or {}).get('observations') or []):
        if not isinstance(ob,dict) or 'id' not in ob or 'field' not in ob:continue
        if ob['id'] in ids:raise ValueError('Duplicate observation ID: '+str(ob['id']))
        ids.add(ob['id']);out.setdefault(ob['field'],[]).append(ob['id'])
    return out,ids

# Arithmetic and comparisons are AST-whitelisted; never use eval().
OPS={ast.Add:lambda a,b:a+b,ast.Sub:lambda a,b:a-b,ast.Mult:lambda a,b:a*b,
     ast.Div:lambda a,b:a/b,ast.Mod:lambda a,b:a%b}
CMP={ast.Eq:lambda a,b:a==b,ast.NotEq:lambda a,b:a!=b,ast.Lt:lambda a,b:a<b,
     ast.LtE:lambda a,b:a<=b,ast.Gt:lambda a,b:a>b,ast.GtE:lambda a,b:a>=b}

def safe_expr(text):
    def ev(node):
        if isinstance(node,ast.Constant) and type(node.value) in (bool,int,float,str,type(None)):return node.value
        if isinstance(node,ast.Name) and node.id in ('True','False','None','true','false','null'):return {'True':True,'False':False,'None':None,'true':True,'false':False,'null':None}[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.Not,ast.USub,ast.UAdd)):
            v=ev(node.operand);return not v if isinstance(node.op,ast.Not) else (-v if isinstance(node.op,ast.USub) else +v)
        if isinstance(node,ast.BoolOp) and isinstance(node.op,(ast.And,ast.Or)):
            vals=[ev(x) for x in node.values];return all(vals) if isinstance(node.op,ast.And) else any(vals)
        if isinstance(node,ast.BinOp) and type(node.op) in OPS:return OPS[type(node.op)](ev(node.left),ev(node.right))
        if isinstance(node,ast.Compare):
            a=ev(node.left)
            for operator,child in zip(node.ops,node.comparators):
                b=ev(child)
                if type(operator) not in CMP or not CMP[type(operator)](a,b):return False
                a=b
            return True
        raise ValueError('Expression not allowed: '+type(node).__name__)
    return ev(ast.parse(text,mode='eval').body)

def interpolate(obj, profile):
    if isinstance(obj,str):
        m=SINGLE.fullmatch(obj)
        if m:return copy.deepcopy(get_path(profile,m.group(1)))
        return FIELD.sub(lambda match: str(get_path(profile,match.group(1))),obj)
    if isinstance(obj,list):return [interpolate(x,profile) for x in obj]
    if isinstance(obj,dict):return {k:interpolate(v,profile) for k,v in obj.items()}
    return obj

def apply(expr,profile):
    paths=FIELD.findall(expr)
    absent=[p for p in paths if missing(get_path(profile,p))]
    if absent:return None,'Missing applies_if fields: '+', '.join(absent)
    rendered=FIELD.sub(lambda m:repr(get_path(profile,m.group(1))),expr)
    try:
        val=safe_expr(rendered)
        if not isinstance(val,bool):raise ValueError('applies_if must evaluate to bool')
        return val,rendered+' => '+str(val)
    except (ValueError,TypeError,ZeroDivisionError,SyntaxError,OverflowError) as exc:
        return None,'Invalid applies_if: '+str(exc)

def validate_space(guidance,space):
    dims=(space or {}).get('dimensions') or {}
    requested=(guidance.get('odd_dimensions') or [])+(guidance.get('fault_dimensions') or [])
    return [x for x in requested if x not in dims]

def validate_llm(prop,obs_ids,cat,space):
    issues=[]
    if not isinstance(prop,dict):return ['proposal is not an object']
    if not prop.get('based_on') or not isinstance(prop.get('based_on'),list) or not set(map(str,prop['based_on']))<=obs_ids:issues.append('unknown or missing evidence IDs')
    pred=prop.get('prediction')
    if not isinstance(pred,dict) or pred.get('metric') not in cat['metric_rules'] or pred.get('direction') not in ('positive','negative','non_inferior'):issues.append('unknown metric/direction')
    for side in ('condition','control'):
        if not isinstance(prop.get(side),dict):issues.append('missing '+side)
    faults=(space or {}).get('fault_types') or []
    for side in ('condition','control'):
        if isinstance(prop.get(side),dict) and any(x not in faults for x in prop[side].get('fault_types',[])):
            issues.append('unknown fault_types in '+side)
    if prop.get('arm_comparability_verified') is not True:issues.append('arm comparability not verified')
    # This flag alone is not an independent proof. Human/manual approval required.
    if prop.get('reviewed_for_physical_validity') is not True:issues.append('physical validity review required')
    return issues

def generate(profile,cat,space=None,proposals=None):
    ev,obs_ids=evidence_index(profile)
    result={'meta':{'schema_version':'2.0','generated_by':'generate_hypotheses.py',
      'profile_id':(profile.get('meta') or {}).get('profile_id'),
      'source_commit':((profile.get('meta') or {}).get('source_revision') or {}).get('commit'),
      'catalog_version':cat['meta']['catalog_version'],
      'generated_at':dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
      'frozen_at_commit':None,'status':'generated_not_validated'},
      'references':copy.deepcopy(cat.get('references',{})),
      'outcome_definitions':copy.deepcopy(cat.get('outcome_definitions',{})),
      'recoverability_oracle_spec':copy.deepcopy(cat.get('recoverability_oracle_spec',{})),
      'metric_rules':copy.deepcopy(cat.get('metric_rules',{})),
      'pattern_application':[],'hypotheses':[],'rejected_proposals':[],'tracking':[]}
    for p in cat['patterns']:
        pid=p['pattern_id'];ok,reason=apply(p['applies_if'],profile)
        if ok is False:
            result['pattern_application'].append({'pattern_id':pid,'applies':False,'reason':reason});continue
        missing_fields=[f for f in p.get('needs',[]) if missing(get_path(profile,f))]
        missing_scenario=validate_space(p.get('scenario_guidance',{}),space) if space is not None else []
        absent_placeholders=[]
        for side in ('condition_template','control_template'):
            # Missing substitutions are blocked rather than silently rendered as 'None'.
            def visit(obj):
                if isinstance(obj,str):absent_placeholders.extend([q for q in FIELD.findall(obj) if missing(get_path(profile,q))])
                elif isinstance(obj,list):
                    for x in obj:visit(x)
                elif isinstance(obj,dict):
                    for x in obj.values():visit(x)
            visit(p[side])
        blocking=[]
        if ok is None:blocking.append(reason)
        if missing_fields:blocking.append('missing profile values: '+', '.join(missing_fields))
        if absent_placeholders:blocking.append('missing template values: '+', '.join(sorted(set(absent_placeholders))))
        if missing_scenario:blocking.append('missing scenario dimensions: '+', '.join(missing_scenario))
        if p.get('requires_oracle'):
            oracle=profile.get('recoverability') or {}
            if any(missing(oracle.get(field)) for field in ('P_matrix','level_c','reach_table')):
                blocking.append('recoverability oracle artifacts not certified/populated')
        pred=copy.deepcopy(p['prediction']);metric=pred['metric']
        base_rule=copy.deepcopy(cat['metric_rules'].get(metric))
        if base_rule is None:blocking.append('unknown metric: '+metric)
        rule={**(base_rule or {}),**p.get('decision_rule',{})}
        h={'id':p['target_id'],'from_pattern':pid,'phase':p.get('phase'),
           'provenance':'pattern_instantiation','claim':p['title'],
           'based_on':sorted({oid for field in p.get('needs',[]) for oid in ev.get(field,[])}),
           'filled_values':{field:get_path(profile,field) for field in p.get('needs',[])},
           'condition':interpolate(copy.deepcopy(p['condition_template']),profile),
           'control':interpolate(copy.deepcopy(p['control_template']),profile),
           'scenario_guidance':copy.deepcopy(p.get('scenario_guidance',{})),
           'prediction':pred,'secondary_prediction':copy.deepcopy(p.get('optional_secondary_prediction',[])),
           'evidence':copy.deepcopy(p.get('source_evidence',[])),
           'decision_rule':rule,'status':'blocked' if blocking else 'proposed'}
        if p.get('design_notes'):h['design_notes']=p['design_notes']
        if blocking:h['blocked_reason']=blocking
        result['hypotheses'].append(h)
        result['pattern_application'].append({'pattern_id':pid,'applies':ok,'reason':reason,'hypothesis_id':h['id']})
    for prop in proposals or []:
        why=validate_llm(prop,obs_ids,cat,space)
        if why:result['rejected_proposals'].append({'proposal':prop,'reasons':why});continue
        # Even reviewed proposals stay blocked until separately approved; never silently add untrusted tests.
        result['rejected_proposals'].append({'proposal':prop,'reasons':['manual approval required before catalog inclusion']})
    for h in result['hypotheses']:
        result['tracking'].append({'id':h['id'],'runs_condition':0,'runs_control':0,'effect_estimate':None,'ci':None,'mismatch_runs':0,'status_history':[]})
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('profile');ap.add_argument('catalog');ap.add_argument('--scenario-space');ap.add_argument('--llm-proposals');ap.add_argument('-o','--out',default='hypotheses.yaml')
    a=ap.parse_args()
    def read(path):return yaml.safe_load(pathlib.Path(path).read_text(encoding='utf-8'))
    profile=read(a.profile);cat=read(a.catalog)
    space=read(a.scenario_space) if a.scenario_space else None
    props=read(a.llm_proposals) if a.llm_proposals else []
    output=generate(profile,cat,space,props)
    pathlib.Path(a.out).write_text(yaml.safe_dump(output,allow_unicode=True,sort_keys=False,width=110),encoding='utf-8')
    for h in output['hypotheses']:print(h['id'],h['status'],h['from_pattern'])
    print('wrote',a.out)
if __name__=='__main__':main()
