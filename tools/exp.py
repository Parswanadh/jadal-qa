#!/usr/bin/env python3
"""Shared experiment registry (stdlib only). Every experiment = one folder + one registry line.
  exp.py new    --id E1.01 --track E1 --title "..." --why "..." --hypothesis "..." --data "..." --regime "..." --method "..." --owner NAME
  exp.py finish --id E1.01 --verdict adopt|maybe|drop|inconclusive --summary "..." --results path/to/results.json [--decision "..."] [--cost "..."]
  exp.py list [--track E1]
  exp.py render            -> experiments/EXPERIMENTS.md (human-readable log of everything)
Registry: /home/parshu/projects/jadal-qa-wt/experiments/registry.jsonl (append-only; latest line per id wins)."""
import argparse, json, os, sys, time, glob
R=os.environ.get('JADAL_EXP_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','experiments'); REG=R+'/registry.jsonl'
def load():
    d={}
    if os.path.exists(REG):
        for l in open(REG):
            if l.strip(): e=json.loads(l); d.setdefault(e['id'],{}).update(e)
    return d
def append(e):
    os.makedirs(R,exist_ok=True)
    with open(REG,'a') as f: f.write(json.dumps(e)+'\n')
def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True)
    n=sp.add_parser('new'); [n.add_argument('--'+k,required=(k in('id','title','why','hypothesis','data','method','owner'))) for k in('id','track','title','why','hypothesis','data','regime','method','owner')]
    f=sp.add_parser('finish'); [f.add_argument('--'+k,required=(k in('id','verdict','summary'))) for k in('id','verdict','summary','results','decision','cost')]
    l=sp.add_parser('list'); l.add_argument('--track'); sp.add_parser('render')
    a=ap.parse_args(); now=time.strftime('%Y-%m-%d %H:%M')
    if a.cmd=='new':
        e={k:v for k,v in vars(a).items() if k!='cmd' and v is not None}; e.update(status='running',started=now); append(e)
        d=f"{R}/{a.id}"; os.makedirs(d,exist_ok=True)
        open(d+'/README.md','w').write(f"# {a.id} — {a.title}\n\n**Why:** {a.why}\n\n**Hypothesis:** {a.hypothesis}\n\n**Data / label source:** {a.data}\n\n**Data regime:** {getattr(a,'regime',None) or 'n/a'}\n\n**Method:** {a.method}\n\n**Owner:** {a.owner}  **Started:** {now}\n\n## Results\n(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)\n")
        print('registered',a.id,'->',d)
    elif a.cmd=='finish':
        d=load(); 
        if a.id not in d: sys.exit('unknown id '+a.id)
        e={'id':a.id,'status':'done','finished':now,'verdict':a.verdict,'summary':a.summary}
        for k in('results','decision','cost'):
            if getattr(a,k): e[k]=getattr(a,k)
        append(e); open(f"{R}/{a.id}/README.md",'a').write(f"\n**Verdict ({now}): {a.verdict}.** {a.summary}\n\nResults file: {a.results or 'n/a'}\n\nDecision: {a.decision or 'n/a'}\n\nCost: {a.cost or 'n/a'}\n"); print('finished',a.id)
    elif a.cmd=='list':
        for i,e in sorted(load().items()):
            if not a.track or e.get('track')==a.track: print(f"{i:8s} {e.get('status','?'):8s} {e.get('verdict','-'):13s} {e.get('title','')[:80]}")
    elif a.cmd=='render':
        d=load(); out=["# Experiment log\n","Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.\n",
                       "| id | track | status | verdict | title | data / regime |","|---|---|---|---|---|---|"]
        for i,e in sorted(d.items()): out.append(f"| {i} | {e.get('track','')} | {e.get('status','')} | {e.get('verdict','-')} | {e.get('title','')} | {e.get('data','')[:60]} / {e.get('regime','') or ''} |")
        out.append("")
        for i,e in sorted(d.items()):
            out+= [f"## {i} — {e.get('title','')}",f"- **Why:** {e.get('why','')}",f"- **Hypothesis:** {e.get('hypothesis','')}",f"- **Data / label source:** {e.get('data','')}",f"- **Regime:** {e.get('regime','n/a')}",f"- **Method:** {e.get('method','')}",f"- **Owner / started / finished:** {e.get('owner','')} / {e.get('started','')} / {e.get('finished','—')}",
                   f"- **Verdict:** {e.get('verdict','pending')} — {e.get('summary','')}",f"- **Decision:** {e.get('decision','')}",f"- **Cost:** {e.get('cost','')}",f"- **Results:** {e.get('results','')}",""]
        open(R+'/EXPERIMENTS.md','w').write('\n'.join(out)); print('rendered',R+'/EXPERIMENTS.md',len(d),'experiments')
main()
