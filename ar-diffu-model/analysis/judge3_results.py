import json,glob,collections,random,re
res={}
def boot(v,B=2000):
    random.seed(0); n=len(v); s=sorted(sum(random.choice(v) for _ in range(n))/n for _ in range(B))
    return [100*s[int(.025*B)],100*s[int(.975*B)]]
for pre in ['gsm8k_ar','gsm8k_diffu','reason']:
    lab={}; ids=[]
    for f in sorted(glob.glob(f'judge3/{pre}_0?.json')):
        ids+=[r['id'] for r in json.load(open(f))]
        for r in json.load(open(f.replace('.json','_out.json'))): lab[r['id']]=bool(r['correct'])
    assert set(ids)==set(lab) and len(ids)==len(lab),pre
    g=collections.defaultdict(list)
    for i in ids: g[i.rsplit('|',1)[0]].append(lab[i])
    for k,v in sorted(g.items()): res[k]={'n':len(v),'acc':100*sum(v)/len(v),'ci':boot(v)}
# pooled by model x budget across temps for reasoning
pool=collections.defaultdict(list)
for k in res:
    if 'reasoning' in k and not k.endswith('-t0'):
        m,_,kk,_=k.rsplit('-',3); pool[f"{m}-{kk}"]+= [1]*round(res[k]['acc']*res[k]['n']/100)+[0]*(res[k]['n']-round(res[k]['acc']*res[k]['n']/100))
for k,v in pool.items(): res[k+'-pooled']={'n':len(v),'acc':100*sum(v)/len(v),'ci':boot(v)}
json.dump(res,open('analysis/judge3_results.json','w'),indent=1)
for k,v in sorted(res.items()): print(f"{k:40s} n={v['n']:3d} {v['acc']:5.1f} [{v['ci'][0]:.1f},{v['ci'][1]:.1f}]")
