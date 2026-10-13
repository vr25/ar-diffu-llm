import json, glob, collections
def lab(fs):
    d={}
    for f in fs:
        x=json.load(open(f)); x=x.values() if isinstance(x,dict) else x
        for r in x: d[r['id']]=bool(r['correct'])
    return d
def kappa(a,b):
    n=len(a); po=sum(x==y for x,y in zip(a,b))/n
    pa=sum(a)/n; pb=sum(b)/n; pe=pa*pb+(1-pa)*(1-pb)
    return po,(po-pe)/(1-pe) if pe<1 else 1.0
out={}
for task,cf in [('factual',['llm_judge_merged.json','llm_judge_ar_merged.json']),
                ('reasoning',['llm_judge_reasoning_v2_merged.json','llm_judge_ar_reasoning_v2_merged.json'])]:
    claude=lab(cf); inp=[]; gpt={}
    for f in sorted(glob.glob(f'judge2/{task}_0?.json')):
        inp+= [r['id'] for r in json.load(open(f))]
        for r in json.load(open(f.replace('.json','_out.json'))): gpt[r['id']]=bool(r['correct'])
    miss=[i for i in inp if i not in gpt]; extra=set(gpt)-set(inp)
    assert not miss and not extra, (task,len(miss),len(extra))
    ids=[i for i in inp]; key=lambda i:i.split('|',1)[1]
    A=[claude[key(i)] for i in ids]; B=[gpt[i] for i in ids]
    po,k=kappa(A,B)
    pm=collections.defaultdict(lambda:[0,0,0])
    for i,a,b in zip(ids,A,B):
        m=i.split('|')[1]; pm[m][0]+=1; pm[m][1]+=a; pm[m][2]+=b
    out[task]={'n':len(ids),'agree':po,'kappa':k,
      'per_model':{m:{'n':v[0],'claude':100*v[1]/v[0],'gpt55':100*v[2]/v[0]} for m,v in pm.items()},
      'disagree':[(i,claude[key(i)],gpt[i]) for i,a,b in zip(ids,A,B) if a!=b]}
json.dump(out,open('analysis/judge_agreement.json','w'),indent=1)
for t,v in out.items():
    print(t,v['n'],f"agree={v['agree']:.3f} kappa={v['kappa']:.3f} ndis={len(v['disagree'])}")
    for m,r in sorted(v['per_model'].items(),key=lambda x:-x[1]['claude']): print(f"  {m:12s} {r['claude']:5.1f} {r['gpt55']:5.1f}")
