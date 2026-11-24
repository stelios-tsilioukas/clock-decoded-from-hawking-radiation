import json, glob, numpy as np
LH=-0.34657359027997264; IN=0.31191623125197537; LM=LH-1
def load(files):
    rows=[]
    for f in files:
        for l in open(f):
            if l.startswith('{'): rows.append(json.loads(l))
    return rows
def tab(rows, keys):
    print('| label | classical | '+' | '.join(keys)+' |')
    print('|'+'---|'*(2+len(keys)))
    for r in sorted(rows,key=lambda r:r['label']):
        print(f"| {r['label']:.4f} | {r['cl']:.6f} | "+' | '.join(f"{r[k][0]:.6f}" for k in keys if k in r)+' |')
    print()
    print('| label | classical 1-|V| | '+' | '.join(keys)+' |')
    print('|'+'---|'*(2+len(keys)))
    for r in sorted(rows,key=lambda r:r['label']):
        print(f"| {r['label']:.4f} | {r['cl_1mV']:.3e} | "+' | '.join(f"{r[k][1]:.3e}" for k in keys if k in r)+' |')
    print()
def bylabel(rows):
    d={}
    for r in rows: d[round(r['label'],6)]=r
    return d
if __name__=='__main__':
    print('## A5 main k1=20 De=1 ee=0.0125')
    R=load(['A5_k20_De1_ee0125.log']); tab(R,['os0','ose','y0','hs','dece','hd'])
    for f in ['A5_k20_De2_ee0125.log','A5_k20_De1_ee025.log']:
        print('##',f); tab(load([f]),['os0','ose','hs','dece'])
    print('## A7')
    print('| k1 | interval | classical | one-sided(e) | decoded(e) | Y0 | quantum | decoder | decoder(Y0) |')
    print('|---|---|---|---|---|---|---|---|---|')
    out={}
    for k1,files in [(10,['A7_k10.log','A8_k10.log']),(20,['A5_k20_De1_ee0125.log','A7_k20.log']),(40,['A7_k40.log','A8_k40.log']),(80,['A7_k80.log'])]:
        d=bylabel(load(files))
        for name,(a,b) in [('H->0.9inner',(LH,IN)),('(H-1)->H',(LM,LH))]:
            ra,rb=d[round(a,6)],d[round(b,6)]
            cl=rb['cl']-ra['cl']; os_=rb['ose'][0]-ra['ose'][0]; de=rb['dece'][0]-ra['dece'][0]; y0=rb['y0'][0]-ra['y0'][0]
            q=k1*(os_-cl)/abs(cl); dc=k1*(de-os_)/abs(cl); dy=k1*(y0-rb['os0'][0]+ra['os0'][0])/abs(cl)
            print(f"| {k1} | {name} | {cl:.6f} | {os_:.6f} | {de:.6f} | {y0:.6f} | {q:.4f} | {dc:.4f} | {dy:.4f} |")
    print()
    print('## A8 decoder correction delta = dece - ose (reading units), parts')
    for k1,files in [(10,['A7_k10.log','A8_k10.log']),(20,['A5_k20_De1_ee0125.log','A7_k20.log','A8_k20.log']),(40,['A7_k40.log','A8_k40.log']),(80,['A7_k80.log'])]:
        rows=[r for r in load(files) if 'dece' in r and 'hs' in r]
        rows=sorted(bylabel(rows).values(),key=lambda r:r['label'])
        L=np.array([r['label'] for r in rows])
        dl=np.array([r['dece'][0]-r['ose'][0] for r in rows])
        dD=np.array([r['hs'][0]-r['ose'][0] for r in rows])
        ds=np.array([r['dece'][0]-r['hs'][0] for r in rows])
        q=np.array([r['ose'][0]-r['cl'] for r in rows])
        print(f'### k1={k1}')
        print('| label | k1*delta | k1*D-part (hs-ose) | k1*sigma-part (dece-hs) | k1*(ose-cl) | k1*(ose-os0) |')
        print('|---|---|---|---|---|---|')
        for r,a,b,c,e in zip(rows,dl,dD,ds,q):
            print(f"| {r['label']:.4f} | {k1*a:.5f} | {k1*b:.5f} | {k1*c:.5f} | {k1*e:.5f} | {k1*(r['ose'][0]-r['os0'][0]):.2e} |")
        for nm,v in [('delta',dl),('D-part',dD),('sigma-part',ds)]:
            for deg in (3,5,7):
                X=np.stack([L**p for p in range(1,deg+1,2)],1)
                c,res,_,_=np.linalg.lstsq(X,k1*v,rcond=None)
                rms=np.sqrt(np.mean((X@c-k1*v)**2))
                print(f"fit k1*{nm}, odd deg {deg}: coeffs(L,L^3,..)={np.round(c,6).tolist()} rms={rms:.1e}")
        print()
