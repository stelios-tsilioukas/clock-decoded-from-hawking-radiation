import sys, json
from windows import find
k1=float(sys.argv[1]); De=float(sys.argv[2]); ee=float(sys.argv[3]); h=float(sys.argv[4])
out=find(k1,De=De,ee=ee,h=h)
print(json.dumps(out)); json.dump(out,open(f'win_k{int(k1)}_De{De}_ee{ee}.json','w'))
