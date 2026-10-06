"""Publish the tested, checksummed snapshot from GitHub Actions."""
import hashlib,json,os
from pathlib import Path
import urllib.request,urllib.error,urllib.parse

ROOT=Path(__file__).resolve().parents[1]
REPO='cabal312512/WaveForge-X'
TAG='v1.1.0'
token=os.environ['GH_TOKEN']


def request(url,method='GET',payload=None,binary=False):
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'WaveForge-X-release','X-GitHub-Api-Version':'2022-11-28'}
    if payload is not None:
        headers['Content-Type']='application/octet-stream' if binary else 'application/json'
        if not binary:payload=json.dumps(payload).encode()
    req=urllib.request.Request(url,data=payload,headers=headers,method=method)
    with urllib.request.urlopen(req,timeout=180) as response:return json.load(response)


api=f'https://api.github.com/repos/{REPO}'
try:
    release=request(api+'/releases/tags/'+TAG)
    if not release['draft']:
        print('Published release already exists; leaving it unchanged.');raise SystemExit(0)
except urllib.error.HTTPError as e:
    if e.code!=404:raise
    release=request(api+'/releases','POST',dict(tag_name=TAG,target_commitish=os.environ['GITHUB_SHA'],name='WaveForge-X 1.1.0 — integrated experimental research',body=(ROOT/'RELEASE.md').read_text(encoding='utf-8'),draft=True,prerelease=False))

upload=release['upload_url'].split('{')[0]
for name in ['certiphy-evidence.zip','method-development.zip','reuse-evidence.zip','SHA256SUMS']:
    data=(ROOT/'data'/name).read_bytes()
    existing=next((a for a in release['assets'] if a['name']==name),None)
    if existing:
        if existing.get('digest')!='sha256:'+hashlib.sha256(data).hexdigest():raise RuntimeError('existing asset differs; refusing overwrite')
        continue
    asset=request(upload+'?'+urllib.parse.urlencode({'name':name}),'POST',data,binary=True)
    if asset['size']!=len(data):raise RuntimeError('asset size mismatch')
    if asset.get('digest') and asset['digest']!='sha256:'+hashlib.sha256(data).hexdigest():raise RuntimeError('asset digest mismatch')
    print(name,asset['size'],'bytes uploaded')
published=request(api+'/releases/'+str(release['id']),'PATCH',dict(draft=False,make_latest='true'))
print(published['html_url'])
