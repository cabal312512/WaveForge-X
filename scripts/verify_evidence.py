"""Verify immutable synthetic evidence; optionally extract the recorded main study."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def verify(extract=False):
    for line in (ROOT/'data/SHA256SUMS').read_text(encoding='utf-8').splitlines():
        expected,name=line.split()
        archive=ROOT/'data'/name
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'archive checksum mismatch: {name}')
        with zipfile.ZipFile(archive) as z:
            manifest=json.loads(z.read('MANIFEST.json'))
            for member,digest in manifest.items():
                data=z.read(member)
                if hashlib.sha256(data).hexdigest()!=digest:raise ValueError(f'member checksum: {member}')
                path=PurePosixPath(member)
                if path.is_absolute() or '..' in path.parts or '\\' in member:raise ValueError('unsafe member path')
                if extract and name=='certiphy-evidence.zip' and member.startswith('results/certiphy/'):
                    target=(ROOT/member).resolve()
                    if not target.is_relative_to(ROOT/'results/certiphy'):raise ValueError('path escaped evidence directory')
                    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
                        raise ValueError(f'refusing to overwrite changed file: {member}')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():target.write_bytes(data)
        print(f'{name}: verified {len(manifest)} members')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--extract',action='store_true');a=p.parse_args();verify(a.extract)
