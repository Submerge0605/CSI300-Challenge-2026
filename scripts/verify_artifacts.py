"""Verify frozen submission files using only the Python standard library."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / 'release_manifest.json').read_text(encoding='utf-8'))
    checked = []
    for entry in manifest['files']:
        target = (root / entry['path']).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Manifest path escapes repository')
        payload = target.read_bytes()
        if len(payload) != entry['bytes']:
            raise ValueError(f"Size mismatch: {entry['path']}")
        if hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise ValueError(f"SHA256 mismatch: {entry['path']}")
        checked.append(entry['path'])
    meta = json.loads((root / 'submission/model/meta.json').read_text(encoding='utf-8'))
    if (meta['kind'], meta['variant']) != ('stfree_extra_trees', 'confidence'):
        raise ValueError('Unexpected model policy')
    if len(meta['models']) != 6 or meta['max_lookback'] != 120:
        raise ValueError('Unexpected model count or lookback')
    for family in ['extra', 'utility']:
        items = [s for s in meta['models'] if s['family'] == family]
        if sorted(s['seed'] for s in items) != [42, 43, 44]:
            raise ValueError('Unexpected seed coverage')
        for spec in items:
            if spec['rounds'] != 350 or len(spec['features']) != 229:
                raise ValueError('Unexpected rounds or feature count')
            if any('is_st' in c for c in spec['features']):
                raise ValueError('ST-dependent feature found')
    return {'status': 'pass', 'files_checked': len(checked), 'boosters': 6,
            'features_per_booster': 229, 'rounds_per_booster': 350}

if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
