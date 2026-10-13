"""Expand an existing geometry/field profile across the seven AXIS variants."""
import argparse
from copy import deepcopy
from artifacts import profile, save
from methods import METHODS

def expand(settings, name):
    result = deepcopy(settings)
    result['id'] = name
    result['cases'] = []
    for case in settings['cases']:
        if 'method' in case:
            raise ValueError('Input profile is already method-expanded')
        for method in METHODS:
            if case['norm'] == 'fracarea' and not method.startswith('conservative_'):
                continue
            item = deepcopy(case)
            item.update(id=case['id']+'-'+method, method=method)
            result['cases'].append(item)
    return result

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--id', required=True)
    args = ap.parse_args()
    save(args.output, expand(profile(args.profile), args.id))
    print(f"Wrote {len(profile(args.output)['cases'])} cases")
