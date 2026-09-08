"""Render authenticated published Spain text evidence; no pixels, cache or model access."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import fmean

METHODS = ('lr', 'three_model', 'k5', 'neighborhood', 'random')


def tables(science):
    result = {name: [] for name in ('scores', 'curves', 'paired', 'transfer', 'structure')}
    for subset, data in sorted(science['subsets'].items()):
        rows = sorted(data['rows'], key=lambda r: r['roi'])
        for window in ('R1', 'R9'):
            for method in METHODS:
                metrics = [row[window][method] for row in rows if method in row[window]]
                rhos = [m['rho'] for m in metrics if m['rho'] is not None]
                base = dict(subset=subset, window=window, method=method,
                            planned=len(rows), valid=len(metrics), missing=len(rows)-len(metrics))
                result['scores'].append(dict(
                    base, mean_aurc=fmean(m['aurc'] for m in metrics) if metrics else None,
                    mean_rho=fmean(rhos) if rhos else None, rho_valid=len(rhos)))
                if metrics:
                    grid = metrics[0]['coverages']
                    if any(m['coverages'] != grid or len(m['selective_mean_risks']) != len(grid)
                           for m in metrics):
                        raise ValueError('inconsistent coverage grid')
                    for index, coverage in enumerate(grid):
                        result['curves'].append(dict(
                            base, coverage=coverage,
                            mean_risk=fmean(m['selective_mean_risks'][index] for m in metrics)))
        for row in rows:
            base = dict(subset=subset, roi=row['roi'])
            a, b = row['R9'].get('lr'), row['R9'].get('k5')
            result['paired'].append(dict(base, difference=b['aurc']-a['aurc'] if a and b else None))
            transfer = row['transfer'] or {}
            result['transfer'].append(dict(base, **{
                key: transfer.get(key) for key in ('coverage', 'roi_max_r9', 'all_rejected')}))
            structure = row['structure']
            shares = structure['shares'] or {}
            result['structure'].append(dict(
                base, status=structure['status'], reason=structure.get('reason'),
                grid_pixels=structure.get('grid_pixels'),
                valid_pixels=structure.get('valid_pixels'),
                **{key: shares.get(key) for key in ('im', 'om', 'ha')}))
    return result


def read_verified(path, sha):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('published evidence SHA-256 mismatch')
    return json.loads(raw)


def render(source, sha, protocol_path, protocol_sha, output):
    science = read_verified(source, sha)
    protocol = read_verified(protocol_path, protocol_sha)
    if (science.get('schema') != 'trustsr.spain-study-science.v1'
            or science.get('protocol_sha256') != protocol_sha
            or protocol.get('status') != 'frozen'
            or set(science['subsets']) != set(protocol['science']['subsets'])):
        raise ValueError('frozen science identity mismatch')
    for subset, spec in protocol['science']['subsets'].items():
        ids = [r['roi'] for r in science['subsets'][subset]['rows']]
        if len(ids) != len(set(ids)) or set(ids) != {m['roi'] for m in spec['members']}:
            raise ValueError('published membership mismatch')
    products = tables(science)
    if output.resolve() != output.absolute() or output.is_symlink():
        raise ValueError('output must not traverse symlinks')
    buffers = {}
    for name, rows in products.items():
        stream = io.StringIO(newline='')
        fields = list(rows[0]) if rows else [
            'subset', 'window', 'method', 'planned', 'valid', 'missing', 'coverage', 'mean_risk']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        path = output / f'spain_{name}.csv'
        raw = stream.getvalue().encode()
        if (path.is_symlink()
                or (path.exists() and (not path.is_file() or path.read_bytes() != raw))):
            raise ValueError('refusing conflicting or symlink publication')
        buffers[path] = raw
    from trustsr.evaluation.external_run import write_once

    output.mkdir(parents=True, exist_ok=True)
    for path, raw in buffers.items():
        write_once(path, raw)
    return products


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--science', type=Path, required=True)
    parser.add_argument('--science-sha256', required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.science, args.science_sha256, args.protocol, args.protocol_sha256, args.output)
