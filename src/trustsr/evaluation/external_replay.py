"""Authenticated cache-only external replay with explicit partial-method denominators.

The caller must bind model provenance and inputs to the frozen study protocol.
Cache integrity establishes identity and bytes, not proof of original model inference.
"""

import hashlib
import math

import torch

from trustsr.artifacts.predictions import PredictionCache, build_identity, tensor_sha256
from trustsr.contracts import SRPair
from trustsr.evaluation.external_scores import evaluate_score_maps, summarize_subset
from trustsr.jsonio import canonical_json
from trustsr.risk.local import ensemble_variance_score
from trustsr.risk.neighborhood import neighborhood_variance_score
from trustsr.risk.proxies import lr_reprojection_l1_score, three_model_disagreement_score

SEEDS = tuple(range(3407, 3412))
K5_SLOTS = tuple(f'ldsr_{seed}' for seed in SEEDS)
SLOTS = (*K5_SLOTS, 'bicubic', 'sen2sr')
DEPENDENCIES = {
    'lr': ('ldsr_3407',), 'random': ('ldsr_3407',),
    'three_model': ('ldsr_3407', 'bicubic', 'sen2sr'),
    'k5': K5_SLOTS, 'neighborhood': K5_SLOTS,
}


def validate_provenances(provenances: dict) -> None:
    """Reject missing/swapped slots and old cloud policy before any cache access."""
    if set(provenances) != set(SLOTS):
        raise ValueError('exact seven prediction slots required')
    base = {k: v for k, v in provenances[K5_SLOTS[0]].items() if k != 'seed'}
    for slot, seed in zip(K5_SLOTS, SEEDS, strict=True):
        value = provenances[slot]
        if (type(value.get('seed')) is not int or value['seed'] != seed
                or {k: v for k, v in value.items() if k != 'seed'} != base):
            raise ValueError('inconsistent frozen LDSR seed identities')
    sen2sr = provenances['sen2sr']
    if (sen2sr.get('cpu_execution_policy') != 'cloud-sen2srlite-96-v1'
            or type(sen2sr.get('cpu_intraop_threads')) is not int
            or sen2sr['cpu_intraop_threads'] != 96):
        raise ValueError('verified cloud SEN2SRLite CPU policy required')


def replay_roi(pair: SRPair, cache: PredictionCache, provenances: dict) -> dict:
    """Read exactly seven identity-bound cache slots; never infer or hide corruption."""
    pair.validate()
    if not isinstance(pair.sample_id, str) or not pair.sample_id.strip():
        raise ValueError('nonempty ROI identity required')
    validate_provenances(provenances)
    # Build all identities first so malformed provenance cannot trigger partial access.
    identities = {slot: build_identity(provenances[slot], pair.source, pair.sample_id, pair.lr)
                  for slot in SLOTS}
    available, receipts = {}, {}
    for slot, identity in identities.items():
        prediction = cache.get(identity)  # Integrity errors are deliberately fatal.
        receipts[slot] = {'cache_key': identity.key, 'status': 'missing'}
        if prediction is not None:
            available[slot] = prediction
            receipts[slot].update(status='valid', sha256=tensor_sha256(prediction))
    failures = {}
    for method, dependencies in DEPENDENCIES.items():
        missing = [slot for slot in dependencies if slot not in available]
        if missing:
            failures[method] = 'missing:' + ','.join(missing)
    if 'ldsr_3407' not in available:
        result = {'roi': pair.sample_id, 'R1': {}, 'R9': {}, 'transfer': None}
    else:
        center = available['ldsr_3407']
        maps = {'lr': lr_reprojection_l1_score(center, pair.lr)}
        if 'three_model' not in failures:
            maps['three_model'] = three_model_disagreement_score(
                [center, available['bicubic'], available['sen2sr']])
        if 'k5' not in failures:
            samples = torch.stack([available[slot] for slot in K5_SLOTS])
            maps['k5'] = ensemble_variance_score(samples)
            maps['neighborhood'] = neighborhood_variance_score(samples, window=3)
        result = evaluate_score_maps(pair, center, maps)
    result.update(
        predictions=receipts, failures={'R1': dict(failures), 'R9': dict(failures)},
        inputs={'source': pair.source, 'lr_sha256': tensor_sha256(pair.lr),
                'hr_sha256': tensor_sha256(pair.hr)},
    )
    return result


def summarize_replay(members: list[str], rows: list[dict]) -> dict:
    """Require exhaustive ROI/method outcomes and retain each effective denominator."""
    if (not members or any(not isinstance(m, str) or not m.strip() for m in members)
            or len(set(members)) != len(members)):
        raise ValueError('unique nonempty planned members required')
    ids = [row['roi'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(members):
        raise ValueError('exactly one row per planned member required')
    ordered = sorted(rows, key=lambda row: row['roi'])
    methods = {}
    for window in ('R1', 'R9'):
        methods[window] = {}
        for row in ordered:
            valid, failed = row[window], row['failures'][window]
            if set(valid) & set(failed) or set(valid) | set(failed) != set(DEPENDENCIES):
                raise ValueError('each method requires exactly one result or failure')
            if any(not isinstance(reason, str) or not reason.strip() for reason in failed.values()):
                raise ValueError('explicit method failure reason required')
            for diagnostic in valid.values():
                value = diagnostic['aurc']
                if (type(value) not in (int, float) or not math.isfinite(value)
                        or not 0 <= value <= 1):
                    raise ValueError('finite AURC in [0, 1] required')
        for method in DEPENDENCIES:
            values = [row[window][method]['aurc'] for row in ordered if method in row[window]]
            methods[window][method] = {
                'planned': len(members), 'valid': len(values),
                'mean_aurc': math.fsum(values) / len(values) if values else None,
                'failures': {row['roi']: row['failures'][window][method]
                             for row in ordered if method in row['failures'][window]},
            }
    paired = [row for row in ordered if {'lr', 'k5'} <= set(row['R9'])]
    pair_failures = {row['roi']: ';'.join(row['failures']['R9'][method]
                     for method in ('lr', 'k5') if method in row['failures']['R9'])
                     for row in ordered if not {'lr', 'k5'} <= set(row['R9'])}
    transfers = []
    for row in ordered:
        transfer = row['transfer']
        if (transfer is not None) != ('k5' in row['R9']):
            raise ValueError('transfer availability must match K5')
        if transfer is not None:
            for key in ('coverage', 'roi_max_r9'):
                value = transfer[key]
                if (type(value) not in (int, float) or not math.isfinite(value)
                        or not 0 <= value <= 1):
                    raise ValueError('invalid transfer value')
            if (type(transfer['all_rejected']) is not bool
                    or transfer['all_rejected'] != (transfer['coverage'] == 0)
                    or (transfer['all_rejected'] and transfer['roi_max_r9'] != 0)):
                raise ValueError('invalid all-rejected transfer semantics')
            transfers.append(transfer)
    return {
        'primary': summarize_subset(members, paired, pair_failures), 'methods': methods,
        'transfer': {'planned': len(members), 'valid': len(transfers),
                     'all_rejected': sum(t['all_rejected'] for t in transfers)},
    }


def replay_subset(
    pairs: list[SRPair], members: list[str], cache: PredictionCache, provenances: dict,
    *, expected_provenances_sha256: str,
) -> dict:
    """Replay a complete subset and its independent image-level structural diagnostic.

    The expected digest and members must come from the final frozen protocol,
    outside the mutable cache. This function is not an access permit or downloader.
    """
    from trustsr.evaluation.external_opensr import compute_correctness

    if hashlib.sha256(canonical_json(provenances)).hexdigest() != expected_provenances_sha256:
        raise ValueError('protocol model provenance SHA-256 mismatch')
    validate_provenances(provenances)
    ids = [pair.sample_id for pair in pairs]
    if (not members or len(set(members)) != len(members) or len(set(ids)) != len(ids)
            or set(ids) != set(members)):
        raise ValueError('exact planned subset membership required')
    for pair in pairs:
        pair.validate()
    by_id = {pair.sample_id: pair for pair in pairs}
    # Authenticate all caches/compute primary rows before any expensive OpenSR call.
    rows = [replay_roi(by_id[roi], cache, provenances) for roi in sorted(members)]
    summary = summarize_replay(members, rows)
    for row in rows:
        pair = by_id[row['roi']]
        if row['predictions']['ldsr_3407']['status'] == 'missing':
            row['structure'] = {
                'status': 'failed', 'reason': 'missing_center', 'shares': None,
                'grid_pixels': None, 'valid_pixels': None,
            }
            continue
        identity = build_identity(provenances['ldsr_3407'], pair.source, pair.sample_id, pair.lr)
        center = cache.get(identity)
        if (center is None
                or tensor_sha256(center) != row['predictions']['ldsr_3407']['sha256']):
            raise ValueError('center cache changed during replay')
        row['structure'] = compute_correctness(
            pair.lr.detach().cpu().numpy(), center.numpy(), pair.hr.detach().cpu().numpy())
    valid = [row['structure'] for row in rows if row['structure']['status'] == 'valid']
    return {
        'schema': 'trustsr.external-cache-replay.v1',
        'model_provenances_sha256': expected_provenances_sha256,
        'rows': rows, 'summary': summary,
        'structure': {
            'planned': len(members), 'valid': len(valid),
            'mean_shares': {key: math.fsum(row['shares'][key] for row in valid) / len(valid)
                            for key in ('im', 'om', 'ha')} if valid else None,
            'failures': {row['roi']: row['structure']['reason'] for row in rows
                         if row['structure']['status'] != 'valid'},
        },
    }
