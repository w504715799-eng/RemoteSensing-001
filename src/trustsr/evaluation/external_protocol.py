"""Derive immutable external-science identity from published local text evidence."""

import hashlib
import json
import math
import re
from pathlib import Path

from trustsr.jsonio import canonical_json

EVIDENCE = {
    'metadata': ('artifacts/datasets/spain-metadata-v1.json',
                 '7ecb39fbd9ce48740a60bdc253aff4704773acfeefed26672aa8043d3530c325'),
    'timing': ('paper/tables/inference-timing-v1.json',
               'aad0d0652e1858590e4768ed1d8275056347cb58a0f784e4263a62d8ce5a5e3a'),
    'cloud': ('paper/tables/sen2sr-cloud-policy-check-v1.json',
              'ed7fed19297b21debcbcad81938def564a212ff53417e19c3b9e980bb8038102'),
    'cpu_execution': ('paper/tables/cloud-cpu-execution-v1.json',
                      '4467cfeb57d291d96fc57bd4c35378829727d7fe6cacf9f73650e2fd85a0c6dc'),
}
PACKAGES = {
    'spain_crops': (140385908, '5f68a962b200ef781cbabbe21e598b71379fafb3c12d614deae652e65bb86319'),
    'spain_urban': (100276003, 'da6d0ae68a78ef9c125d3b3805fde40718e61b1e54db2877d8041b049c2c3775'),
}
KNOWN_VERSIONS = {'torch': '2.12.1+cu130', 'opensr-test': '1.3.3',
                  'opensr-model': '1.1.1', 'sen2sr': '0.8.5', 'mlstac': '0.4.9'}
RUNTIME_PACKAGES = (*KNOWN_VERSIONS, 'numpy', 'pandas', 'scipy', 'safetensors', 'satalign')
TIMING_PLAN = ('paper/protocols/spain-budget-draft-v1.json',
               'a77e313b6932c6cbc69778b4efaca38b9786b2d654423d1ff6911683d96f598c')


def build_draft(repository: Path) -> dict:
    """Read only pinned Git-safe evidence; no external/model/cache IO."""
    evidence = {}
    for name, (relative, digest) in EVIDENCE.items():
        raw = (repository / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('published evidence SHA-256 mismatch')
        evidence[name] = json.loads(raw)
    models = evidence['timing']['models']
    provenances = {f'ldsr_{seed}': dict(models['ldsr']['provenance'], seed=seed)
                  for seed in range(3407, 3412)}
    provenances.update(bicubic=models['bicubic']['provenance'],
                       sen2sr=evidence['cloud']['provenance'])
    subsets = {}
    for name, (size, digest) in PACKAGES.items():
        subsets[name] = {'package_filename': f'{name}.pkl', 'package_sha256': digest,
                         'package_size': size, 'source_path': f'100/{name}/{name}.pkl',
                         'members': evidence['metadata']['subsets'][name]['members']}
    implementation = {}
    for directory in ('src/trustsr', 'scripts/paper'):
        for path in sorted((repository / directory).rglob('*.py')):
            implementation[str(path.relative_to(repository))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    return {
        'schema': 'trustsr.spain-execution-protocol.v1', 'status': 'draft',
        'science': {
            'source_revision': evidence['metadata']['revision'], 'subsets': subsets,
            'provenances': provenances, 'normalization': 'rgbn_clip_10000_divide_10000_full512_v1',
            'scores': ['lr', 'three_model', 'k5', 'neighborhood', 'random'],
            'neighborhood_window': 3, 'risk_windows': [1, 9],
            'transfer_threshold': 7.970395366024563e-06,
            'statistics': 'descriptive_equal_roi_no_bootstrap',
            'structure': 'isolated_opensr133_softmin_checked_registration_v1',
            'failure_policy': 'exhaustive_dependencies_no_automatic_prediction_retry_v1',
            'display': 'first_two_sorted_members_per_subset_no_replacement',
            'limitations': ['source_nodata_unknown_zero_preserved_numeric_contract_applied',
                           'official_quality_metadata_incidentally_seen_not_used_for_selection',
                           'spatial_independence_and_pretraining_overlap_not_established',
                           'embedded_membership_and_pickle_compatibility_checked_after_authorization'],
        },
        'implementation': implementation,
        'runtime_versions': evidence['cpu_execution']['profile']['versions'],
        'budget': None,
    }


def _user_managed_budget(repository: Path) -> dict:
    """Use measured planning times; price and supplier billing belong to the user."""
    relative, digest = TIMING_PLAN
    raw = (repository / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('execution timing evidence SHA-256 mismatch')
    plan = json.loads(raw)
    return {
        'cost_management': 'user',
        'maximum_wall_seconds': plan['proposed_run_wall_limit_seconds'],
        'estimated_wall_seconds': plan['proposed_estimated_wall_seconds'],
        'evidence_sha256': digest,
    }


def build_frozen(repository: Path) -> dict:
    """Freeze the implemented contract with user-managed costs, without external IO."""
    protocol = build_draft(repository)
    protocol['status'] = 'frozen'
    protocol['budget'] = _user_managed_budget(repository)
    return protocol


def load_frozen(raw: bytes, expected_sha256: str, repository: Path) -> dict:
    """Authenticate the separately reviewed protocol, then reject draft/unbound execution."""
    if len(raw) > 1_048_576 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('protocol SHA-256 mismatch')
    protocol = json.loads(raw)
    if canonical_json(protocol) != raw:
        raise ValueError('canonical protocol bytes required for a single digest identity')
    if protocol.get('status') != 'frozen':
        raise ValueError('a final frozen protocol is required before external access')
    draft = build_draft(repository)
    if set(protocol) != set(draft) or protocol['schema'] != draft['schema']:
        raise ValueError('unreviewed protocol schema')
    for field in ('science', 'implementation'):
        if canonical_json(protocol[field]) != canonical_json(draft[field]):
            raise ValueError('science or implementation differs from reviewed execution contract')
    versions = protocol['runtime_versions']
    if (not isinstance(versions, dict) or set(versions) != set(RUNTIME_PACKAGES)
            or any(not isinstance(v, str) or not v.strip() for v in versions.values())
            or any(versions[k] != v for k, v in KNOWN_VERSIONS.items())):
        raise ValueError('complete frozen runtime versions required')
    if canonical_json(versions) != canonical_json(draft['runtime_versions']):
        raise ValueError('runtime differs from the measured cloud environment')
    budget = protocol['budget']
    if isinstance(budget, dict) and 'cost_management' in budget:
        if canonical_json(budget) != canonical_json(_user_managed_budget(repository)):
            raise ValueError('execution limits differ from the user-managed timing contract')
        return protocol
    # Retain support for previously reviewed protocols with explicit monetary budgets.
    fields = {'maximum_wall_seconds', 'estimated_wall_seconds', 'hourly_price', 'currency',
              'evidence_sha256'}
    if not isinstance(budget, dict) or set(budget) != fields:
        raise ValueError('complete reviewed budget required')
    for key in ('maximum_wall_seconds', 'estimated_wall_seconds', 'hourly_price'):
        if (type(budget[key]) not in (int, float) or not math.isfinite(budget[key])
                or budget[key] <= 0):
            raise ValueError('finite positive budget required')
    if budget['estimated_wall_seconds'] > budget['maximum_wall_seconds']:
        raise ValueError('estimated execution exceeds approved wall budget')
    if (not isinstance(budget['currency'], str)
            or re.fullmatch('[A-Z]{3}', budget['currency']) is None
            or not isinstance(budget['evidence_sha256'], str)
            or re.fullmatch('[0-9a-f]{64}', budget['evidence_sha256']) is None):
        raise ValueError('budget currency and evidence digest required')
    return protocol
