"""Protocol-bound external run storage, resumable predictions and verified publication."""

import copy
import fcntl
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from trustsr.artifacts.predictions import PredictionCache
from trustsr.evaluation.external_execution import populate_predictions
from trustsr.evaluation.external_replay import replay_subset
from trustsr.jsonio import atomic_write_bytes, canonical_json


def sync_directory(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_regular(path: Path, limit: int = 8_388_608) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('bounded regular non-symlink file required')
    with path.open('rb') as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('file exceeds size limit')
    return raw


def write_once(path: Path, raw: bytes):
    """Atomic no-replace publication; identical interrupted publication is recoverable."""
    if path.exists() or path.is_symlink():
        if read_regular(path) != raw:
            raise ValueError('refusing to overwrite a different published result')
        return
    fd, name = tempfile.mkstemp(prefix='.publish-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_regular(path) != raw:
                raise ValueError('conflicting publication appeared') from None
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _reconstruct(subsets, protocol, cache, execution, input_receipts):
    provenances = protocol['science']['provenances']
    model_digest = hashlib.sha256(canonical_json(provenances)).hexdigest()
    results, failures = {}, {}
    for name in sorted(subsets):
        members = [m['roi'] for m in protocol['science']['subsets'][name]['members']]
        results[name] = replay_subset(subsets[name], members, cache, provenances,
                                      expected_provenances_sha256=model_digest)
        failures[name] = {}
        for row in results[name]['rows']:
            per_roi = {}
            for slot, receipt in row['predictions'].items():
                attempt = execution['predictions'][receipt['cache_key']]
                if attempt['status'] == 'failed':
                    per_roi[slot] = attempt['reason']
            if per_roi:
                failures[name][row['roi']] = per_roi
    return {'schema': 'trustsr.spain-study-science.v1',
            'protocol_sha256': hashlib.sha256(canonical_json(protocol)).hexdigest(),
            'subsets': results, 'prediction_failures': failures, 'input_receipts': input_receipts}


def run_study(
    subsets, protocol, output: Path, factory, *, allow_ldsr: bool, replay_only=False,
    input_receipts=None, input_seconds=0.0,
):
    """Execute already-authorized, protocol-bound inputs; CLI owns pre-access validation.

    No network access; all persistent files belong to an independent external run.
    The caller must validate package/model/repository path separation before entry.
    """
    if set(subsets) != set(protocol['science']['subsets']):
        raise ValueError('exact frozen subsets required')
    for name, pairs in subsets.items():
        expected = [m['roi'] for m in protocol['science']['subsets'][name]['members']]
        actual = [p.sample_id for p in pairs]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError('exact frozen member set required')
    protocol_sha = hashlib.sha256(canonical_json(protocol)).hexdigest()
    marker = canonical_json({'schema': 'trustsr.spain-run.v1', 'protocol_sha256': protocol_sha})
    if output.is_symlink() or output.resolve() != output.absolute():
        raise ValueError('regular output path without symlink ancestors required')
    if not output.exists():
        if replay_only:
            raise ValueError('replay requires an existing published run')
        output.mkdir(exist_ok=False)
        write_once(output / 'run.json', marker)
        sync_directory(output.parent)
    elif read_regular(output / 'run.json') != marker:
        raise ValueError('run directory belongs to another protocol')
    lock_path = output / '.lock'
    flags = os.O_RDWR | os.O_NOFOLLOW
    if not replay_only:
        flags |= os.O_CREAT
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _locked_run(subsets, protocol, output, factory,
                           allow_ldsr=allow_ldsr, replay_only=replay_only,
                           input_receipts=input_receipts or {}, input_seconds=input_seconds)
    finally:
        os.close(descriptor)


def _locked_run(
    subsets, protocol, output, factory, *, allow_ldsr, replay_only, input_receipts, input_seconds,
):
    state_path, manifest_path = output / 'state.json', output / 'manifest.json'
    published = manifest_path.exists() or manifest_path.is_symlink()
    if replay_only and not published:
        raise ValueError('replay requires a complete publication manifest')
    cache_root = output / 'cache'
    if cache_root.is_symlink() or (cache_root.exists() and not cache_root.is_dir()):
        raise ValueError('regular dedicated cache directory required')
    if published and (not cache_root.is_dir() or not state_path.is_file()):
        raise ValueError('published cache/state is missing')
    cache = PredictionCache(cache_root)
    state = json.loads(read_regular(state_path)) if state_path.exists() else {}
    state_before = copy.deepcopy(state)
    if published:
        manifest = json.loads(read_regular(manifest_path))
        expected_protocol = hashlib.sha256(canonical_json(protocol)).hexdigest()
        if (manifest.get('schema') != 'trustsr.spain-publication.v1'
                or manifest.get('protocol_sha256') != expected_protocol):
            raise ValueError('publication manifest protocol mismatch')
        if manifest['execution_sha256'] != hashlib.sha256(canonical_json(state)).hexdigest():
            raise ValueError('published execution state changed')
        for file, key in (('science.json', 'science_sha256'),
                          ('verification.json', 'verification_sha256')):
            if hashlib.sha256(read_regular(output / file)).hexdigest() != manifest[key]:
                raise ValueError('published result integrity mismatch')

        def no_factory(slot):
            raise ValueError('published execution cannot perform model inference')

        def no_persist():
            raise ValueError('published execution state cannot be changed')

        populate_predictions(subsets, cache, protocol['science']['provenances'], no_factory,
                             state, no_persist, allow_ldsr=False)
        if state != state_before:
            raise ValueError('published state changed')
        result = json.loads(read_regular(output / 'science.json'))
        if (result.get('protocol_sha256') != expected_protocol
                or result.get('input_receipts') != input_receipts):
            raise ValueError('published protocol/input receipts mismatch')
        if not replay_only:
            return result
        reconstructed = _reconstruct(subsets, protocol, cache, state, input_receipts)
        if canonical_json(reconstructed) != canonical_json(result):
            raise ValueError('read-only scientific replay differs from publication')
        return reconstructed

    def persist():
        if state_path.is_symlink():
            raise ValueError('state file must not be a symlink')
        atomic_write_bytes(state_path, canonical_json(state))
        sync_directory(output)

    prediction_start = time.perf_counter()
    populate_predictions(subsets, cache, protocol['science']['provenances'], factory,
                         state, persist, allow_ldsr=allow_ldsr)
    prediction_seconds = time.perf_counter() - prediction_start
    first_start = time.perf_counter()
    first = _reconstruct(subsets, protocol, cache, state, input_receipts)
    first_seconds = time.perf_counter() - first_start
    second_start = time.perf_counter()
    second = _reconstruct(subsets, protocol, cache, state, input_receipts)
    second_seconds = time.perf_counter() - second_start
    science = canonical_json(first)
    if science != canonical_json(second):
        raise ValueError('two cache-only scientific reconstructions differ')
    science_sha = hashlib.sha256(science).hexdigest()
    verification = canonical_json({
        'schema': 'trustsr.spain-verification.v1', 'protocol_sha256': first['protocol_sha256'],
        'science_sha256': science_sha, 'cache_science_replay_equal': True,
        'prediction_inference_verified': False, 'prediction_slots': len(state['predictions']),
    })
    runtime = {
        'scope': 'current successful invocation; prior attempts remain in state.json',
        'input_seconds': input_seconds, 'prediction_phase_seconds': prediction_seconds,
        'reconstruction_seconds': [first_seconds, second_seconds],
    }
    runtime_path = output / 'runtime.json'
    if runtime_path.is_symlink():
        raise ValueError('runtime must not be a symlink')
    atomic_write_bytes(runtime_path, canonical_json(runtime))
    write_once(output / 'science.json', science)
    write_once(output / 'verification.json', verification)
    write_once(manifest_path, canonical_json({
        'schema': 'trustsr.spain-publication.v1', 'protocol_sha256': first['protocol_sha256'],
        'science_sha256': science_sha,
        'verification_sha256': hashlib.sha256(verification).hexdigest(),
        'execution_sha256': hashlib.sha256(canonical_json(state)).hexdigest(),
    }))
    # Match the JSON-normalized public return on a completed-run resume.
    return json.loads(science)
