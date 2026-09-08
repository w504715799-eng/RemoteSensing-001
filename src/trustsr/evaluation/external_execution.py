"""Missing-only external predictions with durable attempts and no automatic retry."""

import hashlib
import time

from trustsr.artifacts.predictions import CacheIntegrityError, build_identity, tensor_sha256
from trustsr.contracts import SRPair
from trustsr.evaluation.external_replay import SLOTS, validate_provenances
from trustsr.jsonio import canonical_json

SUCCESS = {'stored', 'cached', 'recovered'}


def populate_predictions(
    subsets, cache, provenances, factory, state, persist, *, allow_ldsr: bool,
):
    """Populate missing predictions under a caller-owned protocol-bound run lock.

    factory(slot) must produce the complete expected provenance. persist() must
    durably store state; its exceptions stop the call before further inference.
    No HR is passed to a model. State and cache belong to this external run only.
    """
    validate_provenances(provenances)
    inputs, entries = [], {}
    for subset in sorted(subsets):
        pairs = subsets[subset]
        ids = [pair.sample_id for pair in pairs]
        if not pairs or len(set(ids)) != len(ids):
            raise ValueError('unique nonempty subset members required')
        for pair in sorted(pairs, key=lambda p: p.sample_id):
            pair.validate()
            inputs.append({'subset': subset, 'roi': pair.sample_id, 'source': pair.source,
                           'lr_sha256': tensor_sha256(pair.lr),
                           'hr_sha256': tensor_sha256(pair.hr)})
            for slot in SLOTS:
                identity = build_identity(provenances[slot], pair.source, pair.sample_id, pair.lr)
                if identity.key in entries:
                    raise ValueError('duplicate prediction identity across planned slots')
                entries[identity.key] = (slot, pair, identity)
    if not inputs:
        raise ValueError('nonempty planned inputs required')
    binding = hashlib.sha256(canonical_json({'inputs': inputs, 'models': provenances})).hexdigest()
    if state and (set(state) != {'binding_sha256', 'predictions'}
                  or state['binding_sha256'] != binding):
        raise ValueError('resumed execution input/model binding differs')
    records = state.get('predictions', {})
    if not set(records) <= set(entries):
        raise ValueError('unplanned prediction record')
    available = {}
    for key, (_, _, identity) in entries.items():
        prediction = cache.get(identity)  # Whole-cache integrity gate precedes any model.
        available[key] = None if prediction is None else tensor_sha256(prediction)
        old = records.get(key)
        if old is not None:
            if old.get('status') not in SUCCESS | {'started', 'failed'}:
                raise ValueError('invalid prediction attempt state')
            if old['status'] in SUCCESS and (
                prediction is None or available[key] != old.get('sha256')
            ):
                raise CacheIntegrityError('successful prediction changed or disappeared')
            if old['status'] == 'failed' and prediction is not None:
                raise CacheIntegrityError('failed attempt unexpectedly gained a prediction')
    if not allow_ldsr and any(
        available[key] is None and key not in records and slot.startswith('ldsr_')
        for key, (slot, _, _) in entries.items()
    ):
        raise RuntimeError('missing LDSR predictions require explicit GPU scope and CUDA')
    if not state:
        state.update(binding_sha256=binding, predictions={})
        persist()
    records = state['predictions']
    for key, (slot, pair, identity) in entries.items():
        prediction, old = available[key], records.get(key)
        if prediction is not None:
            if old is None or old['status'] == 'started':
                records[key] = {'status': 'cached' if old is None else 'recovered',
                                'sha256': prediction}
                persist()
            continue
        if old is not None:
            if old['status'] == 'started':
                records[key] = {'status': 'failed', 'reason': 'interrupted_prediction'}
                persist()
            continue
        # Factory/asset/provenance failures are fatal, never silently missing data.
        load_started = time.perf_counter()
        model = factory(slot)
        factory_seconds = time.perf_counter() - load_started
        if canonical_json(model.provenance()) != canonical_json(provenances[slot]):
            raise ValueError('model provenance differs from frozen protocol')
        records[key] = {'status': 'started', 'factory_seconds': factory_seconds}
        persist()
        started = time.perf_counter()
        model_input = pair.lr.clone()
        reason = None
        try:
            prediction = model.predict(model_input)
        except (RuntimeError, MemoryError):
            reason = 'prediction_runtime_failed'
        except ValueError:
            reason = 'prediction_contract_failed'
        if reason is None:
            try:
                if tensor_sha256(model_input) != identity.lr_sha256:
                    raise ValueError('model mutated its LR input')
                SRPair(pair.sample_id, pair.source, pair.lr, prediction, pair.scale).validate()
            except (AttributeError, TypeError, ValueError):
                reason = 'prediction_contract_failed'
        elapsed = time.perf_counter() - started
        if reason is not None:
            records[key] = {'status': 'failed', 'reason': reason, 'seconds': elapsed,
                            'factory_seconds': factory_seconds}
        else:
            if cache.get(identity) is not None:
                raise CacheIntegrityError('prediction appeared during a locked execution')
            cache.put(identity, prediction)
            records[key] = {'status': 'stored', 'sha256': tensor_sha256(prediction),
                            'seconds': elapsed, 'factory_seconds': factory_seconds}
        persist()
    return state
