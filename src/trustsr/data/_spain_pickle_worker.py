"""Restricted decoder worker; run by spain_package, never on unauthenticated bytes."""

import io
import json
import pickle
import pickletools
import resource
import sys


def main():
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    import numpy as np
    import pandas as pd
    from pandas._libs.internals import _unpickle_block
    from pandas.core.indexes.base import _new_Index
    from pandas.core.internals.managers import BlockManager

    allowed = {
        ('numpy', 'ndarray'): np.ndarray,
        ('numpy', 'dtype'): np.dtype,
        ('pandas.core.frame', 'DataFrame'): pd.DataFrame,
        ('pandas.core.internals.managers', 'BlockManager'): BlockManager,
        ('pandas._libs.internals', '_unpickle_block'): _unpickle_block,
        ('pandas.core.indexes.base', '_new_Index'): _new_Index,
        ('pandas.core.indexes.base', 'Index'): pd.Index,
        ('pandas.core.indexes.range', 'RangeIndex'): pd.RangeIndex,
        ('builtins', 'slice'): slice,
    }
    for prefix in ('numpy.core', 'numpy._core'):
        allowed[(f'{prefix}.multiarray', '_reconstruct')] = np._core.multiarray._reconstruct
        allowed[(f'{prefix}.multiarray', 'scalar')] = np._core.multiarray.scalar
        allowed[(f'{prefix}.numeric', '_frombuffer')] = np._core.numeric._frombuffer

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return allowed[(module, name)]
            except KeyError:
                raise ValueError('unapproved pickle global') from None

        def persistent_load(self, pid):
            raise ValueError('persistent pickle reference forbidden')

    raw = sys.stdin.buffer.read(150_000_001)
    if len(raw) > 150_000_000:
        raise ValueError('package too large')
    stop = None
    for opcode, _, position in pickletools.genops(raw):
        if opcode.name in ('EXT1', 'EXT2', 'EXT4', 'PERSID', 'BINPERSID'):
            raise ValueError('external pickle reference forbidden')
        if opcode.name == 'STOP':
            stop = position
    if stop != len(raw) - 1:
        raise ValueError('trailing pickle data')
    payload = RestrictedUnpickler(io.BytesIO(raw)).load()
    count = int(sys.argv[1])
    if type(payload) is not dict or not {'L2A', 'HRharm', 'metadata'} <= payload.keys():
        raise ValueError('invalid package schema')
    arrays = {}
    for key, shape in (('L2A', (count, 12, 128, 128)), ('HRharm', (count, 4, 512, 512))):
        value = payload[key]
        if (type(value) is not np.ndarray or value.shape != shape
                or value.dtype.kind not in 'uif' or value.dtype.itemsize > 8):
            raise ValueError('invalid package array')
        arrays[key] = value
    frame = payload['metadata']
    fields = ['roi', 'lr_file', 'hr_file', 'lr_gee_id', 'crs', 'affine']
    if (type(frame) is not pd.DataFrame or len(frame) != count
            or not frame.columns.is_unique or not set(fields) <= set(frame.columns)):
        raise ValueError('invalid embedded metadata')
    rows = frame.loc[:, fields].to_dict(orient='records')
    encoded = json.dumps(rows, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(encoded) > 1024 * 1024:
        raise ValueError('metadata too large')
    # A seekable buffer is required for NPZ; no filesystem output from this process.
    output = io.BytesIO()
    np.savez(output, **arrays, metadata=np.frombuffer(encoded, dtype=np.uint8))
    sys.stdout.buffer.write(output.getvalue())


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Never export untrusted objects, quality values, or payload exception strings.
        sys.stderr.write('restricted package decoding failed\n')
        sys.exit(1)
