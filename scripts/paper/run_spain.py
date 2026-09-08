"""Frozen Spain execution; draft/freeze/preflight never read external pixels."""

import argparse
import hashlib
import json
import signal
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch

from trustsr.data.spain_package import decode_package
from trustsr.evaluation.external_protocol import build_draft, build_frozen, load_frozen
from trustsr.evaluation.external_run import read_regular, run_study, write_once
from trustsr.jsonio import canonical_json

REPOSITORY = Path(__file__).resolve().parents[2]


def verify_import_origin():
    import trustsr

    expected = (REPOSITORY / 'src/trustsr').resolve()
    if (Path(trustsr.__file__).resolve() != expected / '__init__.py'
            or [Path(path).resolve() for path in trustsr.__path__] != [expected]):
        raise ValueError('executing trustsr package is not the protocol-bound repository source')
    for name, module in list(sys.modules.items()):
        if name.startswith('trustsr.') and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            if expected not in path.parents:
                raise ValueError('loaded trustsr module is outside the bound implementation tree')


def validate_paths(output: Path, packages: Path, model_roots: list[Path], repository: Path):
    resolved = output.resolve()
    if resolved != output.absolute() or output.is_symlink():
        raise ValueError('output must not traverse symlinks')
    if set(output.parts) & {'phase2b3a', 'phase2b3b', 'phase2b3c', 'phase2b1b', 'subset-v1'}:
        raise ValueError('output cannot be inside a historical experiment tree')
    for protected in (repository, packages, *model_roots):
        root = protected.resolve()
        if resolved == root or root in resolved.parents or resolved in root.parents:
            raise ValueError('run output must be separate from repository, packages and models')
    if packages.resolve() != packages.absolute() or not packages.is_dir():
        raise ValueError('regular package directory without symlink ancestors required')


def verify_runtime(protocol):
    for name, expected in protocol['runtime_versions'].items():
        if version(name) != expected:
            raise ValueError(f'runtime package differs from frozen protocol: {name}')
    from trustsr.models.bicubic import BicubicX4
    from trustsr.models.cloud_sen2srlite import CloudSEN2SRLiteX4

    expected = protocol['science']['provenances']
    # These provenance-only objects do not load any assets or model weights.
    for slot, model in (('bicubic', BicubicX4()), ('sen2sr', CloudSEN2SRLiteX4(None))):
        if canonical_json(model.provenance()) != canonical_json(expected[slot]):
            raise ValueError('CPU/backend profile differs from verified cloud configuration')
    cpu_info = Path('/proc/cpuinfo').read_text()
    cpu_models = {line.split(':', 1)[1].strip() for line in cpu_info.splitlines()
                  if line.startswith('model name')}
    if cpu_models != {'AMD EPYC 7K62 48-Core Processor'}:
        raise ValueError('CPU hardware differs; development verification is required before freeze')


class ModelFactory:
    """Load verified existing assets only when a missing slot actually needs them."""

    def __init__(self, ldsr_directory, sen2sr_directory):
        self.ldsr_directory = ldsr_directory
        self.sen2sr_directory = sen2sr_directory
        self.loaded = {}

    @staticmethod
    def _require_assets(directory, names):
        if directory is None:
            raise ValueError('existing model directory required for missing predictions')
        directory = Path(directory)
        for name in names:
            path = directory / name
            if (path.resolve() != path.absolute() or path.is_symlink() or not path.is_file()):
                raise ValueError('existing regular verified model assets required; no download')
        return directory

    def __call__(self, slot):
        if slot.startswith('ldsr_'):
            from trustsr.models.ldsr_assets import CHECKPOINT_NAME
            from trustsr.models.ldsr_s2 import LDSRS2X4

            root = self._require_assets(self.ldsr_directory, [CHECKPOINT_NAME])
            if 'ldsr' not in self.loaded:
                if (not torch.cuda.is_available()
                        or torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 4090'):
                    raise ValueError('verified CUDA hardware required')
                self.loaded['ldsr'] = LDSRS2X4.from_pretrained(root, device='cuda:0')
            return self.loaded['ldsr'].for_seed(int(slot.removeprefix('ldsr_')))
        if slot == 'sen2sr':
            from trustsr.models.cloud_sen2srlite import CloudSEN2SRLiteX4
            from trustsr.models.sen2srlite import MODEL_ASSET_SHA256

            root = self._require_assets(self.sen2sr_directory, MODEL_ASSET_SHA256)
            if slot not in self.loaded:
                self.loaded[slot] = CloudSEN2SRLiteX4.from_pretrained(root, device='cpu')
            return self.loaded[slot]
        if slot == 'bicubic':
            from trustsr.models.bicubic import BicubicX4

            if slot not in self.loaded:
                self.loaded[slot] = BicubicX4()
            return self.loaded[slot]
        raise ValueError('unplanned model slot')


def _timeout(signum, frame):
    raise TimeoutError('approved execution wall-time limit reached')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    draft = commands.add_parser('draft', help='write a text-only non-runnable protocol draft')
    draft.add_argument('--output', type=Path, required=True)
    freeze = commands.add_parser('freeze', help='freeze the contract; user manages costs')
    freeze.add_argument('--output', type=Path, required=True)
    for name in ('preflight', 'run', 'replay'):
        command = commands.add_parser(name)
        command.add_argument('--protocol', type=Path, required=True)
        command.add_argument('--protocol-sha256', required=True)
        if name != 'preflight':
            command.add_argument('--confirm-external-access', action='store_true')
            command.add_argument('--package-directory', type=Path, required=True)
            command.add_argument('--output', type=Path, required=True)
        if name == 'run':
            command.add_argument('--allow-gpu', action='store_true')
            command.add_argument('--ldsr-model-directory', type=Path)
            command.add_argument('--sen2sr-model-directory', type=Path)
    args = parser.parse_args(argv)
    try:
        verify_import_origin()
        if args.command in ('draft', 'freeze'):
            builder = build_frozen if args.command == 'freeze' else build_draft
            protocol = builder(REPOSITORY)
            raw = canonical_json(protocol)
            digest = hashlib.sha256(raw).hexdigest()
            if args.command == 'freeze':
                load_frozen(raw, digest, REPOSITORY)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_once(args.output, raw)
            print(json.dumps({'status': protocol['status'], 'sha256': digest}))
            return 0
        if args.command in ('run', 'replay') and not args.confirm_external_access:
            raise ValueError('dedicated frozen external-access acknowledgement required')
        protocol = load_frozen(read_regular(args.protocol, 1_048_576),
                               args.protocol_sha256, REPOSITORY)
        verify_runtime(protocol)
        if args.command == 'preflight':
            print(json.dumps({'status': 'ready', 'protocol_sha256': args.protocol_sha256,
                              'external_pixels_read': False}))
            return 0
        ldsr = getattr(args, 'ldsr_model_directory', None)
        sen2sr = getattr(args, 'sen2sr_model_directory', None)
        validate_paths(args.output, args.package_directory,
                       [root for root in (ldsr, sen2sr) if root is not None], REPOSITORY)
        old_threads = torch.get_num_threads()
        old_handler = signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, protocol['budget']['maximum_wall_seconds'])
        try:
            torch.set_num_threads(1)
            input_start = time.perf_counter()
            subsets, receipts = {}, {}
            for name, spec in protocol['science']['subsets'].items():
                subsets[name], receipts[name] = decode_package(
                    args.package_directory / spec['package_filename'],
                    expected_sha256=spec['package_sha256'], expected_size=spec['package_size'],
                    members=spec['members'])
            result = run_study(subsets, protocol, args.output, ModelFactory(ldsr, sen2sr),
                               allow_ldsr=bool(getattr(args, 'allow_gpu', False)
                                               and torch.cuda.is_available()),
                               replay_only=args.command == 'replay', input_receipts=receipts,
                               input_seconds=time.perf_counter() - input_start)
            result_sha = hashlib.sha256(canonical_json(result)).hexdigest()
            print(json.dumps({'status': 'verified', 'protocol_sha256': args.protocol_sha256,
                              'science_sha256': result_sha}))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            torch.set_num_threads(old_threads)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
