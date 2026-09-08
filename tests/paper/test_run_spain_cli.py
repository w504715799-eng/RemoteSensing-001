import importlib

import pytest


def api():
    return importlib.import_module('scripts.paper.run_spain')


def test_draft_cli_has_no_pixel_model_or_network_dependency(tmp_path, monkeypatch):
    module = api()

    def forbidden(*args, **kwargs):
        pytest.fail('draft accessed external data')

    monkeypatch.setattr(module, 'decode_package', forbidden)
    path = tmp_path / 'draft.json'
    assert module.main(['draft', '--output', str(path)]) == 0
    assert '"status":"draft"' in path.read_text()


def test_execution_without_acknowledgement_stops_before_protocol_or_pixels(tmp_path, monkeypatch):
    module = api()

    def forbidden(*args, **kwargs):
        pytest.fail('missing access acknowledgement reached data/protocol')

    monkeypatch.setattr(module, 'read_regular', forbidden)
    with pytest.raises(SystemExit):
        module.main(['run', '--protocol', '/missing/protocol.json', '--protocol-sha256', '0' * 64,
                     '--package-directory', str(tmp_path), '--output', str(tmp_path / 'run')])
    assert not (tmp_path / 'run').exists()


def test_draft_cannot_be_executed_even_with_access_flag(tmp_path, monkeypatch):
    import hashlib

    module = api()
    draft = tmp_path / 'draft.json'
    module.main(['draft', '--output', str(draft)])

    def forbidden(*args, **kwargs):
        pytest.fail('draft reached external pixels')

    monkeypatch.setattr(module, 'decode_package', forbidden)
    with pytest.raises(SystemExit):
        module.main(['run', '--protocol', str(draft), '--protocol-sha256',
                     hashlib.sha256(draft.read_bytes()).hexdigest(),
                     '--package-directory', str(tmp_path), '--output', str(tmp_path / 'run'),
                     '--confirm-external-access'])
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('location', ['repo', 'packages', 'parent', 'old_phase', 'symlink'])
def test_run_paths_cannot_overlap_protected_trees(tmp_path, location):
    root = tmp_path / 'repo'
    packages = tmp_path / 'packages'
    root.mkdir()
    packages.mkdir()
    output = root / 'run'
    if location == 'packages':
        output = packages / 'run'
    if location == 'parent':
        output = tmp_path
    if location == 'old_phase':
        output = tmp_path / 'phase2b3c' / 'run'
    if location == 'symlink':
        alias = tmp_path / 'alias'
        alias.symlink_to(root, target_is_directory=True)
        output = alias / 'run'
    with pytest.raises(ValueError):
        api().validate_paths(output, packages, [], root)


def test_cache_only_factory_does_not_require_model_directories():
    factory = api().ModelFactory(None, None)
    assert factory is not None
    with pytest.raises(ValueError):
        factory('ldsr_3407')


def test_stale_installed_package_cannot_generate_or_execute_protocol(tmp_path, monkeypatch):
    import trustsr

    module = api()
    monkeypatch.setattr(trustsr, '__file__', str(tmp_path / 'stale-wheel/trustsr/__init__.py'))
    with pytest.raises(SystemExit):
        module.main(['draft', '--output', str(tmp_path / 'draft.json')])
    assert not (tmp_path / 'draft.json').exists()


def test_freeze_cli_writes_loadable_protocol_without_external_access(tmp_path, monkeypatch):
    import hashlib
    import json

    module = api()

    def forbidden(*args, **kwargs):
        pytest.fail('text-only freeze reached runtime or external data')

    monkeypatch.setattr(module, 'decode_package', forbidden)
    monkeypatch.setattr(module, 'verify_runtime', forbidden)
    path = tmp_path / 'frozen.json'
    assert module.main(['freeze', '--output', str(path)]) == 0
    raw = path.read_bytes()
    result = module.load_frozen(raw, hashlib.sha256(raw).hexdigest(), module.REPOSITORY)
    assert result['status'] == 'frozen'
    assert result['budget']['cost_management'] == 'user'
    assert 'hourly_price' not in json.loads(raw)['budget']
    # Repeating text publication is idempotent, but conflicting files are never overwritten.
    assert module.main(['freeze', '--output', str(path)]) == 0
    path.write_text('{}')
    with pytest.raises(SystemExit):
        module.main(['freeze', '--output', str(path)])
    assert path.read_text() == '{}'
