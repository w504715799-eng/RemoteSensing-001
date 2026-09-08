import json

import pytest

from research.trustmask.catalog import fetch_metadata, metadata_intervals


def header(directory=(100, 20), collection=(50, 30)):
    return (b'WX' + directory[0].to_bytes(8, 'little') + directory[1].to_bytes(8, 'little')
            + (1).to_bytes(8, 'little') + collection[0].to_bytes(8, 'little')
            + collection[1].to_bytes(8, 'little'))


def test_ranges_are_only_the_declared_top_level_metadata():
    assert metadata_intervals(header(), 1000) == {'collection': (50, 30), 'directory': (100, 20)}


@pytest.mark.parametrize('raw,total', [
    (b'x' * 42, 1000), (header()[:-1], 1000), (header(), 119),
    (header(directory=(70, 20)), 1000), (header(collection=(0, 30)), 1000),
    (header(directory=(100, 0)), 1000), (header(directory=(100, 9_000_000)), 20_000_000),
    (header(collection=(50, 70_000)), 100_000), (header(), True),
])
def test_bad_header_is_rejected_without_network_followup(raw, total):
    with pytest.raises(ValueError):
        metadata_intervals(raw, total)


def test_orchestrator_fetches_no_pixel_or_nested_intervals():
    collection = json.dumps({'taco_version': '0.4.0'}).encode()
    raw = header(collection=(50, len(collection)))
    calls = []
    def fetch(offset, length):
        calls.append((offset, length))
        return {(0, 42): raw, (50, len(collection)): collection,
                (100, 20): b'PAR1' + b'\0' * 12 + b'PAR1'}[(offset, length)]
    result = fetch_metadata(fetch, 1000)
    assert calls == [(0, 42), (50, len(collection)), (100, 20)]
    assert result['directory'] == b'PAR1' + b'\0' * 12 + b'PAR1'


def test_wrong_version_stops_before_directory_fetch():
    collection = json.dumps({'taco_version': '2.0'}).encode()
    calls = []
    def fetch(offset, length):
        calls.append((offset, length))
        return header(collection=(50, len(collection))) if offset == 0 else collection
    with pytest.raises(ValueError):
        fetch_metadata(fetch, 1000)
    assert len(calls) == 2


def test_whole_file_response_is_rejected_before_body_read():
    from research.trustmask.catalog import read_range_response
    class Response:
        status = 200
        headers = {}
        def read(self, n):
            raise AssertionError('must not read a full file response')
    with pytest.raises(ValueError):
        read_range_response(Response(), 0, 42, 1000)


def test_wrong_range_or_short_body_is_rejected():
    from research.trustmask.catalog import read_range_response
    class Response:
        status = 206
        headers = {'Content-Range': 'bytes 0-41/1000'}
        def read(self, n):
            return b'X' * 41
    with pytest.raises(ValueError):
        read_range_response(Response(), 0, 42, 1000)
    with pytest.raises(ValueError):
        read_range_response(Response(), 2, 42, 1000)


def test_redirect_handler_never_reads_response_body():
    import urllib.error
    import urllib.request

    from research.trustmask.catalog import NoBodyRedirect
    class Body:
        def read(self, *args):
            raise AssertionError('redirect body must not be read')
        def close(self):
            pass
    with pytest.raises(urllib.error.HTTPError) as caught:
        NoBodyRedirect().http_error_302(urllib.request.Request('https://huggingface.co/a'),
                                      Body(), 302, 'redirect', {'Location': 'https://hf.co/b'})
    caught.value.close()


def test_historical_identity_requires_published_digest(tmp_path):
    from research.trustmask.audit_catalog import HISTORY, historical_ids
    for path, _ in HISTORY:
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'samples': [{'sample_id': str(i)} for i in range(120)]}))
    with pytest.raises(ValueError, match='SHA'):
        historical_ids(tmp_path)


def test_nested_directory_is_relative_to_its_authorized_parent():
    from research.trustmask.catalog import nested_directory_interval
    raw = b'#y' + (100).to_bytes(8, 'little') + (20).to_bytes(8, 'little')
    assert nested_directory_interval(raw, 200, 500, 1000) == (300, 20)


@pytest.mark.parametrize('raw,parent,length,top', [
    (b'#y' + (1).to_bytes(8, 'little') + (20).to_bytes(8, 'little'), 200, 500, 1000),
    (b'#y' + (490).to_bytes(8, 'little') + (20).to_bytes(8, 'little'), 200, 500, 1000),
    (b'#y' + (100).to_bytes(8, 'little') + (0).to_bytes(8, 'little'), 200, 500, 1000),
    (b'#y' + (100).to_bytes(8, 'little') + (20).to_bytes(8, 'little'), 900, 500, 1000),
    (b'#y' + (100).to_bytes(8, 'little') + (20).to_bytes(8, 'little'), True, 500, 1000),
    (b'x' * 18, 200, 500, 1000),
])
def test_nested_bad_offsets_stop_before_any_directory_request(raw, parent, length, top):
    from research.trustmask.catalog import nested_directory_interval
    with pytest.raises(ValueError):
        nested_directory_interval(raw, parent, length, top)
