import io
import struct
import zipfile

import pytest

from research.trustmask.zip_metadata import (
    directory_entries,
    eocd_interval,
    read_metadata_member,
)


def archive(content=b'{"source_id":"S2_sample"}', compression=zipfile.ZIP_DEFLATED):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=compression) as z:
        z.writestr('ROI_0000/lr.tif', b'PIXEL_PAYLOAD_DO_NOT_READ' * 8)
        z.writestr('ROI_0000/metadata.json', content)
    raw = stream.getvalue()
    start, length, count = eocd_interval(raw[-22:], len(raw))
    entries = directory_entries(raw[start:start + length], start, count)
    return raw, start, entries


@pytest.mark.parametrize('compression', [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_only_json_ranges_are_read_and_image_request_stops_before_fetch(compression):
    raw, start, entries = archive(compression=compression)
    entry = entries[1]
    calls = []
    def fetch(offset, length):
        assert entry['offset'] <= offset < offset + length <= start
        calls.append((offset, length))
        return raw[offset:offset + length]
    assert read_metadata_member(fetch, entry, start) == b'{"source_id":"S2_sample"}'
    assert len(calls) == 3
    with pytest.raises(ValueError):
        read_metadata_member(fetch, entries[0], start)
    assert len(calls) == 3


@pytest.mark.parametrize('field,value', [(0, b'bad!'), (1, 1), (3, 65535), (4, 1),
                                        (5, 9000000), (6, 1), (7, 1)])
def test_unreviewed_or_overlapping_end_record_is_rejected(field, value):
    fields = [b'PK\x05\x06', 0, 0, 2, 2, 100, 1000, 0]
    fields[field] = value
    with pytest.raises(ValueError):
        eocd_interval(struct.pack('<4s4H2LH', *fields), 1122)


def test_truncated_or_miscounted_directory_is_rejected():
    raw, start, _ = archive()
    for directory, count in [(raw[start:-23], 2), (raw[start:-22], 1)]:
        with pytest.raises(ValueError):
            directory_entries(directory, start, count)


@pytest.mark.parametrize('change', ['name', 'method', 'crc', 'size', 'encrypted', 'offset'])
def test_bad_local_header_is_rejected_before_payload(change):
    raw, start, entries = archive()
    entry = entries[1]
    damaged = bytearray(raw)
    o = entry['offset']
    if change == 'name':
        damaged[o + 30] = ord('X')
    elif change == 'method':
        struct.pack_into('<H', damaged, o + 8, 0)
    elif change == 'crc':
        struct.pack_into('<L', damaged, o + 14, 0)
    elif change == 'size':
        struct.pack_into('<L', damaged, o + 22, 262145)
    elif change == 'encrypted':
        struct.pack_into('<H', damaged, o + 6, 1)
    else:
        entry = dict(entry, offset=start - 10)
    calls = []
    def fetch(offset, length):
        calls.append((offset, length))
        return bytes(damaged[offset:offset + length])
    with pytest.raises(ValueError):
        read_metadata_member(fetch, entry, start)
    assert len(calls) <= 2


def test_crc_failure_and_oversized_json_are_rejected():
    raw, start, entries = archive(compression=zipfile.ZIP_STORED)
    altered = bytearray(raw)
    altered[start - 1] ^= 1
    with pytest.raises(ValueError):
        read_metadata_member(lambda o, n: bytes(altered[o:o+n]), entries[1], start)
    raw, start, entries = archive(b'A' * 262145)
    def no_fetch(*args):
        raise AssertionError('oversized metadata rejected before any read')
    with pytest.raises(ValueError):
        read_metadata_member(no_fetch, entries[1], start)


def test_image_declared_payload_cannot_overlap_next_metadata_header():
    raw, start, entries = archive(compression=zipfile.ZIP_STORED)
    directory = bytearray(raw[start:-22])
    # This passes a bound that forgets the image's local filename bytes.
    bad_size = entries[1]['offset'] - entries[0]['offset'] - 30
    struct.pack_into('<L', directory, 20, bad_size)
    with pytest.raises(ValueError, match='overlapping'):
        directory_entries(bytes(directory), start, 2)


def test_streaming_zip_data_descriptor_uses_central_sizes():
    class Streaming(io.BytesIO):
        def seek(self, *args):
            raise OSError('stream cannot seek')
    stream = Streaming()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('ROI_0000/metadata.json', b'{"id":"S2"}')
    raw = stream.getvalue()
    start, length, count = eocd_interval(raw[-22:], len(raw))
    entry = directory_entries(raw[start:start+length], start, count)[0]
    assert entry['flags'] & 8
    assert read_metadata_member(lambda o, n: raw[o:o+n], entry, start) == b'{"id":"S2"}'


def test_deflate_output_cannot_exceed_falsely_declared_size():
    raw, start, entries = archive(b'A' * 20000)
    entry = dict(entries[1], size=100)
    altered = bytearray(raw)
    struct.pack_into('<L', altered, entry['offset'] + 22, 100)
    with pytest.raises(ValueError):
        read_metadata_member(lambda o, n: bytes(altered[o:o+n]), entry, start)
