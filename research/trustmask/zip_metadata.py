"""Bounded classic ZIP directory and JSON reads; no generic archive extraction."""

import re
import struct
import zlib


def eocd_interval(raw, total):
    """Accept only a classic, single-disk, comment-free final 22-byte record."""
    if type(total) is not int or total < 22 or type(raw) is not bytes or len(raw) != 22:
        raise ValueError('exact ZIP end record required')
    sig, disk, cd_disk, disk_count, count, length, start, comment = struct.unpack('<4s4H2LH', raw)
    if (sig != b'PK\x05\x06' or disk or cd_disk or comment or disk_count != count
            or not 0 < count < 65535 or not 46 <= length <= 8 * 1024 * 1024
            or start < 30 or start + length != total - 22):
        raise ValueError('unreviewed ZIP layout or directory bounds')
    return start, length, count


def directory_entries(raw, start, count):
    """Parse bounded central records, preserving names and local-record boundaries."""
    if (type(raw) is not bytes or not 46 <= len(raw) <= 8 * 1024 * 1024
            or type(start) is not int or start < 30
            or type(count) is not int or not 0 < count < 65535):
        raise ValueError('invalid directory contract')
    pos, entries = 0, []
    for _ in range(count):
        if pos + 46 > len(raw):
            raise ValueError('truncated central record')
        f = struct.unpack_from('<4s6H3L5H2L', raw, pos)
        sig, flags, method, crc, compressed, size = f[0], f[3], f[4], f[7], f[8], f[9]
        namesize, extra, comment, disk, offset = f[10], f[11], f[12], f[13], f[16]
        end = pos + 46 + namesize + extra + comment
        if (sig != b'PK\x01\x02' or disk or not namesize or end > len(raw)
                or max(compressed, size, offset) == 0xffffffff
                or offset + 30 + namesize + compressed > start):
            raise ValueError('invalid central record or local bounds')
        name = raw[pos + 46:pos + 46 + namesize].decode('utf-8' if flags & 2048 else 'cp437')
        if '\x00' in name:
            raise ValueError('NUL in ZIP name')
        entries.append(dict(name=name, offset=offset, flags=flags, method=method,
                            crc=crc, compressed=compressed, size=size, name_bytes=namesize))
        pos = end
    if pos != len(raw) or len({e['name'] for e in entries}) != count:
        raise ValueError('directory count, trailing bytes or duplicate name')
    ordered = sorted(entries, key=lambda e: e['offset'])
    for i, entry in enumerate(ordered):
        limit = ordered[i + 1]['offset'] if i + 1 < count else start
        if entry['offset'] + 30 + entry['name_bytes'] + entry['compressed'] > limit:
            raise ValueError('overlapping local records')
        entry['limit'] = limit
    return entries


def read_metadata_member(fetch, entry, directory_start):
    """Read one directory-selected metadata.json after validating its local header.

    The callback must enforce exact HTTP ranges. Entries must come from the pinned
    central directory; passing an invented dict provides no source authorization.
    """
    name = entry['name']
    if (not isinstance(name, str)
            or not re.fullmatch(r'(?:[A-Za-z0-9_-]+/)*metadata\.json', name)):
        raise ValueError('only metadata.json may be read')
    fields = [entry[k] for k in ('offset', 'flags', 'method', 'crc', 'compressed', 'size', 'limit')]
    if any(type(v) is not int or v < 0 for v in fields) or type(directory_start) is not int:
        raise ValueError('integer member bounds required')
    offset, flags, method, crc, compressed, size, limit = fields
    if (flags & ~0x80e or method not in (0, 8) or not 0 < compressed <= 65536
            or not 0 < size <= 262144 or offset + 30 > limit or limit > directory_start):
        raise ValueError('unreviewed compression, flags or metadata size')
    def read(o, n):
        if n <= 0 or o < offset or o + n > limit:
            raise ValueError('read exceeds selected local record')
        b = fetch(o, n)
        if type(b) is not bytes or len(b) != n:
            raise ValueError('short metadata range')
        return b
    header = read(offset, 30)
    sig, version, local_flags, local_method, _, _, local_crc, csize, usize, nlen, extra = (
        struct.unpack('<4s5H3L2H', header))
    encoded = name.encode('utf-8' if flags & 2048 else 'cp437')
    if (sig != b'PK\x03\x04' or version > 20 or local_flags != flags or local_method != method
            or nlen != len(encoded) or extra > 65535):
        raise ValueError('central/local header mismatch')
    for local, expected in ((local_crc, crc), (csize, compressed), (usize, size)):
        if local != expected and not (flags & 8 and local == 0):
            raise ValueError('central/local size or CRC mismatch')
    data_start = offset + 30 + nlen + extra
    if data_start + compressed > limit:
        raise ValueError('metadata payload exceeds local boundary')
    prefix = read(offset + 30, nlen + extra)
    if prefix[:nlen] != encoded:
        raise ValueError('central/local filename mismatch')
    payload = read(data_start, compressed)
    if method == 0:
        result = payload
    else:
        decoder = zlib.decompressobj(-15)
        try:
            result = decoder.decompress(payload, size + 1)
        except zlib.error:
            raise ValueError('invalid deflate stream') from None
        if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError('incomplete or oversized deflate stream')
    if len(result) != size or zlib.crc32(result) != crc:
        raise ValueError('metadata size or CRC mismatch')
    return result
