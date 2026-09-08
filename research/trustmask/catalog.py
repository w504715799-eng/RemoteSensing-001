"""Bounded legacy TACO metadata retrieval; never enumerates or reads image payloads."""

import json
import urllib.error
import urllib.parse
import urllib.request


def metadata_intervals(header: bytes, total_bytes: int) -> dict:
    if type(total_bytes) is not int or total_bytes < 42:
        raise ValueError('positive source size required')
    if type(header) is not bytes or len(header) != 42 or header[:2] not in (b'WX', b'#y'):
        raise ValueError('legacy 42-byte TACO header required')
    intervals = {
        'directory': (int.from_bytes(header[2:10], 'little'),
                      int.from_bytes(header[10:18], 'little')),
        'collection': (int.from_bytes(header[26:34], 'little'),
                       int.from_bytes(header[34:42], 'little')),
    }
    for name, (start, length) in intervals.items():
        cap = 8 * 1024 * 1024 if name == 'directory' else 64 * 1024
        if start < 42 or not 1 <= length <= cap or start + length > total_bytes:
            raise ValueError('metadata interval outside bounded source contract')
    (a, n), (b, m) = intervals.values()
    if max(a, b) < min(a + n, b + m):
        raise ValueError('metadata intervals overlap')
    return intervals


def fetch_metadata(fetch_range, total_bytes: int) -> dict:
    """Read only header, collection JSON, and top-level Parquet via exact-range callback."""
    raw = fetch_range(0, 42)
    intervals = metadata_intervals(raw, total_bytes)
    collection = fetch_range(*intervals['collection'])
    if type(collection) is not bytes or len(collection) != intervals['collection'][1]:
        raise ValueError('collection length mismatch')
    obj = json.loads(collection)
    if not isinstance(obj, dict):
        raise ValueError('collection must be a JSON object')
    versions = [obj[k] for k in ('taco_version', 'version') if k in obj]
    if not versions or any(v != '0.4.0' for v in versions):
        raise ValueError('unreviewed or conflicting TACO collection version')
    directory = fetch_range(*intervals['directory'])
    if (type(directory) is not bytes or len(directory) != intervals['directory'][1]
            or len(directory) < 12 or directory[:4] != b'PAR1' or directory[-4:] != b'PAR1'):
        raise ValueError('invalid top-level Parquet directory')
    return {'header': raw, 'collection': collection, 'directory': directory, 'intervals': intervals}


def read_range_response(response, offset, length, total_bytes):
    """Check headers before reading at most the requested metadata length plus one byte."""
    expected = f'bytes {offset}-{offset + length - 1}/{total_bytes}'
    if (response.status != 206 or response.headers.get('Content-Range') != expected
            or response.headers.get('Content-Encoding', 'identity') != 'identity'):
        raise ValueError('server did not return the exact unencoded metadata range')
    raw = response.read(length + 1)
    if len(raw) != length:
        raise ValueError('metadata response body length mismatch')
    return raw


class NoBodyRedirect(urllib.request.HTTPRedirectHandler):
    """Surface redirects to our bounded loop without urllib reading their bodies."""

    def http_error_302(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def fetch_http_range(url, offset, length, total_bytes):
    """Exact HTTPS range with bounded body-free redirects to HF-owned asset hosts."""
    if (any(type(v) is not int for v in (offset, length, total_bytes)) or offset < 0
            or not 1 <= length <= 8 * 1024 * 1024 or offset + length > total_bytes):
        raise ValueError('invalid bounded HTTP range')
    opener = urllib.request.build_opener(NoBodyRedirect())
    current = url
    for _ in range(6):
        parsed = urllib.parse.urlsplit(current)
        host = parsed.hostname or ''
        if (parsed.scheme != 'https' or parsed.username or parsed.password
                or parsed.port not in (None, 443)
                or not (host == 'huggingface.co' or host.endswith('.huggingface.co')
                        or host == 'hf.co' or host.endswith('.hf.co'))):
            raise ValueError('unexpected metadata redirect target')
        request = urllib.request.Request(current, headers={
            'Range': f'bytes={offset}-{offset + length - 1}', 'Accept-Encoding': 'identity'})
        try:
            response = opener.open(request, timeout=30)
        except urllib.error.HTTPError as error:
            code, location = error.code, error.headers.get('Location')
            error.close()
            if code not in (301, 302, 303, 307, 308) or not location or len(location) > 32768:
                raise ValueError(f'metadata HTTP request failed (status {code})') from None
            current = urllib.parse.urljoin(current, location)
            continue
        with response:
            return read_range_response(response, offset, length, total_bytes)
    raise ValueError('too many metadata redirects')


def nested_directory_interval(header, parent_offset, parent_length, top_directory_start):
    """Locate only a nested metadata footer inside an authorized top-directory parent.

    Nested TORTILLA headers are18bytes, unlike the42byte outer collection header.
    Reading42bytes here could reach asset bytes and is deliberately unsupported.
    """
    if (any(type(v) is not int for v in (parent_offset, parent_length, top_directory_start))
            or parent_offset < 42 or parent_length < 30
            or parent_offset + parent_length > top_directory_start):
        raise ValueError('invalid top-authorized parent interval')
    if type(header) is not bytes or len(header) != 18 or header[:2] not in (b'#y', b'WX'):
        raise ValueError('exact18byte nested metadata header required')
    offset = int.from_bytes(header[2:10], 'little')
    length = int.from_bytes(header[10:18], 'little')
    if offset < 18 or not 12 <= length <= 64 * 1024 or offset + length > parent_length:
        raise ValueError('nested metadata outside authorized parent or size cap')
    return parent_offset + offset, length
