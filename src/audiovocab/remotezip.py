"""Read individual members of a large remote zip over HTTP range requests, without downloading the whole archive.

Used for FMA (fma_large.zip is ~93 GB; we need ~3k of its 106k 30 s clips). Works for any host that honours
`Range` (os.unil.cloud.switch.ch, zenodo.org).
"""
import io
import zipfile

import requests
import truststore

truststore.inject_into_ssl()  # corporate TLS inspection

UA = {"User-Agent": "Mozilla/5.0 audio-vocab-research"}


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP. Reads ahead in blocks to keep zipfile's small reads cheap."""

    def __init__(self, url, block=1 << 20, session=None):
        self.url, self.block, self.pos = url, block, 0
        self.s = session or requests.Session()
        # A 1-byte ranged GET, not HEAD: the proxy drops HEAD requests to some hosts.
        r = self.s.get(url, headers={**UA, "Range": "bytes=0-0"}, timeout=60, allow_redirects=True)
        r.raise_for_status()
        self.url = r.url
        self.size = int(r.headers["content-range"].rsplit("/", 1)[1])
        self._buf_start, self._buf = 0, b""
        self.requests = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _fetch(self, start, end):
        self.requests += 1
        r = self.s.get(self.url, headers={**UA, "Range": f"bytes={start}-{end - 1}"}, timeout=120)
        if r.status_code != 206:
            raise IOError(f"range request not honoured: HTTP {r.status_code}")
        return r.content

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        start, end = self.pos, self.pos + n
        b0, b1 = self._buf_start, self._buf_start + len(self._buf)
        if not (b0 <= start and end <= b1):
            fetch_end = min(self.size, max(end, start + self.block))
            self._buf_start, self._buf = start, self._fetch(start, fetch_end)
            b0 = start
        out = self._buf[start - b0:end - b0]
        self.pos = end
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_remote_zip(url, **kw):
    return zipfile.ZipFile(HTTPRangeFile(url, **kw))
