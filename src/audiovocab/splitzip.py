"""Read members of a split zip (name.z01, name.z02, ..., name.zip) with the standard library, no 7-Zip.

The parts are exposed as one virtual file. zipfile then finds the central directory at the end, but it shifts every
member offset by the size of the disks before the central directory's disk. Members on other disks are corrected
using each entry's disk number (`ZipInfo.volume`).

Limitation: Python refuses multi-disk *zip64* archives (FSD50K's eval split is one). For those, join the parts with
`zip -s 0 name.zip --out joined.zip` first, as colab/fsd50k_embed.ipynb does.
"""
import io
import zipfile


class _Concat(io.RawIOBase):
    def __init__(self, paths):
        self.files = [open(p, "rb") for p in paths]
        self.sizes = [f.seek(0, 2) for f in self.files]
        self.starts = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.size, self.pos = sum(self.sizes), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i = max(k for k, s in enumerate(self.starts) if s <= self.pos)
            f = self.files[i]
            f.seek(self.pos - self.starts[i])
            chunk = f.read(min(n, self.sizes[i] - (self.pos - self.starts[i])))
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    def close(self):
        for f in self.files:
            f.close()
        super().close()


class SplitZip(zipfile.ZipFile):
    def __init__(self, paths):
        self._cat = _Concat(paths)
        super().__init__(self._cat)
        cd_disk = len(paths) - 1
        shift = self._cat.starts[cd_disk]  # what zipfile added to every offset
        for zi in self.infolist():
            zi.header_offset = zi.header_offset - shift + self._cat.starts[zi.volume]

    def close(self):
        super().close()
        self._cat.close()
