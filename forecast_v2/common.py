from pathlib import Path
import hashlib
import json
import os
import socket
import time
import ctypes
from contextlib import contextmanager


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def safe_path(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes package')
    return path


def verify_data(root):
    root = Path(root); manifest = read(root / 'manifest.json')
    for relative, digest in manifest['files'].items():
        if sha(safe_path(root, relative)) != digest:
            raise ValueError(f'Data fingerprint changed: {relative}')
    groups = [set(manifest['splits'][s]) for s in ('train', 'val', 'calibration')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Overlapping sequences')
    if set.union(*groups) & set(manifest['excluded_exposed_test']):
        raise ValueError('Original test sequences are forbidden in v2 development')
    return manifest


def sources():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def alive(pid):
    if os.name == 'nt':
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle: return False
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code)); kernel.CloseHandle(handle)
        return bool(ok and code.value == 259)
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False


@contextmanager
def lock(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'run.lock'
    if path.exists():
        old = read(path)
        if old['host'] != socket.gethostname() or alive(old['pid']):
            raise RuntimeError('Run is locked. Never copy an active run; use a new run folder on another PC.')
        path.unlink()
    with path.open('x') as stream:
        json.dump({'pid': os.getpid(), 'host': socket.gethostname(), 'started': time.time()}, stream)
    try: yield
    finally: path.unlink(missing_ok=True)
