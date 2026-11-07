"""Publish a complete command once, including on Linux exFAT volumes.

Never expose a partially written destination or overwrite an unconsumed job.
There is deliberately no copy/O_EXCL fallback visible to the owner.
"""
import ctypes
import errno
import os


def publish(temp, destination):
    try:
        os.link(temp, destination)
        return
    except OSError as error:
        if error.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS):raise
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise OSError(errno.ENOTSUP, 'Command storage requires atomic no-replace publication')
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    # AT_FDCWD=-100, RENAME_NOREPLACE=1. The kernel checks the destination
    # atomically; an existing request always remains untouched.
    if rename(-100, os.fsencode(temp), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), os.fspath(destination))
