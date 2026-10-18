"""Local durable checkpoints, exclusive runner lock and child lifetime guard.

Each path has one writer. File contents are fsynced before atomic replacement.
This protects process interruption; it is not a guarantee against disk failure.
"""
from __future__ import annotations
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import time
import uuid
import zipfile
from datetime import datetime,timezone

def utc_now():return datetime.now(timezone.utc).isoformat()
def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def _replace(source,target):
    for attempt in range(8):
        try:os.replace(source,target);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.02*(attempt+1))

def atomic_bytes(path,payload,keep_previous=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    if keep_previous and path.exists():atomic_bytes(path.with_name(path.name+'.previous'),path.read_bytes())
    _replace(temp,path)
    if os.name!='nt':
        descriptor=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)

def atomic_json(path,value,keep_previous=True):
    atomic_bytes(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf-8'),keep_previous)

def sealed_json(path,value,keep_previous=False):
    value=dict(value);value.pop('record_sha256',None);value['record_sha256']=canonical_hash(value)
    atomic_json(path,value,keep_previous)

def read_sealed(path):
    value=read_json(path);payload=dict(value);digest=payload.pop('record_sha256',None)
    if digest!=canonical_hash(payload):raise ValueError('Artifact content seal differs: '+str(path))
    return value

def recover_json(path):
    """Explicitly recover previous valid JSON, retaining damaged bytes and notice."""
    path=Path(path)
    try:return read_json(path),False
    except (json.JSONDecodeError,UnicodeDecodeError):
        previous=path.with_name(path.name+'.previous');value=read_json(previous)
        damaged=path.with_name(path.name+'.corrupt.'+uuid.uuid4().hex)
        atomic_bytes(damaged,path.read_bytes())
        atomic_json(path,value,keep_previous=False)
        atomic_json(path.with_name(path.name+'.recovery.json'),dict(utc=utc_now(),recovered_from=str(previous),retained_damage=str(damaged)),False)
        return value,True

def atomic_npz(path,**arrays):
    """Deterministic ZIP timestamps make same-seed recovery byte-identical."""
    import numpy as np
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for key,array in sorted(arrays.items()):
            item=io.BytesIO();np.save(item,np.asarray(array),allow_pickle=False)
            info=zipfile.ZipInfo(key+'.npy',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,item.getvalue(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    path=Path(path);data=payload.getvalue()
    if path.exists():
        if path.read_bytes()!=data:raise ValueError(f'Refusing to replace a different saved catalogue: {path}')
        return
    atomic_bytes(path,data)

class RunLock:
    """An OS lock, not a PID-file liveness heuristic; crash releases it."""
    def __init__(self,path):self.path=Path(path);self.stream=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('a+b')
        if self.path.stat().st_size==0:self.stream.write(b'0');self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.stream.close();self.stream=None
            raise RuntimeError('Another runner holds the campaign lock')
        return self
    def __exit__(self,*args):
        if self.stream is not None:
            self.stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_UN)
            self.stream.close();self.stream=None

class ChildJob:
    """On Windows, parent death closes this job and kills assigned children."""
    def __init__(self):
        self.handle=None
        if os.name!='nt':return
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_=[('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),('LimitFlags',w.DWORD),
                      ('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',w.DWORD),
                      ('Affinity',ctypes.c_size_t),('PriorityClass',w.DWORD),('SchedulingClass',w.DWORD)]
        class IO(ctypes.Structure):_fields_=[(x,ctypes.c_ulonglong) for x in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
        class Extended(ctypes.Structure):
            _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];self.kernel.CreateJobObjectW.restype=w.HANDLE
        self.kernel.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD];self.kernel.SetInformationJobObject.restype=w.BOOL
        self.kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];self.kernel.OpenProcess.restype=w.HANDLE
        self.kernel.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];self.kernel.AssignProcessToJobObject.restype=w.BOOL
        self.kernel.CloseHandle.argtypes=[w.HANDLE];self.kernel.CloseHandle.restype=w.BOOL
        self.handle=self.kernel.CreateJobObjectW(None,None)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        limits=Extended();limits.BasicLimitInformation.LimitFlags=0x2000
        if not self.kernel.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            error=ctypes.get_last_error();self.close();raise ctypes.WinError(error)
    def assign(self,process):
        if os.name!='nt':return
        handle=self.kernel.OpenProcess(0x0100|0x0001|0x0400,False,process.pid)
        if not handle:raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle,handle):raise ctypes.WinError(ctypes.get_last_error())
        finally:self.kernel.CloseHandle(handle)
    def close(self):
        if self.handle:self.kernel.CloseHandle(self.handle);self.handle=None
