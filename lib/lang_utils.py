import sys
import importlib
from collections.abc import Iterable
from dataclasses import dataclass

def from_str(cast_func, s, default_value):
    try:
        return cast_func(s)
    except ValueError:
        return default_value

def coalesce_fn(v, fn, default_value):
    if v is None:
        return default_value

    return fn(v)

def coalesce(*repertoire):
    v = None
    
    if not isinstance(repertoire, Iterable):
        repertoire = (repertoire,)
        
    for r in repertoire:
        if r is not None:
            v = r() if callable(r) else r

        if v is not None:
            break

    return v

def when(v, if_true, if_false):
    if v:
        return if_true() if callable(if_true) else if_true
    else:
        return if_false() if callable(if_false) else if_false

def to_number(v):
    assert isinstance(v, str)
    
    if '.' in v:
        return float(v)
    else:
        return int(v)

def make_module(module_name, source_code):
    module_spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    exec(source_code, module.__dict__)
    return module

@dataclass
class ScopedVars:
    pass