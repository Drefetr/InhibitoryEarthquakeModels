"""Load preserved scientific modules without running campaign entry points."""
from pathlib import Path
import importlib.util
import sys
ROOT=Path(__file__).resolve().parent
MODULES={'kernels':'iteration_5/code/kernels.py','empirical':'iteration_5/code/empirical_process.py','simulator':'iteration_12/code/clock_simulator.py','paired':'iteration_12/code/paired_history.py','stable':'iteration_12/code/stable_history.py','oracle':'iteration_17/code/oracle_likelihood.py','cuda':'iteration_18/code/cuda_oracle.py','restricted':'iteration_20/code/restricted_history.py','fitting':'iteration_20/code/restricted_fit.py','staged':'iteration_21/code/staged_fit.py','profile':'iteration_24/code/profile_optimizer.py','level_only':'iteration_26/code/level_only_fit.py','observed':'iteration_27/code/adequacy.py','heterogeneity':'iteration_28/code/heterogeneity.py'}
def load(name):
 path=ROOT/MODULES[name];sys.path.insert(0,str(path.parent));module_name=path.stem
 if module_name in sys.modules:
  module=sys.modules[module_name]
  if Path(module.__file__).resolve()!=path.resolve():raise RuntimeError('Conflicting module: '+module_name)
  return module
 spec=importlib.util.spec_from_file_location(module_name,path)
 module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module
 try:spec.loader.exec_module(module)
 except BaseException:
  sys.modules.pop(module_name,None);raise
 return module
