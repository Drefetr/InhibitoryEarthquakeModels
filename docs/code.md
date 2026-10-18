# Code guide

The engine tree preserves historical paths and exact numerical Python bytes because modules verify dependency hashes. `provenance/extraction_manifest.json` maps originals to public files. Additional source files named by the accepted observed protocol are retained for its source-binding checks; their historical campaign workflows are not all standalone without the unshipped states.

```python
import sys
sys.path.insert(0, "engine")
from api import load
KernelSpec = load("kernels").KernelSpec
kernel = KernelSpec("ml", 1.0, alpha=0.5)
weights, rates = kernel.mixture
observed = load("observed")  # PredictiveHistory, evaluate_window, fit_training
oracle = load("oracle")      # KnownLaw, evaluate_oracle
```

Loader modules `simulator`, `paired`, `stable`, `restricted`, `fitting`, `staged`, `profile`, `level_only` and `heterogeneity` expose the other implementations. Importing these modules does not launch a scientific campaign. CUDA operations require a separate appropriate PyTorch installation and device; they were not tested during public preparation. Some preserved modules set numerical thread environment variables or disable bytecode writing, matching their original behaviour.

The physical loading construction and independent-background diagnostic family are distinct. The precise clock retains paired timestamps and positive event gaps: do not jitter projected ties. Mark transforms, productivity conventions, left limits, full quiet exposure and the terminal no-event interval matter. Read the methods supplement before adapting APIs.

Use public scripts for supported workflows. Direct historical `__main__` entry points can need unshipped states. The observed wrapper checks source and external-data hashes before fitting. Implementation availability is distinct from possessing every historical study input.
