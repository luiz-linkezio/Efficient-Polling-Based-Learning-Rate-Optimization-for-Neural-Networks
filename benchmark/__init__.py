"""The experiment behind the paper, shared by the notebook and the command line.

Not part of the published package, which is ``src/efficient_polling_lr_scheduler``:
this is what compares its methods against the others.

- ``datasets``: the five datasets, read from the files their authors publish.
- ``models``: the CNN and the MLP every method trains.
- ``methods``: the thirteen configurations and their hyperparameters.
- ``sweep``: runs them over seeds, records each run, reads the records back.
- ``plots``: the figures, drawn from those records.

``notebooks/benchmark.ipynb`` and ``python -m benchmark`` both drive this code,
so they cannot disagree about what a run is.
"""

from .methods import LABELS, METHODS, Hyperparameters
from .sweep import Experiment, load_runs, results_table

__all__ = ["LABELS", "METHODS", "Experiment", "Hyperparameters", "load_runs", "results_table"]
