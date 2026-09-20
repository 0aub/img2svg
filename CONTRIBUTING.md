# Contributing

`make test` runs the suite inside the Docker image; `make example` regenerates
`examples/out`.

Two things worth knowing before changing the pipeline.

**Every stage has a reason, written down.** The module docstrings explain *why*
each non-obvious step exists, usually by naming the failure it prevents. If you
change one, update the reason. If you cannot state the reason, the step is
probably wrong.

**Fidelity is not the objective function.** Mean ΔE is a regression guard. Several
deliberate behaviours — container regularisation, gradient flattening — make it
worse on purpose. A change that improves the score is not automatically an
improvement; look at the output.

New heuristics should be stated rules with an escape hatch, not silent magic.
`--container` is the pattern: a documented threshold, a printed note saying what
it decided, and a flag to turn it off.
