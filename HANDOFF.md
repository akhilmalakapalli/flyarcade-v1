# v1.4 handoff — in progress

Read experiments/v14/development_plan.json first. Its commit ee7d012 precedes
training. Branch is flyarcade-v1.4-performance from dashboard b444631.
All v1.1/v1.2/v1.3 artifacts, dashboard files and spent confirmatory seeds are
protected; no historical rerun/rescore is allowed. README.md stays unchanged.

Run with one numerical-library thread. scripts/v14_suite.py runs all primary
learners first, then bounded secondary factors and eligible imitation rescue,
and locks selected_configs.json before validation. Each trial exits 7 at safe
checkpoint boundaries and is resumed by the suite. Do not overwrite completed
results or retry a terminal failure outside the declared attempt budget.

Outstanding: primary grid, secondary factors, locked fresh validation, aggregation,
scientific audit and final documentation. No confirmatory execution is permitted.
Artifacts live only under artifacts/v14, experiments/v14 and ignored runs/v14.
