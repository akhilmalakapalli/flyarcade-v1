# v1.4 development and fresh validation

| Task | Historical | Linear dev | MLP-PPO dev | GRU-PPO dev | Selected | Validation | Target | Ticks | Readout | Status |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| catch | 0.661 | 0.728 | 0.203 | 0.212 | linear (ticks) | 0.939 | 0.85 | 8 | descending | TARGET_MET |
| dodge | 0.926 | 0.942 | 0.808 | 0.781 | linear (primary) | 0.941 | 0.97 | 4 | descending | BELOW_TARGET |
| snake | 1.483 | 1.133 | 0.642 | 0.917 | linear (all_readout) | 12.157 | 4.0 | 8 | all | TARGET_MET |
| pong | 0.793 | 0.544 | 0.384 | 0.305 | linear (longer) | 0.940 | 0.9 | 8 | all | TARGET_MET |
| flappy | 0.605 | 0.003 | 0.017 | 0.196 | gru (longer) | 0.297 | 0.8 | 4 | descending | BELOW_TARGET |

Uncertainty is descriptive across three trained seeds. Primary comparisons isolate readout architecture under one common PPO configuration; they do not rank fully optimized architectures. Secondary factors are conditional on development winners. Historical results are references, not matched controls. No repeated tuning against validation or new confirmatory testing occurred.

## Interpretation

Primary winners: catch=linear, dodge=linear, snake=linear, pong=linear, flappy=gru.

Meaningful numerical gains over historical references: catch, snake, pong. These differences are descriptive, not controlled historical contrasts.

GRU consistently best across tasks: False.

Rescue attempts: flappy.

## Conditional secondary effects

| Task | Factor | Compared with | Mean change | Same PPO budget | Eligible |
|---|---|---|---:|---|---|
| catch | ticks | catch-primary-linear | +0.225 | True | True |
| dodge | ticks | dodge-primary-linear | -0.024 | True | True |
| snake | ticks | snake-primary-linear | +2.183 | True | True |
| snake | all_readout | snake-ticks-linear | +8.258 | True | True |
| pong | ticks | pong-primary-linear | +0.105 | True | True |
| pong | all_readout | pong-ticks-linear | +0.218 | True | True |
| pong | longer | pong-all_readout-linear | +0.072 | False | True |
| flappy | ticks | flappy-primary-gru | -0.061 | True | False |
| flappy | all_readout | flappy-primary-gru | +0.060 | True | False |
| flappy | longer | flappy-primary-gru | +0.161 | False | True |
| flappy | curriculum | flappy-longer-gru | -0.069 | True | True |
| flappy | rescue | flappy-longer-gru | -0.058 | True | True |

These are predeclared conditional comparisons on development data. Rescue additionally uses teacher transitions; it is not a primary architecture arm.

Not established: no matched sensory-only comparator; limited optimization, encoding, integration and readout remain alternatives.

Architecture-dependent development evidence at a common PPO operating point; not confirmation of a universally best learner.
