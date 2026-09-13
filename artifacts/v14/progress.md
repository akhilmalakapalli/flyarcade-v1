# v1.4 development progress

| Task | Historical | Linear dev | MLP-PPO dev | GRU-PPO dev | Selected | Validation | Target | Ticks | Readout | Status |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| catch | 0.661 | 0.728 | 0.203 | 0.212 | linear (ticks) | 0.939 | 0.85 | 8 | descending | TARGET_MET |
| dodge | 0.926 | 0.942 | 0.808 | 0.781 | linear (primary) | 0.941 | 0.97 | 4 | descending | BELOW_TARGET |
| snake | 1.483 | 1.133 | 0.642 | 0.917 | linear (all_readout) | 12.157 | 4.0 | 8 | all | TARGET_MET |
| pong | 0.793 | 0.544 | 0.384 | 0.305 | linear (longer) | 0.940 | 0.9 | 8 | all | TARGET_MET |
| flappy | 0.605 | 0.003 | 0.017 | 0.196 | gru (longer) | 0.297 | 0.8 | 4 | descending | BELOW_TARGET |

Historical scores are reference-only and use different budgets/learners. All v1.4 primary arms share the same transitions and fixed substrate. Snake uses raw food; other scores are fractions. No confirmatory testing.
