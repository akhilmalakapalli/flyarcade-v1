# v1.4 development progress

| Task | Historical | Linear dev | MLP-PPO dev | GRU-PPO dev | Selected | Validation | Target | Ticks | Readout | Status |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| catch | 0.661 | 0.728 | 0.203 | 0.212 | linear (ticks) | pending | 0.85 | 8 | descending | IN_PROGRESS |
| dodge | 0.926 | 0.942 | 0.808 | 0.781 | linear (primary) | pending | 0.97 | 4 | descending | IN_PROGRESS |
| snake | 1.483 | 1.133 | 0.642 | 0.917 | linear (all_readout) | pending | 4.0 | 8 | all | IN_PROGRESS |
| pong | 0.793 | 0.544 | 0.384 | 0.305 | linear (longer) | pending | 0.9 | 8 | all | IN_PROGRESS |
| flappy | 0.605 | 0.003 | 0.017 | 0.196 | gru (longer) | pending | 0.8 | 4 | descending | IN_PROGRESS |

Historical scores are reference-only and use different budgets/learners. All v1.4 primary arms share the same transitions and fixed substrate. Snake uses raw food; other scores are fractions. No confirmatory testing.
