"""Fresh v1.5 seed namespace, disjoint from every historical and v1.4 seed.

Historical/dashboard seeds are < 1.2e10 and v1.4 seeds lie in [1e11, 6e11). v1.5 seeds
lie in [1e12, 1.5e12). Each (task, purpose) owns a 5e9 block; a trained seed owns a
1e7 slot. Neural (+1e9) and behaviour (+2e9) RNG offsets stay inside the purpose
block and never overlap raw episode seeds (which stay below 1e8 within the block).
Purposes 15-17 are reserved for future confirmatory testing and are rejected here.
"""

TASK_ORDER = ("catch", "dodge", "snake", "pong", "flappy")
SEED_BASE = 1_000_000_000_000
TASK_BLOCK = 100_000_000_000
PURPOSE_BLOCK = 5_000_000_000
SEED_SLOT = 10_000_000
MAX_SEEDS = 10
PURPOSES = (
    "software",  # 0 unit tests / replay checks
    "diagnosis",  # 1 phase-1 representation and failure diagnostics
    "feature_fit",  # 2 standardizer statistics
    "train",  # 3 training episodes
    "model",  # 4 parameter initialisation
    "rollout",  # 5 action sampling / minibatch order
    "curve",  # 6 learning-curve episodes
    "checkpoint_select",  # 7 within-run checkpoint choice
    "dev_eval",  # 8 development score of the chosen checkpoint
    "dev_probe",  # 9 state-dependence probe (development)
    "imitation",  # 10 rescue warm-start transitions
    "validation",  # 11 fresh validation (locked selections only)
    "validation_probe",  # 12 validation state probe
    "diagnosis_training",  # 13 phase-1 diagnostic training runs
    "spare",  # 14 unused
)
CONFIRMATORY_RESERVED = (15, 16, 17)
NEURAL_OFFSET = 1_000_000_000
BEHAVIOUR_OFFSET = 2_000_000_000


def seed_for(task, purpose, seed=0, index=0):
    if (
        task not in TASK_ORDER
        or purpose not in PURPOSES
        or type(seed) is not int
        or type(index) is not int
        or not 0 <= seed < MAX_SEEDS
        or not 0 <= index < SEED_SLOT
    ):
        raise ValueError("invalid v1.5 seed request; confirmatory purposes are forbidden")
    return (
        SEED_BASE
        + TASK_ORDER.index(task) * TASK_BLOCK
        + PURPOSES.index(purpose) * PURPOSE_BLOCK
        + seed * SEED_SLOT
        + index
    )


def purpose_interval(task, purpose):
    start = seed_for(task, purpose)
    return start, start + PURPOSE_BLOCK - 1


def confirmatory_interval(task, k):
    if k not in CONFIRMATORY_RESERVED:
        raise ValueError("not a reserved confirmatory purpose index")
    start = SEED_BASE + TASK_ORDER.index(task) * TASK_BLOCK + k * PURPOSE_BLOCK
    return start, start + PURPOSE_BLOCK - 1
