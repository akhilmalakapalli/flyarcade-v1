# Headless lane arcade tasks

Run `scripts/run_suite.py` from the repository root for the frozen experiment.
Implementation lives in `src/flyarcade/games.py` so installed-package imports work.

Catch and dodge each have five horizontal lanes, a player that begins in the
middle, and a falling object whose lane is drawn uniformly with an independent
environment RNG. One episode is 24 actions: six objects, each landing after four
actions. Actions 0/1/2 move left/stay/right; boundaries clamp movement. Catch earns
a success by occupying the object's lane at landing; dodge by avoiding it. New
object lanes are drawn independently of policy RNG and actions. Observations are
normalized player lane, object lane and falling phase. There is no hidden reward
or future random-state input to the controller.

Training reward is +1/-1 at landing, plus 0.1 times the change in absolute lane
distance (toward the object for catch, away for dodge). This dense shaping makes
the task easier and is **not** the evaluation metric. Evaluation reports the
fraction of six landings that succeed. The random baseline samples all three
actions equally; the deterministic heuristic follows the visible object for
catch and moves off its lane for dodge. Both see the same observations.

These are intentionally minimal arcade control tasks, not a fly behavior model
or a graphical game product. Structured feature channels and neuron stimulation
are artificial interfaces. The controller receives no task identity bit; separate
training or transfer must adapt the same action space to each reward rule.
