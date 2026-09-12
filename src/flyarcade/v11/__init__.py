"""v1.1 credit-assignment rebuild: actor-critic e-prop over the frozen v1 topology.

The v1 study (commit ec27ad0) showed that task information is decodable from the
MaleCNS-derived circuit while global reward-modulated STDP collapsed the policy
onto a constant action. This package replaces the learning rule only. The
connectome, its body IDs, the selected 2,040-neuron subgraph, the LIF dynamics and
the sensory encoding are unchanged from v1.
"""

from flyarcade.v11.actor_critic import ActorCritic
from flyarcade.v11.controller import STAGES, V11Controller
from flyarcade.v11.core import EpropCore
from flyarcade.v11.features import Standardizer

__all__ = ["ActorCritic", "EpropCore", "Standardizer", "V11Controller", "STAGES"]
