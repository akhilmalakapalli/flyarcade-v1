"""Shared Snake controller construction for the analysis and visualisation scripts."""

from flyarcade.v11.controller import V11Controller
from flyarcade.v11.snake import CHANNELS
from flyarcade.v11.snake import features as snake_features


def make_controller(graph, seed, standardizer, plan):
    return V11Controller(
        graph,
        seed,
        stage="readout",
        standardizer=standardizer,
        encoder=snake_features,
        channels=CHANNELS,
        observation_width=CHANNELS,
        actions=4,
        **plan["hyperparameters"],
    )


def build_from_trial(*args, **kwargs):
    return make_controller(*args, **kwargs)
