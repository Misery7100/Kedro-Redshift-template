from kedro.pipeline import node, pipeline
from .nodes import *

# ------------------------- #

def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                event_based_ranking,
                dict(
                    app_events_ranking="app_events_ranking",
                    config="params:ranking"
                ),
                "event_based_ranks",
                name="event_based_ranking"
            )
        ]
    )
