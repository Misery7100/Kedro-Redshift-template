from kedro.pipeline import node, pipeline
from .nodes import *

# ------------------------- #

def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                deep_clustering_segmentation,
                dict(
                    app_events_segmentation="app_events_segmentation",
                    anon_ids_for_segmentation="anon_ids_for_segmentation",
                    config="params:segmentation"
                ),
                "aid_segm_mapping",
                name="deep_clustering_segmentation"
            )
        ]
    )
