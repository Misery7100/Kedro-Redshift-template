from kedro.pipeline import node, pipeline
from .nodes import *

# ------------------------- #

def create_pipeline(**kwargs):
    return pipeline(
        [   
            node(
                extract_uid_mapping,
                dict(
                    app_events_merged="app_events_merged"
                ),
                "uid_aid_mapping",
                name="extract_uid-aid_mapping"
            ),
            node(
                preprocess_events_for_segmentation,
                dict(
                    app_events_merged="app_events_merged",
                    config="segmentation_config",
                    event_weights="event_weights"
                ),
                [
                    "app_events_segmentation",
                    "anon_ids_for_segmentation"
                ],
                name="preprocess_events_for_segmentation"
            ),
            node(
                preprocess_events_for_ranking,
                dict(
                    app_events_merged="app_events_merged",
                    aid_segm_mapping="aid_segm_mapping",
                    offer_eg="offer_eg",
                    config="ranking_config", 
                    dyn_params="dyn_params",
                    event_weights="event_weights"
                ),
                "app_events_ranking",
                name="preprocess_events_for_ranking"
            ),
        ]
    )
