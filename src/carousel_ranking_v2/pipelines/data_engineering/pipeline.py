from kedro.pipeline import node, pipeline
from .nodes import *

# ------------------------- #

def create_pipeline(local: bool = False, **kwargs):

    if local:
        return pipeline(
            [   
                node(
                    init_dynamic_parameters,
                    None,
                    "dyn_params",
                    name="init_dynamic_parameters"
                )
            ]
        )

    else:
        return pipeline(
            [   
                node(
                    init_dynamic_parameters,
                    None,
                    "dyn_params",
                    name="init_dynamic_parameters"
                ),
                node(
                    merge_app_events,
                    dict(
                        app_events="app_events", 
                        app_events_extended="app_events_extended",
                        config="pulling_config",
                        dyn_params="dyn_params"
                    ),
                    "app_events_merged",
                    name="merge_app_events"
                ),
                # node(
                #     pull_transactions,
                #     dict(
                #         transactions="transaction_eg", 
                #         config="pulling_config",
                #         dyn_params="dyn_params"
                #     ),
                #     "transactions",
                #     name="pull_transactions"
                # )
            ]
        )
    