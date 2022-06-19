from typing import Dict
from kedro.pipeline import Pipeline
from carousel_ranking_v2.pipelines import data_engineering as de
from carousel_ranking_v2.pipelines import data_preprocessing as dp
from carousel_ranking_v2.pipelines import user_segmentation as us
from carousel_ranking_v2.pipelines import event_ranking as er
from carousel_ranking_v2.pipelines import transaction_ranking as tr
from carousel_ranking_v2.pipelines import result_allocation as ra

# ------------------------- #

def register_pipelines() -> Dict[str, Pipeline]:

    dep = de.create_pipeline()
    delp = de.create_pipeline(local=True)

    dpp = dp.create_pipeline()
    usp = us.create_pipeline()
    erp = er.create_pipeline()
    trp = tr.create_pipeline()
    rap = ra.create_pipeline()

    rnp = erp + trp
    mlp = usp + rnp

    return {

        "data_engineering": dep,
        "data_engineering_local": delp,
        "data_preprocessing" : dpp,
        "user_segmentation" : usp,
        "event_ranking" : erp,
        "transaction_ranking" : trp,
        "result_allocation" : rap,
        "ranking" : rnp,
        "machine_learning" : mlp,
        "full_cycle": dep + dpp + mlp + rap,
        "eng_preproc": dep + dpp,
        "eng_preproc_local": delp + dpp,
        "__default__": dep + dpp
        
    }
