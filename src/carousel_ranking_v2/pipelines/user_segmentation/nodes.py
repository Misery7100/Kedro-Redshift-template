from typing import Any, Dict, List, Tuple
import logging
import tensorflow as tf
from mlflow.keras import log_model
from mlflow.models.signature import infer_signature

from carousel_ranking_v2.extras.df import vaex as vx
from carousel_ranking_v2.extras.utils.typing import *
from carousel_ranking_v2.extras.ml.keras.models import DeepClustering

log = logging.getLogger(__name__)

# ------------------------- #

def deep_clustering_segmentation(

        app_events_segmentation: VaexDataFrame,
        anon_ids_for_segmentation: List[str],
        config: Config

    ) -> Tuple[Dict[str, str], Any]:

    data = app_events_segmentation.copy()

    data = data.to_pandas_df() # only when data isn't heavy

    model = DeepClustering(
                data.shape, 
                config['n_clusters']
            )

    optimizer = tf.keras.optimizers.SGD(**config['optimizer'])
    loss = tf.keras.losses.KLDivergence()

    model.compile(optimizer=optimizer, loss=loss)
    model.fit(data, **config['fit'])

    pred = model.predict(data)
    segm = dict( zip(anon_ids_for_segmentation, pred.argmax(1).astype(str)) )

    # mlflow signature
    sign = infer_signature(data, pred)
    log_model(
        model, 
        'deepclustering', 
        signature=sign, 
        registered_model_name='deepclustering'
    )

    return segm

# ------------------------- #