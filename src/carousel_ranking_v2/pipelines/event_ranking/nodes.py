from typing import Any, Dict, Tuple
import logging
import tensorflow as tf

from carousel_ranking_v2.extras.ml.keras.models import DeepAutoRec
from carousel_ranking_v2.extras.ml.keras.losses import MMSE
from carousel_ranking_v2.extras.ml.keras.callbacks import KerasTqdmBar
from carousel_ranking_v2.extras.utils.typing import *

log = logging.getLogger(__name__)

# ------------------------- #

def event_based_ranking(

        app_events_ranking: Dict[str, VaexDataFrame],
        config: Config,
        clip: Tuple[float] = (0, 37)

    ) -> Dict[str, Dict[str, float]]:

    segmentwise_ranks = dict()

    for segment in app_events_ranking.keys():

        data = app_events_ranking.get(segment) 

        #? loaded as bound method for unexpected reason
        data = data().to_pandas_df()
        data = (data
                    .groupby('anonymous_user_id')
                    .sum()
                    .reset_index(drop=True)
                )
        data = data.clip(*clip)
        data = data[data.max(axis=1) != 0]

        model = DeepAutoRec(data_shape=data.shape)
        optimizer = tf.keras.optimizers.Adam(**config['optimizer'])
        loss = MMSE()

        model.compile(optimizer=optimizer, loss=loss)
        model.fit(
            data, data, 
            verbose=0, 
            callbacks=[KerasTqdmBar()],
            **config['fit']
        )

        ranked = model.predict(data).mean(axis=0)
        result = dict(sorted(
                dict(zip(data.columns, ranked)).items(),
                key=lambda x: x[1],
                reverse=True
            ))

        segmentwise_ranks[segment] = result
    
    return segmentwise_ranks

# ------------------------- #