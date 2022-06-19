from typing import Any, Dict, List
import gc
import logging
import vaex
from vaex.ml import LabelEncoder
from kedro.extras.decorators.memory_profiler import mem_profile

from carousel_ranking_v2.extras.df import vaex as vx
from carousel_ranking_v2.extras.utils.typing import *
from .subnodes import *

log = logging.getLogger(__name__)

# ------------------------- #

@mem_profile
def extract_uid_mapping(app_events_merged: VaexDataFrame) -> VaexDataFrame:

    uid = 'user_id'
    aid = 'anonymous_user_id'
    ids = [uid, aid]

    aem = app_events_merged.copy()

    aem = aem[ids]
    aem = aem[aem[uid].notna() & aem[aid].notna()]
    aem = aem.extract()

    enc = LabelEncoder(
            features=[uid, aid],
            prefix=''
        )
    enc.fit(aem)

    aem = enc.transform(aem)
    aem = aem.groupby(by=aid, agg={uid : vaex.agg.first(uid, aid)})

    uid_dict = dict( (j, i) for i, j in enc.labels_[uid].items() )
    aid_dict = dict( (j, i) for i, j in enc.labels_[aid].items() )

    aem[aid] = aem[aid].apply(lambda x: aid_dict.get(x))
    aem[uid] = aem[uid].apply(lambda x: uid_dict.get(x))

    aem = aem.to_pandas_df()
    aem.set_index(aid, inplace=True)
    mapping = aem.to_dict()['user_id']

    del aem
    gc.collect()

    return mapping

# ------------------------- #

@mem_profile
def preprocess_events_for_segmentation(

        app_events_merged: VaexDataFrame,
        config: Config,
        event_weights: Config

    ) -> VaexDataFrame:

    aem = app_events_merged.copy()

    aem = add_event_weights(aem, event_weights)
    #aem, _ = vx.multihotenc(aem, preproc_conf['multihotenc'], materialize=True)
    aem, _ = vx.onehotenc(aem, config['onehotenc'], materialize=True)
    #grp = vx.groupby(aem, 'anonymous_user_id', preproc_conf['groupby'])
    grp2 = get_activity_characteristics(
            aem, 
            period=config['period'],
            histograms=config['histograms']
        )

    # aem = preprocess_merchants_segmentation(
    #         aem, 
    #         drop=preproc_conf['drop_2']
    #     )

    # aem = vx.drop(aem, preproc_conf['drop_2'])
    # columns = [ x for x in aem.get_column_names()
    #             if not (x.startswith('__') or x == 'anonymous_user_id') ]
    # groupby = [['', col, col, 'count'] for col in columns]

    # aem = vx.groupby(aem, 'anonymous_user_id', groupby)

    # grp2 = grp2.join(
    #         grp, 
    #         right_on='anonymous_user_id', 
    #         left_on='anonymous_user_id', 
    #         how='right'
    #     )

    #log.critical(grp2)

    uids = grp2.unique(grp2.anonymous_user_id)

    grp2 = vx.drop(grp2, ['anonymous_user_id'])
    grp2, _ = vx.scale(grp2, grp2.get_column_names())
    
    return grp2, uids

# ------------------------- #

@mem_profile
def preprocess_events_for_ranking(

        app_events_merged: VaexDataFrame,
        aid_segm_mapping: Dict[str, str],
        offer_eg: PandasDataFrame,
        config: Config,
        dyn_params: Config,
        event_weights: Config

    ) -> VaexDataFrame:

    aem = app_events_merged.copy()
    config = config['events']

    aem = filter_suitable_offers(aem, offer_eg, dyn_params['refdate'])
    aem = add_segments(aem, aid_segm_mapping)
    aem = preprocess_event_type_ranking(
            data=aem, 
            rescale=config['rescale'], 
            weights=event_weights
        )
    aem = aem[config['leave_columns']]

    splitted_data = dict()

    for segment in aem.unique(aem.segment):

        data = aem[aem.segment == segment]
        data, _ = vx.onehotenc(data, config['onehotenc'])
        offer_cols = [ c for c in data.get_column_names() 
                    if c.startswith('offer_id') ]

        for c in offer_cols:
            data[c] *= data['event_type']
        
        data = vx.drop(data, config['drop'])
        splitted_data[segment] = data

    return splitted_data

# ------------------------- #

@mem_profile
def preprocess_transactions_for_ranking(

        transactions: VaexDataFrame,

    ) -> VaexDataFrame:

    tr = transactions.copy()

    return tr

# ------------------------- #
