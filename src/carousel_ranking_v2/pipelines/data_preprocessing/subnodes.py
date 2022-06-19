from typing import Any, Dict, List, Tuple
from copy import copy
import pandas as pd
from scipy.stats import norm
import re
import logging

from carousel_ranking_v2.extras.utils.typing import *
from carousel_ranking_v2.extras.df import vaex as vx

log = logging.getLogger(__name__)

# ------------------------- #

def preprocess_channels(data: VaexDataFrame) -> VaexDataFrame:

    df = data.copy()

    df['channels'] = df.channels.apply(lambda x: str(set(map(

                lambda y: ''.join([s for s in y if s.isalpha() or s == '_']), 
                x.split(','))

            )))
    
    df.materialize(['channels'], inplace=True)

    for k in ['in_store', 'online', 'delivery']:
        df[f'{k}_event'] = df.channels.apply(lambda x: 1 if re.search(k, x) else 0)
        df.materialize([f'{k}_event'], inplace=True)
    
    df = vx.drop(df, ['channels'])

    return df

# ------------------------- #

def preprocess_anon_uid(data: VaexDataFrame) -> VaexDataFrame:

    df = data.copy()

    df['anonymous_user_id'] = df.anonymous_user_id.apply(lambda x: 

                f'{x[:8]}-{x[8:12]}-{x[12:16]}-{x[16:20]}-{x[20:]}'
                if len(x) < 36 else x
                
            )
    df.materialize(['anonymous_user_id'], inplace=True)

    return df

# ------------------------- #

def __build_histogram(

        tag: str = 'activity',
        subcondition: str = None,
        period: int = 35

    ) -> List[Tuple[str]]:

    histogram = [ (f'event_timestamp == {day}', 
                  'event_weight', f'activity_{day}', 'sum') 
                  if subcondition is None else
                  (f'(event_timestamp == {day}) & ({subcondition})', 
                  'event_weight', f'{tag}_{day}', 'sum')
                  for day in range(1, period) ]
    
    return histogram

# ------------------------- #

def add_event_weights(
    
        data: VaexDataFrame,
        config: Config
    
    ) -> VaexDataFrame:

    df = data.copy()

    df['event_weight'] = df.event_type.apply(lambda x: config.get(x, 1.))
    df.materialize(['event_weight'], inplace=True)

    return df

# ------------------------- #

def get_activity_characteristics(
    
        data: VaexDataFrame,
        period: int = 35,
        histograms: List[str] = list()
    
    ) -> VaexDataFrame:

    df = data.copy()

    groups = [
        (None, 'event_timestamp', 'activity_std', 'std'),
        (None, 'event_timestamp', 'activity_mean', 'mean')
    ]

    # build event histogram
    groups += __build_histogram(period=period)
    
    for h in histograms:
        groups += __build_histogram(
                period=period, 
                subcondition=h,
                tag=f'hist_{"_".join(h.split())}'
            )

    df = vx.groupby(df, by='anonymous_user_id', groups=groups)

    return df

# ------------------------- #

def get_specific_interests(data: VaexDataFrame) -> VaexDataFrame:

    pass

# ------------------------- #

def __weight_event_types_ranking(

        data: VaexDataFrame,
        weights: Config

    ) -> VaexDataFrame:

    df = data.copy()

    df['event_type'] = df.event_type.apply(lambda x: weights.get(x, 0.))
    df.materialize(['event_type'], inplace=True)

    return df

# ------------------------- #

def __get_offer_set(

        data: PandasDataFrame,
        refresh: str

    ) -> list:

    of = data.copy()

    of['enddate'] = of.enddate.apply(lambda x: (x - pd.to_datetime(refresh)).total_seconds())
    of = of.query(
            '(offercappingtypeid != 3) & '\
            '(offerid.notnull().values) & '\
            '(offerid != -1) & '\
            '(enddate > 0)'                         
        ).copy()
    
    offer_set = of.offerid.to_list()

    return offer_set

# ------------------------- #

def filter_suitable_offers(

        data: VaexDataFrame,
        offer_eg: PandasDataFrame,
        refresh: str

    ) -> VaexDataFrame:

    df = data.copy()
    
    offer_set = __get_offer_set(offer_eg, refresh)
    query = 'df.offer_id == ' + ' | df.offer_id == '.join(map(str, offer_set))
    df = df.filter(eval(query))

    return df

# ------------------------- #

def __rescale_events_by_timestamp(

        data: VaexDataFrame,
        config: Config

    ) -> VaexDataFrame:

    df = data.copy()

    # force numerical representataion 
    # for config values
    for k, v in config.items():
        config[k] = float(v)

    df['event_type'] = df.event_type / (1.0001 + \
        df.event_timestamp.apply(lambda x: x ** config['impact'] if x > config['shift'] else 0))

    df.materialize(['event_type'], inplace=True)

    return df

# ------------------------- #

def preprocess_event_type_ranking(

        data: VaexDataFrame,
        rescale: Config,
        weights: Config

    ) -> VaexDataFrame:

    df = data.copy()

    df = __weight_event_types_ranking(df, weights)
    df = __rescale_events_by_timestamp(df, rescale)

    return df

# ------------------------- #

def add_segments(

        data: VaexDataFrame,
        aid_segm_mapping: Dict[str, str]
        
    ) -> VaexDataFrame:

    df = data.copy()

    df['segment'] = df.anonymous_user_id.apply(lambda x: aid_segm_mapping.get(x, 'null'))
    df.materialize(['segment'], inplace=True)

    return df

# ------------------------- #