from typing import Any, Dict, List
import re
import logging

from carousel_ranking_v2.extras.utils.typing import *
from carousel_ranking_v2.extras.df import vaex as vx

log = logging.getLogger(__name__)

# ------------------------- #

def extract_cols_info(cols: Dict[str, Any]):

    column_names = list(cols.keys())
    dtypes = dict(
        (col, dty) if dty is None else (col, eval(dty))
        for col, dty in cols.items()
    )

    return column_names, dtypes

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