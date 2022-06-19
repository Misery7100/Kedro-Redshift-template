import re
import vaex
import numpy as np
import datetime
import logging
from vaex.ml import OneHotEncoder, LabelEncoder, StandardScaler, MultiHotEncoder
from typing import Callable, List, Tuple, Any, Union
from ..utils.typing import *

log = logging.getLogger(__name__)

# ------------------------- #

vaex_agg_func = {

    'count' : vaex.agg.count,
    'mean' : vaex.agg.mean,
    'max' : vaex.agg.max,
    'sum' : vaex.agg.sum,
    'first' : vaex.agg.first,
    'std' : vaex.agg.std

}

# ------------------------- #

def drop(
    
        data: VaexDataFrame, 
        cols: List[str]
    
    ) -> VaexDataFrame:

    df = data.copy()
    cols = [ x for x in cols
            if x in list(df.get_column_names()) ]
    df = df.drop(cols)

    return df

# ------------------------- #

def fillna(
    
        data: VaexDataFrame, 
        cols: List[str],
        value: Any
    
    ) -> VaexDataFrame:

    df = data.copy()
    df = df.fillna(value=value, column_names=cols)
    df.materialize(cols, inplace=True)

    return df

# ------------------------- #

def force_int(

    data: VaexDataFrame, 
    cols: List[str]

    ) -> VaexDataFrame:

    df = data.copy()
    
    for col in cols:
        df[col] = df[col].astype('int64')
    
    df.materialize(cols, inplace=True)

    return df

# ------------------------- #

def datetime(
    
        data: VaexDataFrame, 
        cols: List[str],
        fillnaval: datetime.date,
        dtformat: str = r'\d{4}-\d{2}-\d{2}',
    
    ) -> VaexDataFrame:

    df = data.copy()

    df = fillna(df, cols=cols, value=fillnaval)

    for col in cols:
        df[col] = (df[col]
                    .apply(lambda x: re.search(dtformat, x).string)
                    .apply(np.datetime64)
                )
                
    df.materialize(cols, inplace=True)

    return df

# ------------------------- #

def days_difference(
    
        data: VaexDataFrame, 
        cols: List[str],
        basecol: str,
        dropbase: bool = True
    
    ) -> VaexDataFrame:

    df = data.copy()

    for col in cols:
        df[col] = (df[basecol] - df[col]).apply(lambda x: x.days)
    
    df.materialize(cols, inplace=True)

    if dropbase:
        df = drop(df, [basecol])

    return df

# ------------------------- #

def apply_function(
    
        data: VaexDataFrame, 
        cols: List[str]
    
    ) -> VaexDataFrame:

    return data

# ------------------------- #

def onehotenc(
    
        data: VaexDataFrame, 
        cols: List[str],
        enc: OneHotEncoder = None,
        materialize: bool = False,
        dropcols: bool = True
    
    ) -> Tuple[VaexDataFrame, OneHotEncoder]:

    df = data.copy()

    if enc is None:
        enc = OneHotEncoder(features=cols, prefix='')
        enc.fit(df)

    df = enc.transform(df)

    if dropcols:
        df = drop(df, cols)
    
    if materialize:
        df.materialize(list(df.virtual_columns.keys()), inplace=True)
    
    return df, enc

# ------------------------- #

def multihotenc(

        data: VaexDataFrame, 
        cols: List[str],
        enc: MultiHotEncoder = None,
        materialize: bool = False,
        dropcols: bool = True

    ) -> Tuple[VaexDataFrame, MultiHotEncoder]:

    df = data.copy()

    if enc is None:
        enc = MultiHotEncoder(features=cols, prefix='')
        enc.fit(df)
    
    df = enc.transform(df)

    if dropcols:
        df = drop(df, cols)
    
    if materialize:
        df.materialize(list(df.virtual_columns.keys()), inplace=True)
    
    return df, enc

# ------------------------- #

def scale(
    
        data: VaexDataFrame, 
        cols: List[str],
        exclude: List[str] = list(),
        scaler: StandardScaler = None # TODO: replace with custom
    
    ) -> Tuple[VaexDataFrame, StandardScaler]:

    df = data.copy()

    sc = [ x for x in df.get_column_names() 
           if not re.search(r'({})'.format('|'.join(exclude)), x)
           or re.search(r'({})'.format('|'.join(cols)), x) ]
    
    sc = sc or df.get_columns_names()

    if scaler is None:
        scaler = StandardScaler(features=sc, prefix='')
        scaler.fit(df)
    
    df = scaler.transform(df)
    df.materialize(sc, inplace=True)

    return df, scaler

# ------------------------- #

def filt(
    
        data: VaexDataFrame, 
        filt: str = None
    
    ) -> VaexDataFrame:

    df = data.copy()

    if not filt is None:
        df = df.filter(eval(filt))

    return df

# ------------------------- #

def constant(
    
        data: VaexDataFrame, 
        col: str,
        val: Any
    
    ) -> VaexDataFrame:

    df = data.copy()

    if col in df.get_column_names():
        raise ValueError(f'{col=} already exists')

    df[col] = np.full(df.shape[0], val)
    df.materialize([col], inplace=True)

    return df

# ------------------------- #

def categorical(
    
        data: VaexDataFrame, 
        cols: List[str] = tuple(),
        enc: LabelEncoder = None
    
    ) -> Tuple[VaexDataFrame, LabelEncoder]:

    df = data.copy()

    if enc is None:
        enc = LabelEncoder(features=cols, prefix='', allow_unseen=True)
        enc.fit(df)
    
    df = enc.transform(df)
    df.materialize(list(df.virtual_columns.keys()), inplace=True)

    return df, enc

# ------------------------- #

def groupby(
    
        data: VaexDataFrame,
        by: str, 
        groups: List[Tuple],
        fillnaval: Any = 0.
    
    ) -> VaexDataFrame:

    df = data.copy()

    agg = dict( (name, vaex_agg_func[f](col, selection=query)) if query 
                else (name, vaex_agg_func[f](col))
                for query, col, name, f in groups )
    cols = [n for _, _, n, _ in groups]

    df = df.groupby(by=by, agg=agg)
    df = fillna(df, cols=cols, value=fillnaval)
    df = df.sort(by)

    return df

# ------------------------- #