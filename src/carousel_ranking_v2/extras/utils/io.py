import re
import os
import gc
import glob
from xxlimited import Str
import vaex
import logging
import pandas as pd
import shutil
import numpy as np
import sqlalchemy as db
from typing import Any, Dict, List
from .typing import *
from sqlalchemy.engine.cursor import LegacyCursorResult

log = logging.getLogger(__name__)
_TEMPDIR = '.temp'

# ------------------------- #

def safe_dir(path: str):

    path = os.path.abspath(path)

    if not os.path.exists(path):
        os.makedirs(path)

# ------------------------- #

def safe_rmtree(_dir: str):

    _dir = os.path.abspath(_dir)

    if os.path.exists(_dir):
        shutil.rmtree(_dir)

# ------------------------- #

def safe_rm(path: str):

    path = os.path.abspath(path)

    if os.path.exists(path):
        os.remove(path)

# ------------------------- #

def build_chunk(
        
        chunk: PandasDataFrame,
        savepath: str,
        droplist: List[str] = list()
        
    ):

    dupl_cols = [ x for x in chunk.columns
                if re.search(r'.\d', x) ]

    droplist += dupl_cols
    droplist = [ x for x in droplist
                if x in chunk.columns ]
    
    chunk = chunk.drop(droplist, axis=1)
    vaex_df = vaex.from_pandas(chunk, copy_index=False)

    log.info('exporting chunk to hdf5 via vaex')

    vaex_df.export_hdf5(path=savepath, progress=False)

    del vaex_df
    gc.collect()

# ------------------------- #

def load_chunks(chunkdir: str) -> VaexDataFrame:
    
    # ......................... #

    def tryint(s):
        try:
            return int(s)

        except:
            return s

    def alphanum_key(s):
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]

    # ......................... #

    hdf5_list = glob.glob(f'{chunkdir}/*.hdf5')
    hdf5_list.sort(key=alphanum_key)
    hdf5_list = np.array(hdf5_list)

    return vaex.open_many(hdf5_list)

# ------------------------- #

def fetch_chunkwise(

        res: LegacyCursorResult,
        column_names: List[str],
        chunksize: int,
        dtypes: Dict[str, str] = dict(),
        droplist: List[str] = list()

    ) -> VaexDataFrame:

    savedir = 'data/02_intermediate'
    tempdir = os.path.join(savedir, _TEMPDIR)
    
    safe_dir(tempdir)

    for i, part in enumerate(res.partitions(chunksize)):

        log.info(f'fetching chunk #{i}')
        df = pd.DataFrame(data=part, columns=column_names)
        df = df.drop(columns=droplist)

        for col, dtype in dtypes.items():
            if dtype in [np.float32, np.float64, float]:
                df[col] = df[col].fillna(0.)
            
            if not dtype is None:
                df[col] = df[col].astype(dtype)

        build_chunk(df, os.path.join(tempdir, f'{i}.hdf5'))

        del df
        gc.collect()

    df = load_chunks(tempdir)

    return df

# ------------------------- #
