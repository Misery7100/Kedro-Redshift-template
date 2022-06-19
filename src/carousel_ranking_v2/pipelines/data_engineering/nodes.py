from typing import Any, Dict, List, Tuple
import sqlalchemy as db
from sqlalchemy.sql.expression import func
import numpy as np
import os
import pytz
import gc
import logging
import datetime
import vaex

from carousel_ranking_v2.extras import io, vx
from carousel_ranking_v2.extras.utils.typing import *

from .subnodes import *

log = logging.getLogger(__name__)

# ------------------------- #

def init_dynamic_parameters() -> Dict[str, Any]:
    
    """Initialize dynamic parameters (e.g. refresh time, unknown date)

    :return: dynamic parameters
    :rtype: Dict[str, Any]
    """

    refresh = ( datetime.datetime.now(pytz.utc) + \
                datetime.timedelta(hours=5) ).date().strftime('%Y-%m-%d')

    unkdate = datetime.datetime.now(pytz.utc).strftime('%Y-%m-%d')

    params = dict(
        refdate=refresh,
        unkdate=unkdate
    )
    
    return params

# ------------------------- #

def merge_app_events(
    
        app_events: RedshiftTableConn, 
        app_events_extended: RedshiftTableConn,
        config: Config,
        dyn_params: Config
    
    ) -> vaex.dataframe.DataFrame:

    """Pull, merge and preprocess app events from two tables using SQLAlchemy

    :param app_events: SQLAlchemy table with connection
    :type app_events: RedshiftTableConn
    :param app_events_extended: SQLAlchemy table with connection
    :type app_events_extended: RedshiftTableConn
    :param config: pulling configuration with additional parameters
    :type config: Config
    :param dyn_params: previously initialized dynamic parameters
    :type dyn_params: Config
    :return: merged and preprocessed data
    :rtype: vaex.dataframe.DataFrame
    """
    
    ae = app_events.table
    aex = app_events_extended.table
    conn = app_events.conn

    fetch_conf = config['events']
    preproc_conf = config['events_preprocessing']

    app_events_extended.conn.close()

    ae_cols = app_events.columns
    aex_cols = app_events_extended.columns

    ae_column_names, ae_dtypes = extract_cols_info(ae_cols)
    aex_column_names, aex_dtypes = extract_cols_info(aex_cols)
    column_names = ae_column_names + aex_column_names
    dtypes = {**ae_dtypes, **aex_dtypes}

    columns = [aex.columns[k] for k in aex_column_names] + \
              [ae.columns[k] for k in ae_column_names]
                 

    query = (db
        .select([aex])
        .with_only_columns(columns)
        .where(aex.columns.event_timestamp >= datetime.datetime.now() - datetime.timedelta(days=fetch_conf['period']))
        #.where(func.length(ae.columns.anonymous_user_id) > 20)
        .where(aex.columns.offer_id != None)
        .join(ae, (aex.columns.event_id == ae.columns.event_id) and (aex.columns.event_timestamp == ae.columns.event_timestamp))
        .limit(fetch_conf['limit'])
    )

    log.info(f'query execution started')
    res = conn.execution_options(stream_results=True).execute(query)
    log.info(f'query execution finished')

    df = io.fetch_chunkwise(
        res,
        column_names,
        chunksize=fetch_conf['chunksize'],
        dtypes=dtypes,
        droplist=['event_id']
    )

    del res
    gc.collect()

    # preprocessing

    df = vx.constant(df, 'refresh_date', dyn_params['refdate'])

    log.critical(df.dtypes)
    log.critical(df)

    # for value, cols in preproc_conf['fillna'].items():
    #     df = vx.fillna(df, cols, value)
    
    # df = preprocess_anon_uid(df)
    # df = preprocess_channels(df)
    # df = vx.force_int(df, preproc_conf['force_int'])
    # df = vx.datetime(df, preproc_conf['datetime'], dyn_params['unkdate'])
    # df = vx.days_difference(df, preproc_conf['days_difference'], 'refresh_date')

    return df

# ------------------------- #

def pull_transactions(
    
        transactions: RedshiftTableConn,
        config: Config,
        dyn_params: Config
    
    ) -> vaex.dataframe.DataFrame:

    """Pull and preprocess transactions using SQLAlchemy

    :param transactions: SQLAlchemy table with connection
    :type transactions: RedshiftTableConn
    :param config: pulling configuration with additional parameters
    :type config: Config
    :param dyn_params: previously initialized dynamic parameters
    :type dyn_params: Config
    :return: pulled and preprocessed data
    :rtype: vaex.dataframe.DataFrame
    """
    
    tr = transactions.table
    conn = transactions.conn
    cols = transactions.columns
    fetch_conf = config['transactions']
    preproc_conf = config['transactions_preprocessing']

    column_names, dtypes = extract_cols_info(cols)
    columns = [tr.columns[k] for k in column_names]

    query = (db
            .select([tr])
            .with_only_columns(columns)
            .where(tr.columns.createddatetime >= datetime.datetime.now() - datetime.timedelta(days=fetch_conf['period']))
            .where(tr.columns.luckyuserid != None)
            .limit(fetch_conf['limit'])
        )
    
    log.info(f'query execution started')
    res = conn.execution_options(stream_results=True).execute(query)
    log.info(f'query execution finished')
    
    df = io.fetch_chunkwise(
        res,
        column_names,
        chunksize=fetch_conf['chunksize'],
        dtypes=dtypes
    )

    del res
    gc.collect()

    df = vx.constant(df, 'refresh_date', dyn_params['refdate'])

    for value, cols in preproc_conf['fillna'].items():
        df = vx.fillna(df, cols, value)
    
    df = vx.datetime(df, preproc_conf['datetime'], dyn_params['unkdate'])
    df = vx.days_difference(df, preproc_conf['days_difference'], 'refresh_date')

    return df

# ------------------------- #