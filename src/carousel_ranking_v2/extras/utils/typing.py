from typing import Dict, List, Any, Callable
from vaex.dataframe import DataFrame as vdf
from pandas import DataFrame as pdf
from ..datasets.sqlalchemy import TableWithConn

# ------------------------- #

VaexDataFrame = vdf
PandasDataFrame = pdf
RedshiftTableConn = TableWithConn
Config = Dict[str, Any]