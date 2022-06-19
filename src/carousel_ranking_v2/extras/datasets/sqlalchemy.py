import sqlalchemy as db
import pandas as pd
from typing import Any, Dict, List
from sqlalchemy import text
from kedro.io import AbstractDataSet

# ------------------------- #

class TableWithConn:

    def __init__(
        
            self, 
            tb: db.Table, 
            conn: db.engine.base.Connection,
            columns: Dict[str, Any]
        
        ) -> None:

        self.table = tb
        self.conn = conn
        self.columns = columns

# ------------------------- #

class RedshiftDataSet(AbstractDataSet):

    def __init__(
        
            self, 
            host: str,
            credentials: Dict[str, Any] = None,
            table: str = None,
            columns: List[str] = None
        
        ):

        self._host = host
        st = table.split('.')

        if len(st) == 1:
            self._schema, self._table = table, None
        
        else:
            self._schema, self._table = st

        self._user = credentials['user']

        _password = credentials['password']

        self.redshift_connection(_password)
        self._metadata = db.MetaData()
        self._columns = columns
        self.filepath = f'{self._host}/{self._schema}/{self._table}'

    # ......................... #

    def _load(self) -> TableWithConn:

        tb = db.Table(

                self._table, 
                self._metadata, 
                autoload=True, 
                autoload_with=self._engine

            )

        return TableWithConn(
            tb=tb, 
            conn=self._connection, 
            columns=self._columns
        )
    
    # ......................... #

    def _save(self) -> None:
        pass
    
    # ......................... #

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self.filepath)
    
    # ......................... #

    def redshift_connection(self, pwd: str) -> None:
        
        self._engine = db.create_engine(
                f'redshift://{self._user}:{pwd}@{self._host}',
                connect_args={'options': f'-csearch_path={self._schema}'},
                pool_recycle=3600,
                pool_size=20,
                max_overflow=20,
                encoding='utf8',
                echo=True
            )

        self._connection = self._engine.connect()

# ------------------------- #

class RedshiftFullDataSet(RedshiftDataSet):

    def _load(self) -> pd.DataFrame:

        tb = db.Table(
                self._table, 
                self._metadata, 
                autoload=True, 
                autoload_with=self._engine
            )

        if self._columns:
            query = db.select([tb.columns[x] for x in self._columns])

        else:
            query = db.select([tb])

        data = self._connection.execute(query).fetchall()
        df = pd.DataFrame(data=data, columns=self._columns or tb.columns.keys())
        self._connection.close()

        return df

# ------------------------- #

class RedshiftSQLDataSet(RedshiftDataSet):

    def __init__(self, sql: str, **kwargs):

        super().__init__(**kwargs)

        self._sql = sql

    def _load(self) -> pd.DataFrame:

        data = self._connection.execute(text(self._sql)).fetchall()
        df = pd.DataFrame(data=data, columns=self._columns)
        self._connection.close()

        return df

# ------------------------- #