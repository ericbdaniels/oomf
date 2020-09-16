from uuid import uuid4
import datetime
from typing import List, Tuple, Union

import omf
import rich
import pandas as pd

from . import gslib, utils


class Element(utils.Wrapper):
    def __init__(self, element):
        super().__init__(element)
        self.data_names = [d.name for d in self.data]
        self.size = len(self.data[0].array)

    def __repr__(self) -> str:
        return f"<OMFElement> (Name: {self.name}, Type: {self.subtype})"

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame({f"{d.name},{d.uid}": d.array[:] for d in self.data})

    def to_csv(self, filename: str, nan_value: float = -999.0) -> None:
        df = self.to_pandas()
        df.fillna(nan_value, inplace=True)
        df.to_csv(filename)

    def to_gslib(self, filename: str, nan_value: float = -999.0):
        df = self.to_pandas()
        df.fillna(nan_value, inplace=True)
        gslib.write(df, filename)


class PointSet(Element):
    def __init__(self):
        pass

    @classmethod
    def from_pandas(
        name: str,
        dataframe: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        origin: Tuple[float] = (0, 0, 0),
        description="",
        subtype="point",
        **kwargs,
    ):
        origin = kwargs.get("origin", "")
        vertices_array = omf.data.Vector3Array(
            array=dataframe[[x_col, y_col, z_col]].values
        )
        geometry = omf.PointSetGeometry(origin=origin, vertices=vertices_array)
        data = dataframe.drop([x_col, y_col, z_col]).values.tolist()
        pointset = omf.PointSetElement(
            name=name,
            geometry=geometry,
            data=data,
            description=description,
            subtype=subtype,
        )
        return Element(pointset)


class LineSet:
    pass


class Volume:
    pass


class Surface:
    pass


class Project(utils.Wrapper):
    def __init__(self, filepath: str):
        reader = omf.OMFReader(filepath)
        omf_prj = reader.get_project()
        super().__init__(omf_prj)
        self.elements = utils.dotdict({e.name: Element(e) for e in self.elements})

    def summarize(self):
        rich.inspect(self, title=f"OMFProject: {self.name}")

    def __repr__(self) -> str:
        return f"<OMF Project> (Name: {self.name}, Date Created: {self.date_created})"