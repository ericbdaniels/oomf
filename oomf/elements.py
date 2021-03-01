from uuid import uuid4
import datetime
from typing import List, Tuple, Union
import pathlib

import omf
import omfvista
import rich
import pandas as pd
import numpy as np
import itertools 

from . import gslib, utils


class Element:
    def __init__(self, element):
        self._element = element
        self.name = element.name
        self.element_type = element.subtype
        self.data_names = [d.name for d in self._element.data]
        self.size = len(self._element.data[0].array) if len(self._element.data) else 0
        for d in self._element.data:
            setattr(self, d.name, d.array)

    def __repr__(self) -> str:
        return f"<OOMFElement> (Name: {self.name}, Type: {self.element_type})"

    def _calc_centroids(self):
        segments = self._element.geometry.segments.array[:]
        verts = self._element.geometry.vertices.array[:]
        start = segments[:, 0]
        end = segments[:, 1]
        x_centroids = np.mean([verts[:, 0][start], verts[:, 0][start]], axis=0)
        y_centroids = np.mean([verts[:, 1][start], verts[:, 1][start]], axis=0)
        z_centroids = np.mean([verts[:, 2][start], verts[:, 2][start]], axis=0)
        return x_centroids, y_centroids, z_centroids
    
    def _calc_cell_centroids(self):
        ox,oy,oz = self._element.geometry.origin
        cumulative_x = self._element.geometry.tensor_u[:].cumsum()
        xs = [ox+cumulative_x[i]-.5*cumulative_x[i] for i in range(cumulative_x.size)]
        cumulative_z = self._element.geometry.tensor_v[:].cumsum()
        zs = [oz+cumulative_z[i]-.5*cumulative_z[i] for i in range(cumulative_z.size)]
        cumulative_y = self._element.geometry.tensor_w[:].cumsum()
        ys = [oy+cumulative_y[i]-.5*cumulative_y[i] for i in range(cumulative_y.size)]
        coords = np.array(list(itertools.product(xs,ys,zs)))
        return coords[:,0], coords[:,1], coords[:,2]

    def to_pandas(self, coords=True, **kwargs) -> pd.DataFrame:
        data_dict = {f"{d.name}": d.array[:] for d in self._element.data}
        location_types = [d.location for d in self._element.data]
        if self.element_type != "volume" and coords:
            if all(l == "segments" for l in location_types) and len(location_types):
                x, y, z = self._calc_centroids()
                return pd.DataFrame({"x": x, "y": y, "z": z, **data_dict}, **kwargs)

            elif all(l == "cells" for l in location_types):
                coords = self._calc_cell_centroids()
                coords_dict = {"x":coords[0], "y":coords[1], "z":coords[2]}
                return pd.DataFrame({**coords_dict, **data_dict}, **kwargs)

            elif all(l == "vertices" for l in location_types):
                coords_dict = {
                    "x": self._element.geometry.vertices[:, 0],
                    "y": self._element.geometry.vertices[:, 1],
                    "z": self._element.geometry.vertices[:, 2],
                }
                return pd.DataFrame({**coords_dict, **data_dict}, **kwargs)
        else:
            return pd.DataFrame(data_dict, **kwargs)

    def to_csv(self, filename: str, nan_value: float = -999.0) -> None:
        df = self.to_pandas()
        df.fillna(nan_value, inplace=True)
        df.to_csv(filename)

    def to_gslib(self, filename: str, nan_value: float = -999.0):
        df = self.to_pandas()
        df.fillna(nan_value, inplace=True)
        gslib.write(df, filename)


class PointSet(Element):
    def __init__(self, element):
        super().__init__(element)
        self._as_vtk = omfvista.point_set_to_vtk(self._element)

    def to_vtk(self, filename: str, binary: bool = True):
        self._as_vtk.save(filename, binary=binary)

    @classmethod
    def _from_pandas(
        cls,
        name: str,
        dataframe: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        origin: Tuple[float] = (0, 0, 0),
        description="",
        subtype="volume",
    ):
        vertices_array = omf.data.Vector3Array(
            array=dataframe[[x_col, y_col, z_col]].values
        )
        geometry = omf.PointSetGeometry(origin=origin, vertices=vertices_array)
        data_df = dataframe.drop([x_col, y_col, z_col], axis=1)
        pointset = omf.PointSetElement(
            name=name,
            geometry=geometry,
            data=utils.scalardata_from_df(data_df),
            description=description,
            subtype=subtype,
        )
        return cls(pointset)


class LineSet(Element):
    def __init__(self, element):
        super().__init__(element)
        self._as_vtk = omfvista.line_set_to_vtk(self._element)

    def to_vtk(self, filename: str, binary: bool = True):
        self._as_vtk.save(filename, binary=binary)

    @classmethod
    def _from_pandas(
        cls,
        name: str,
        dataframe: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        origin: Tuple[float] = (0, 0, 0),
        description="",
        subtype="borehole",
    ):
        vertices_array = omf.data.Vector3Array(
            array=dataframe[[x_col, y_col, z_col]].values
        )
        geometry = omf.LineSetGeometry(origin=origin, vertices=vertices_array)
        data_df = dataframe.drop([x_col, y_col, z_col], axis=1)
        lineset = omf.LineSetElement(
            name=name,
            geometry=geometry,
            data=utils.scalardata_from_df(data_df),
            description=description,
            subtype=subtype,
        )
        return cls(lineset)


class Volume(Element):
    def __init__(self, element):
        super().__init__(element)
        self._as_vtk = omfvista.volume_to_vtk(self._element)

    def _get_gslib_idx(self):
        x_siz = self.geometry.tensor_u.size
        y_siz = self.geometry.tensor_v.size
        z_siz = self.geometry.tensor_w.size
        idx = np.arange(0, x_siz * y_siz * z_siz, 1.0)
        gslib_idx = np.reshape(idx, [x_siz, y_siz, z_siz]).flatten(order="F")
        return gslib_idx

    def to_gslib(self, filename: str, nan_value: float = -999.0):
        gslib_idx = self._get_gslib_idx()
        df = self.to_pandas(index=gslib_idx).sort_index()
        df.fillna(nan_value, inplace=True)
        gslib.write(df, filename)

    def to_vtk(self, filename: str, binary: bool = True):
        self._as_vtk.save(filename, binary=binary)

    @classmethod
    def _from_pandas(
        cls,
        name: str,
        dataframe: pd.DataFrame,
        nx: int,
        ny: int,
        nz: int,
        xsiz: float,
        ysiz: float,
        zsiz: float,
        xmin: float,
        ymin: float,
        zmin: float,
        description="",
        subtype="volume",
    ):
        geometry = utils.gridgeom_from_griddef(
            nx, ny, nz, xsiz, ysiz, zsiz, xmin, ymin, zmin
        )
        volume = omf.VolumeElement(
            name=name,
            geometry=geometry,
            data=utils.scalardata_from_df(dataframe),
            description=description,
            subtype=subtype,
        )
        return cls(volume)


class Surface(Element):
    def __init__(self, element):
        super().__init__(element)
        self._as_vtk = omfvista.surface_to_vtk(self._element)

    def to_vtk(self, filename: str, binary: bool = True):
        self._as_vtk.save(filename, binary=binary)


class Project(utils.Wrapper):
    def __init__(self, filepath: str):
        reader = omf.OMFReader(filepath)
        omf_prj = reader.get_project()
        super().__init__(omf_prj)
        element_type_map = {
            "volume": Volume,
            "point": PointSet,
            "color": PointSet,
            "blasthole": PointSet,
            "line": LineSet,
            "borehole": LineSet,
            "surface": Surface,
        }
        self.elements = utils.dotdict(
            {e.name: element_type_map[e.subtype](e) for e in self.elements}
        )

    def summarize(self):
        rich.inspect(self, title=f"OMFProject: {self.name}")

    def to_gslib(self, dir_path: str):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        gslib_dir = pathlib.Path(dir_path)
        for e_name in self.elements:
            e = getattr(self.elements, e_name)
            filepath = gslib_dir.joinpath(f"{e_name}.dat")
            e.to_gslib(filepath)

    def to_vtk(self, dir_path: str):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        vtk_dir = pathlib.Path(dir_path)
        for e_name in self.elements:
            e = getattr(self.elements, e_name)
            filepath = vtk_dir.joinpath(f"{e_name}.dat")
            e.to_vtk(filepath)

    def to_csv(self, dir_path: str):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        csv_dir = pathlib.Path(dir_path)
        for e_name in self.elements:
            e = getattr(self.elements, e_name)
            filepath = csv_dir.joinpath(f"{e_name}.dat")
            e.to_gslib(filepath)

    def __repr__(self) -> str:
        return f"<OOMF Project> (Name: {self.name}, Date Created: {self.date_created})"
