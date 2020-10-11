import inspect
import omf
import pandas as pd
import numpy as np


class Wrapper:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def __dir__(self):
        return dir(self._obj)

def scalardata_from_df(df:pd.DataFrame, data_location:str="vertices"):
    data = []
    for col in df.columns:
        array_vector = df[col].values.tolist()
        idata = omf.ScalarData(
           name=col,
               array=array_vector,
       location=data_location
       )
        data.append(idata)
    return data

def gridgeom_from_griddef(nx,ny,nz,xsiz,ysiz,zsiz,xmin,ymin,zmin):
    tensor_u = np.array([xsiz for i in range(nx)])
    tensor_v = np.array([ysiz for i in range(ny)])    
    tensor_w = np.array([zsiz for i in range(nz)])
    geom = omf.volume.VolumeGridGeometry(origin=(xmin,ymin,zmin), tensor_u=tensor_u, tensor_v=tensor_v, tensor_w=tensor_w)
    return geom

class dotdict(dict):
    """A dict with dot access and autocompletion.

    The idea and most of the code was taken from
    http://stackoverflow.com/a/23689767,
    http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
    http://stackoverflow.com/questions/2390827/how-to-properly-subclass-dict-and-override-get-set
    """

    def __init__(self, *a, **kw):
        dict.__init__(self)
        self.original_keys = {}
        self.update(*a, **kw)
        self.__dict__ = self

    def __setattr__(self, key, value):
        if key in dict.__dict__:
            raise AttributeError("This key is reserved for the dict methods.")
        dict.__setattr__(self, key, value)

    def __setitem__(self, key, value):
        if key in dict.__dict__:
            raise AttributeError("This key is reserved for the dict methods.")
        dict.__setitem__(self, key, value)

    def validate_key(self, key):
        if " " in key:
            new_key = key.replace(" ", "_")
            self.original_keys[new_key] = key
            return new_key
        else:
            return key

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            k = self.validate_key(k)
            self[k] = v

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

    def to_dict(self):
        return {k: v for k, v in self.items()}

    def __repr__(self):
        items = "\n\t".join([f"{k}:{v}," for k, v in self.items()])
        return f"<DotDict>:\n\t{items}"


def delegates(to=None, keep=False):
    "Decorator: replace `**kwargs` in signature with params from `to` - from fast.ai"

    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop("kwargs")
        s2 = {
            k: v
            for k, v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd
        }
        sigd.update(s2)
        if keep:
            sigd["kwargs"] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f
