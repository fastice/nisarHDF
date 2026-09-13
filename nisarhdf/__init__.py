#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:14:50 2024

@author: ian
"""

import importlib as _importlib
import sys as _sys
import types as _types

__all__ = ['autoScaleRange', 'colorBar', 'createDivider',
           'formatGeojson', 'nisarBaseHDF',
           'nisarBaseGeocodedHDF',
           'nisarBaseRangeDopplerHDF', 'nisarGCOVHDF', 'nisarOrbit',
           'nisarRIFGHDF', 'nisarRSLCHDF', 'nisarRUNWHDF', 'nisarGOFFHDF',
           'nisarGUNWHDF', 'nisarROFFHDF', 'readVrtAsXarray',
           'writeMultiBandVrt']

#
# Names are resolved lazily rather than imported here.  Importing them eagerly
# pulled matplotlib, geopandas, rasterio, rioxarray, dask and boto3 into every
# process that touched the package -- startup the console scripts largely never
# used, and showimage (which needs only numpy and gdal) never used at all; it
# went from 2.1 s to 0.4 s.  'from nisarhdf import X' and 'nisarhdf.X' behave
# exactly as before; only the moment of the import moves.
#
_moduleOf = {'autoScaleRange': 'nisarhdfPlottingTools',
             'colorBar': 'nisarhdfPlottingTools',
             'createDivider': 'nisarhdfPlottingTools',
             'formatGeojson': 'formatGeojson',
             'nisarBaseHDF': 'nisarBaseHDF',
             'nisarBaseRangeDopplerHDF': 'nisarBaseRangeDopplerHDF',
             'nisarBaseGeocodedHDF': 'nisarBaseGeocodedHDF',
             'nisarOrbit': 'nisarOrbit',
             'nisarRIFGHDF': 'nisarRIFGHDF',
             'nisarRSLCHDF': 'nisarRSLCHDF',
             'nisarRUNWHDF': 'nisarRUNWHDF',
             'nisarGOFFHDF': 'nisarGOFFHDF',
             'nisarGCOVHDF': 'nisarGCOVHDF',
             'nisarGUNWHDF': 'nisarGUNWHDF',
             'nisarROFFHDF': 'nisarROFFHDF',
             'writeMultiBandVrt': 'writeMultiBandVrt',
             'readVrtAsXarray': 'readVrtAsXarray'}


_resolved = {}


class _LazyPackage(_types.ModuleType):
    """Package type that resolves the public names in `_moduleOf` on demand.

    A plain PEP 562 module-level `__getattr__` is not enough here.  Most of
    the public names (nisarOrbit, writeMultiBandVrt, nisarROFFHDF, ...) are
    also the names of the submodules that define them, and importing such a
    submodule -- from anywhere, including one sibling importing another --
    makes the import machinery bind the *module* as an attribute of this
    package.  That satisfies normal attribute lookup, so `__getattr__`
    would never fire and
    `nisarhdf.nisarOrbit` would yield the module instead of the class, which
    is not what the eager imports used to give.  Resolving in
    `__getattribute__` keeps the old meaning regardless of import order.
    """

    def __getattribute__(self, name):
        if name in _moduleOf:
            if name not in _resolved:
                module = _importlib.import_module(f'{__name__}.'
                                                  f'{_moduleOf[name]}')
                _resolved[name] = getattr(module, name)
            return _resolved[name]
        return super().__getattribute__(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(__all__))


_sys.modules[__name__].__class__ = _LazyPackage

