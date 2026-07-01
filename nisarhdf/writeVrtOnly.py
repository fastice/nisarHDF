#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 2026

@author: ian

Functions for writing a VRT that points directly to bands in a NISAR HDF5 file,
without extracting data to a binary or tiff.  Parallel to _createBinaryVrt in
writeMultiBandVrt.py but uses VRTRasterBand + SimpleSource instead of
VRTRawRasterBand, since the source is HDF5 not raw binary.
"""

import os
import numpy as np
from osgeo import gdal, osr

from nisarhdf.nisarBaseHDF import gdalTypes


# Reverse map: GDAL GDT → dtype string, for probing h5py datasets
_gdalTypeNames = {v: k for k, v in gdalTypes.items()}


def getHDF5BandPath(productObj, field, layer=None):
    '''
    Return the full HDF5-internal path for a given field.

    Mirrors the access pattern in nisarBaseHDF.getImageData:
      - RSLC / GCOV / GSLC:
            /science/{sar}/{product}/{bands}/{frequency}/{field}
      - All others without layer:
            /science/{sar}/{product}/{bands}/{frequency}/{productType}/{polarization}/{field}
      - ROFF / GOFF with layer:
            /science/{sar}/{product}/{bands}/{frequency}/{productType}/{polarization}/{layer}/{field}

    Parameters
    ----------
    productObj : nisarBaseHDF-derived object
        Opened NISAR product (h5 must still be open).
    field : str
        Dataset name (e.g. 'HH', 'unwrappedPhase', 'slantRangeOffset').
    layer : str, optional
        Layer key for offset products (e.g. 'layer1').  Default None.

    Returns
    -------
    str
        Internal HDF5 path starting with '/'.
    '''
    parts = ['', 'science', productObj.sar, productObj.product,
             productObj.bands, productObj.frequency]

    if productObj.product not in ['GCOV', 'RSLC', 'GSLC']:
        if field != 'digitalElevationModel':
            parts.append(productObj.productType)
            parts.append(productObj.polarization)
            if layer is not None:
                parts.append(layer)

    parts.append(field)
    return '/'.join(parts)


def _getH5DType(productObj, field, layer=None):
    '''
    Return the GDAL data-type constant for a field by querying the open h5py
    dataset directly — no data is loaded.

    Parameters
    ----------
    productObj : nisarBaseHDF-derived object
    field : str
    layer : str, optional

    Returns
    -------
    int
        GDAL GDT_* constant.  Falls back to gdal.GDT_Float32 on any error.
    '''
    try:
        bands = productObj.h5[productObj.product][productObj.bands]
        if productObj.product in ['RSLC', 'GCOV', 'GSLC']:
            h5ds = bands[productObj.frequency][field]
        else:
            h5ds = bands[productObj.frequency][productObj.productType][productObj.polarization]
            if layer is not None:
                h5ds = h5ds[layer]
            h5ds = h5ds[field]
        dtype_str = str(np.dtype(h5ds.dtype))
        return gdalTypes.get(dtype_str, gdal.GDT_Float32)
    except Exception:
        return gdal.GDT_Float32


def _createHDF5Vrt(vrtFile, xSize, ySize, sourceFiles, descriptions,
                   eTypes, noDataValues,
                   geoTransform=None,
                   metaData=None,
                   epsg=None):
    '''
    Write a VRT whose bands point to HDF5 subdatasets via SimpleSource.

    This is the HDF5 counterpart of _createBinaryVrt in writeMultiBandVrt.py.
    VRTRawRasterBand cannot be used for HDF5 sources; VRTRasterBand +
    SimpleSource is used instead.

    Parameters
    ----------
    vrtFile : str
        Output VRT filename.
    xSize : int
        Raster width in pixels (range samples).
    ySize : int
        Raster height in pixels (azimuth lines).
    sourceFiles : list of str
        GDAL HDF5 subdataset strings, e.g.
        ['HDF5:"/abs/path.h5"://science/LSAR/RSLC/swaths/frequencyA/HH'].
    descriptions : list of str
        Band description/name for each source.
    eTypes : list of int
        GDAL GDT_* data-type constants, one per band.
    noDataValues : list
        No-data value for each band (None or np.complex64 → not set).
    geoTransform : list of 6 floats, optional
        Standard GDAL geotransform.  Defaults to identity.
    metaData : dict, optional
        Dataset-level metadata (values will be cast to str by GDAL).
    epsg : int, optional
        EPSG code for the SRS.  None → no SRS written.
    '''
    if geoTransform is None:
        geoTransform = [-0.5, 1., 0., -0.5, 0., 1.]

    nBands = len(sourceFiles)

    if os.path.exists(vrtFile):
        os.remove(vrtFile)

    drv = gdal.GetDriverByName('VRT')
    vrt = drv.Create(vrtFile, xSize, ySize, bands=0)
    vrt.SetGeoTransform(geoTransform)

    if epsg is not None:
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(epsg)
        vrt.SetSpatialRef(sr)

    if metaData is not None:
        vrt.SetMetadata(metaData)

    for bandNum, (srcFile, desc, eType, noData) in enumerate(
            zip(sourceFiles, descriptions, eTypes, noDataValues),
            start=1):

        vrt.AddBand(eType)
        band = vrt.GetRasterBand(bandNum)
        band.SetMetadataItem('Description', desc)

        if noData is not None and not isinstance(noData, np.complex64):
            band.SetNoDataValue(noData)

        # Add a SimpleSource pointing at the HDF5 subdataset.
        # GDAL resolves 'new_vrt_sources' metadata items into proper
        # <SimpleSource> XML elements when the VRT driver flushes.
        source_xml = (
            '<SimpleSource>'
            f'  <SourceFilename relativeToVRT="0">{srcFile}</SourceFilename>'
            '  <SourceBand>1</SourceBand>'
            '</SimpleSource>'
        )
        band.SetMetadataItem('source_0', source_xml, 'new_vrt_sources')

    vrt.FlushCache()
    vrt = None


def writeVrtOnly(productObj, filenameRoot, fields,
                 vrtFile=None, layer=None, verbose=False):
    '''
    Write a VRT that points directly to bands in the open HDF5 file,
    without extracting any data.

    The VRT has the same geotransform and metadata block that a normal
    binary write would produce.  For range-Doppler products the geotransform
    uses positive dy (azimuth time increases downward, tiff=False).  For
    geocoded products dy matches the normal tiff convention (tiff=True).

    Note: GDAL requires the HDF5 subdataset format
    ``HDF5:"/abs/path/file.h5"://science/LSAR/...`` — quotes around the
    filename and a double slash (``//``) before the internal path.
    Remote (s3:// / https://) hdfFile paths are embedded as-is;
    GDAL's HDF5 virtual-I/O support is required for those to work.

    Parameters
    ----------
    productObj : nisarBaseHDF-derived object
        Opened NISAR product; metadata parsed, data not required.
    filenameRoot : str
        Root output name.
    fields : list of str
        Fields to include as VRT bands (e.g. ['HH'] or ['unwrappedPhase']).
    vrtFile : str, optional
        Override default VRT path ({filenameRoot}.vrt).
    layer : str, optional
        Layer key for ROFF/GOFF products (e.g. 'layer1').  Default None.
    '''
    productObj.assembleMeta()
    meta = productObj.meta.copy()
    meta['bands'] = fields
    # parseSLCVrtNew in the GrIMP C code calls strstr(get_value(…,"ByteOrder"),…)
    # which crashes with a NULL deref if the key is absent.  NISAR HDF5 data is
    # little-endian; GDAL reads through the band handle so this value is only
    # used by the C metadata parser, not for actual byte-swapping.
    meta['ByteOrder'] = 'LSB'

    if vrtFile is None:
        vrtFile = f'{filenameRoot}.vrt'

    # Dimensions
    rgSize = getattr(productObj, f'{productObj.lookType}RangeSize')
    azSize = getattr(productObj, f'{productObj.lookType}AzimuthSize')

    # Geotransform: positive dy for range-Doppler, normal tiff convention for GEO
    isGeo = (getattr(productObj, 'coordType', 'RD') == 'GEO')
    geoTransform = productObj.getGeoTransform(tiff=isGeo)

    # EPSG only for geocoded products
    epsg = productObj.epsg if isGeo else None

    # Build per-band info
    hdfPath = os.path.abspath(productObj.hdfFile)
    sourceFiles, descriptions, dataTypes, noDataValues = [], [], [], []

    for field in fields:
        internalPath = getHDF5BandPath(productObj, field, layer=layer)
        # GDAL HDF5 subdataset format (as reported by gdalinfo -sd on the file):
        #   HDF5:"/abs/path/file.h5"://science/LSAR/...
        # Quotes around the filename ARE required by GDAL's GDALOpen parser.
        # The internal path must start with '//' (double slash); a single '/'
        # produces "No such file or directory" even though the file exists.
        # For s3:// / https:// URLs os.path.abspath would mangle the scheme,
        # so fall back to the raw hdfFile attribute; quotes still apply.
        ipath = '//' + internalPath.lstrip('/')
        if hdfPath.startswith('/'):
            srcPath = f'HDF5:"{hdfPath}":{ipath}'
        else:
            srcPath = f'HDF5:"{productObj.hdfFile}":{ipath}'

        sourceFiles.append(srcPath)
        descriptions.append(field)
        dataTypes.append(_getH5DType(productObj, field, layer=layer))
        noDataValues.append(productObj.findNoDataValue(field, tiff=True))

    _createHDF5Vrt(vrtFile, rgSize, azSize,
                   sourceFiles, descriptions, dataTypes, noDataValues,
                   geoTransform=geoTransform,
                   metaData=meta,
                   epsg=epsg)

    if verbose:
        print(f'Written HDF5-backed VRT: {vrtFile}')
        for sf in sourceFiles:
            print(f'  -> {sf}')
