#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:37:10 2024

@author: ian
"""

import numpy as np
from nisarhdf.nisarBaseHDF import nisarBaseHDF
from datetime import timedelta
import scipy
import geojson
from nisarhdf import formatGeojson
import os


class nisarBaseRangeDopplerHDF(nisarBaseHDF):
    '''
    This class creates objects to work with nisar Level 1 Range Doppler images.
    '''

    def __init__(self,  **keywords):
        '''
        NISAR ROFF HDF Reader

        Parameters
        ----------
        sar : str, optional
            SAR Idendtifier (always LSAR for now). The default is 'LSAR'.
        frequency : str, optional
            frequency band to extract. The default is 'frequencyA'.
        polarization : str, optional
            Polarization. The default is 'HH'.
        layer : str, optional
            Layer to use. The default is 'layer1'.

        Returns
        -------
        None.

        '''
        self.gridType = 'geolocationGrid'
        self.coordType = 'RD'
        nisarBaseHDF.__init__(self, **keywords)
        self.levelParams = []
        self.RDParams = ['RangeSize', 'AzimuthSize',
                         'RangePixelSize', 'AzimuthPixelSize',
                         'ZeroDopplerTimeSpacing',
                         'NearRange', 'FirstZeroDopplerTime',
                         'IncidenceCenter']

    def getEPSG(self):
        '''
        Get epsg
        Returns
        -------
        None.
        '''
        self.epsg = self.toScalar(
            self.h5[self.product]['metadata'][self.gridType]['epsg'])

    def getExtent(self):
        '''
        Get extent (left, right, bottom, top) for imshow

        Returns
        -------
        None.

        '''
        dr = getattr(self, f'{self.lookType}RangePixelSize')
        da = getattr(self, f'{self.lookType}ZeroDopplerTimeSpacing')
        self.extentUL = [self.slantRangeData[0] - 0.5 * dr,
                         self.slantRangeData[-1] + 0.5 * da,
                         self.zeroDopplerTimeData[-1] - 0.5 * da,
                         self.zeroDopplerTimeData[0] + 0.5 * da]
        self.extentLL = [self.slantRangeData[0] - 0.5 * dr,
                         self.slantRangeData[-1] + 0.5 * da,
                         self.zeroDopplerTimeData[0] + 0.5 * da,
                         self.zeroDopplerTimeData[-1] - 0.5 * da]

    def getPolarizations(self):
        '''
        Parse the list of polarizations
        '''
        swaths = self.h5[self.product]['swaths']
        self.polarizations = [x.decode("utf-8") for x in 
            swaths[self.frequency]['listOfPolarizations']] 
        
    # def getSlantRangeData(self):
    #     ''' Get slant range data '''
    #     bands = self.h5[self.product][self.bands]
    #     self.slantRangeData = np.array(
    #         bands[self.frequency][self.productType]['slantRange'])

    # def getZeroDopplerTimeData(self):
    #     ''' Get zero Doppler time '''
    #     bands = self.h5[self.product][self.bands]
    #     self.zeroDopplerTimeData = np.array(
    #         bands[self.frequency][self.productType]['zeroDopplerTime'])

    #
    # Parameters for multi-look products RUNW and RIFG.
    #

    def getMLSlantRange(self, SLC=False):
        '''
        Compute near/center/far ranges for a multi-look image

        Returns
        -------
        None.

        '''
        if not hasattr(self, 'SLCNearRange'):
            self.getSLCSlantRange()
        # Compute near/center/far ranges.
        self.MLNearRange = self.SLCNearRange + \
            self.SLCRangePixelSize * (self.NumberRangeLooks - 1) * 0.5
        self.MLFarRange = self.MLNearRange + (self.MLRangeSize - 1) * \
            self.SLCRangePixelSize * self.NumberRangeLooks
        self.MLCenterRange = (self.MLNearRange + self.MLFarRange) * 0.5
        # Compute ML pixel size
        self.MLRangePixelSize = self.SLCRangePixelSize * self.NumberRangeLooks
        #
        bands = self.h5[self.product][self.bands]
        # Compute slant range coordinates for ML image
        dR = self.SLCRangePixelSize * self.NumberRangeLooks
        N = int(np.round((self.MLFarRange - self.MLNearRange) / dR + 1))
        self.slantRangeData = np.linspace(self.MLNearRange, self.MLFarRange, N)
      

    def getMLSize(self):
        '''
        Get number of looks in range and azimuth for multilook product.

        Returns
        -------
        None.

        '''
        # Compute ML sizes
        self.MLRangeSize = int(
            np.floor(self.SLCRangeSize / self.NumberRangeLooks))
        self.MLAzimuthSize = int(
            np.floor(self.SLCAzimuthSize / self.NumberAzimuthLooks))

    def getMLZeroDopplerTime(self, secondary=False, SLC=False):
        '''
        Input y5 and return dictionary with zero Doppler spacing and
        first, middle, and last zero Doppler.

        Returns
        -------
        None.

        '''
        if not SLC:
            self.getSLCZeroDopplerTime(secondary=secondary)
        #
        # get start, mid, and end times
        # Note computing from SLC since test products seem to have incorrect
        # value
        self.MLFirstZeroDopplerTime = self.SLCFirstZeroDopplerTime + \
            0.5 * (self.NumberAzimuthLooks - 1) * self.SLCZeroDopplerTimeSpacing
        self.MLZeroDopplerTimeSpacing = self.NumberAzimuthLooks * self.SLCZeroDopplerTimeSpacing
        self.MLLastZeroDopplerTime = \
            (self.MLFirstZeroDopplerTime +
             (self.MLAzimuthSize - 1) * self.MLZeroDopplerTimeSpacing)
        self.MLMidZeroDopplerTime = (self.MLFirstZeroDopplerTime +
                                     self.MLLastZeroDopplerTime) * 0.5
        #
        # compute the nominal time, modulo 86400 to get 24 hour time
        # for cases where data take time > 86400
        self.NominalTime = str(timedelta(
            seconds=np.around(self.MLFirstZeroDopplerTime % 86400., decimals=5)))
        #
        bands = self.h5[self.product][self.bands]
        if not SLC:
            self.zeroDopplerTimeData = np.array(
                     bands[self.frequency][self.productType]['zeroDopplerTime'])

    #
    # Interpolation and data cube methods
    #

    def setupRangeDopplerGrid(self):
        '''
        Form the R/D grid for the product

        Returns
        -------
        None.

        '''
        self.zeroDopplerGrid, self.slantRangeGrid = np.meshgrid(
            self.zeroDopplerTimeData, self.slantRangeData, indexing='ij')

    def setupDataCube(self, cubeName, zGridPts=None):
        '''
        Create an interpolator for the incidence angle data cube

        Parameters
        ----------
        cubeName: str
            name of data cube
        Returns
        -------

        interpolator for the cube

        '''
        metadata = self.h5[self.product]['metadata']
        # Adjust for range, zeroDoppler, z order
        cubeData = np.moveaxis(np.moveaxis(
            np.array(metadata['geolocationGrid'][cubeName]), 0, -1), 0, 1)
        # Get coordinates
        slantRange = np.array(metadata['geolocationGrid']['slantRange'])
        zeroDopplerTime = \
            np.array(metadata['geolocationGrid']['zeroDopplerTime'])
        heightAboveEllipsoid = \
            np.array(metadata['geolocationGrid']['heightAboveEllipsoid'])
        # Extract subset for baseline
        if zGridPts is not None:
            heightAboveEllipsoid = heightAboveEllipsoid[zGridPts]
        # Create interpolator
        setattr(self, f'{cubeName}Interpolator',
                scipy.interpolate.RegularGridInterpolator(
                        (slantRange, zeroDopplerTime, heightAboveEllipsoid),
                        cubeData, bounds_error=False, fill_value=np.nan))

    def interpGrid(self, interpolator, slantRange, zeroDopplerTime, z,
                   maskNoData=None):
        '''
        Apply regular grid interpolator to a matrix

        Parameters
        ----------
        interpolator : TYPE
            Predefined regular grid interpolator for desired variabile.
        slantRange : nparray or list
            slantRange coordinate to interpolate to.
        zeroDopplerTime : nparray or list
            zeroDoppler coordinate to interpolate to..
        z : np array
            Elevation coordinate to interpolate to.
        maskNoData : bool np.array, optional
            Locations to mask. The default is None.
        Returns
        -------
        None.

        '''
        result = interpolator(
            list(zip(self.ensureNP(slantRange).flatten(),
                     self.ensureNP(zeroDopplerTime).flatten(),
                     self.ensureNP(z).flatten()))
                              ).reshape(self.ensureNP(slantRange).shape)
        if maskNoData is not None:
            result[self.maskNoData] = np.nan
        return result

    def xyCube(self, slantRange, zeroDopplerTime, z, maskNoData=None,
               save=False):
        '''
        Interpolate x and y coordinates from data cube

        Parameters
        ----------
        slantRange : nparray
           slantRange coordinates to interpolate to.
        zeroDopplerTime: nparray
            zeroDopplerTime coordinates to interpolate to.
        z : nparray
            Elevation coordinatesto interpolate to.
       save: bool, optional
            If true, save the result as self.result (e.g.,
            self.xGrid, self.yGrid). The default is False.
        Returns
        -------
        coordinateX, coordinateY : nparrays
             x and y components for the projected coordinate system

        '''
        # Setup interpolators if needed.
        if not hasattr(self, 'coordinateX'):
            self.setupDataCube('coordinateX', zGridPts=None)
        if not hasattr(self, 'coordinateY'):
            self.setupDataCube('coordinateY', zGridPts=None)
        # interpolate
        coordinateX = self.interpGrid(self.coordinateXInterpolator,
                                      slantRange, zeroDopplerTime, z,
                                      maskNoData=maskNoData)
        coordinateY = self.interpGrid(self.coordinateYInterpolator,
                                      slantRange, zeroDopplerTime, z,
                                      maskNoData=maskNoData)
        if save:
            self.xGrid = coordinateX
            self.yGrid = coordinateY
        return coordinateX, coordinateY

    def RDtoLatLon(self, slantRange, zdTime, z):
        '''
        Transform R/D coords to lat lon.

        Parameters
        ----------
        slantRange : float
            Slant range in m.
        zdTime : float
            zero Doppler time in s.
        z : float
            WGS84 elevation.

        Returns
        -------
        lat, lon, z
            Lat/lon/z coordinates.

        '''
        x, y = self.xyCube(slantRange, zdTime, z)
        if self.epsg in [4326]:
            return y, x
        lat, lon = self.xytoll(x, y)
        return lat, lon

    #
    # Range Doppler geometry paramters
    #

    def getSceneCenterSatelliteHeight(self):
        '''
        Compute the nominal spacecraft altitude mid scene

        Returns
        -------
        None.

        '''
        midZeroTime = getattr(self, f'{self.lookType}MidZeroDopplerTime')
        # print(self.getSatelliteHeight([midZeroTime]))
        self.SpaceCraftAltitude = self.getSatelliteHeight([midZeroTime])[0]

    def getCenterIncidenceAngle(self):
        '''
        Get center incidence angle

        Returns
        -------
        None.

        '''
        slantRange = getattr(self, f'{self.lookType}CenterRange')
        zeroDopplerTime = getattr(self,f'{self.lookType}MidZeroDopplerTime')
        incAngle =  self.incidenceAngleCube([slantRange],
                                            [zeroDopplerTime],
                                            [0])[0].item()
        # Fall back if cube fails
        if np.isnan(incAngle):
            elevAngle, incAngle = self.computeAngles(slantRange,
                                                zeroDopplerTime,
                                                0,
                                                degrees=True)
        setattr(self, f'{self.lookType}IncidenceCenter', incAngle)
    #
    # Corner and center points.
    #

    def getCorners(self):
        '''
        Extract corners from bounding polygon with redudant points.
        '''
        # Get epsg
        self.getEPSG()
        #
        # Extract minimum rectangle.
        x, y = self._getRectangle()
        #print(x, y)
        #
        # Sort and id corners
        self._idCorners(x, y)

    def _getRectangle(self):
        '''
        From a bounding polygon from a NISAR HDF with vertices between corners,
        extract the polygon with only the corner vertices.

        Parameters
        ----------
        boundingPolygon : POLYGON
            Bounding polygon from hdf.

        Returns
        -------
        x, y: np float array
            corner points in xy coordinates.

        '''
        x, y = np.zeros(4), np.zeros(4)
        if self.epsg is None:
            self.getEPSG()
        # Compute corners
        nearFar = [getattr(self, f'{self.lookType}NearRange'),
                   getattr(self, f'{self.lookType}FarRange')]
        earlyLate = [getattr(self, f'{self.lookType}FirstZeroDopplerTime'),
                     getattr(self, f'{self.lookType}LastZeroDopplerTime')]
        #
        i = 0
        # ll, ul, lr, ur
        for r in nearFar:
            for t in earlyLate:
                # Changed to direct computation so secondary would work
                x[i], y[i] = self.lltoxy(*self.RTtoLatLon(r, t, 0)[0:2])
                i += 1
        # EN, LN, EF, LF
        key = (self.LookDirection.lower(), self.PassType.lower())
        mapping = {
           ("left", "ascending"): np.array([2, 3, 0, 1]),
           ("left", "descending"): np.array([1, 0, 3, 2]),
           ("right", "ascending"): np.array([0, 1, 2, 3]),
           ("right", "descending"):  np.array([3, 2, 1, 0]),
        }
        x = x[mapping[key]]
        y = y[mapping[key]]
        #if self.epsg in [4326]:
        #    return y, x
        return x, y

    def _idCorners(self, x, y):
        '''
        Identify the corners as ll, lr, ur, ul and save in dict
        self.corners = {'ll': (lat, lon), 'lr'....}
        Parameters
        ----------
        x, y : float arrays
            x&y coordinate of corners in arbitrary order.

        Returns
        -------
        None.

        '''
        # order from looping on range then azimuth in _getRectangle
        ll, ul, lr, ur = 0, 1, 2, 3
        lat, lon = self.xytoll(x, y)
        # save in dict
        self.corners = {'ll': (lat[ll], lon[ll]), 'lr': (lat[lr], lon[lr]),
                        'ur': (lat[ur], lon[ur]), 'ul': (lat[ul], lon[ul])}

    def getCenterLatLon(self):
        '''
        Get the center lat/lon for image (at elevation =0)

        Returns
        -------
        None.

        '''
        centerRange = getattr(self, f'{self.lookType}CenterRange')
        centerTime = getattr(self, f'{self.lookType}MidZeroDopplerTime')
        #lat, lon = self.RDtoLatLon([centerRange], [centerTime], [0])
        # Calculate to avoid problem with 
        lat, lon, _ = self.RTtoLatLon(np.array([centerRange]), 
                                      np.array([centerTime]),
                                      np.array([0]))
        self.CenterLatLon = [lat, lon]
        # for compatability with geocoded
        self.centerLat = lat
        self.centerLon = lon

    def getGeoTransform(self, tiff=True, grimp=True):
        '''
        Compute the geotranform for R/D coordinates

        Returns
        -------
        geotranform : list
            A geotransform with slant range and zero Doppler times

        '''
        dR = getattr(self, f'{self.lookType}RangePixelSize')
        rOrigin = getattr(self, f'{self.lookType}NearRange') - dR / 2
        #
        dA = getattr(self, f'{self.lookType}ZeroDopplerTimeSpacing')
        aOrigin = \
            getattr(self, f'{self.lookType}FirstZeroDopplerTime') - dA / 2
        if tiff:
            aOrigin += getattr(self, f'{self.lookType}AzimuthSize') * dA
            dA = -dA
        geoTransform = [rOrigin, dR, 0., aOrigin, 0., dA]
        if self.downsampleFactorRow > 1 or self.downsampleFactorColumn > 1:
            geoTransform = self.rescale_geoTransform(geoTransform,
                                                     self.downsampleFactorRow,
                                                     self.downsampleFactorColumn)
        return geoTransform
#
# GrIMP Specific Code
#

    def getGeojson(self, secondary=False):
        '''
        Get params as geojson file for ML product

        Parameters
        ----------
        filename : str, optional
            filename. The default is None, which will create
            geodatNRxNA.geojson.
        path : str, optional
            Path where file will be created. The default is '.'.

        Returns
        -------
        None.

        '''
        if not hasattr(self, 'geodatDict'):
            self.genGeodatProperties()
        if not hasattr(self, 'corners'):
            self.getCorners()
        #
        # print(self.corners)
        geoJsonGeometry = geojson.Polygon(
            [[self.corners[x][::-1] for x in ['ll', 'ul', 'ur', 'lr', 'll']]])
        #print(geoJsonGeometry)
        #print(self.corners)

        self.geodatGeojson = geojson.Feature(geometry=geoJsonGeometry,
                                             properties=self.geodatDict)

        return formatGeojson(geojson.dumps(self.geodatGeojson))

    def writeGeodatGeojson(self, filename=None, path='.', secondary=False):
        '''
        Write a geodat geojson file for ML product

        Parameters
        ----------
        filename : str, optional
            filename. The default is None, which will create
            geodatNRxNA.geojson.
        path : str, optional
            Path where file will be created. The default is '.'.

        Returns
        -------
        None.

        '''
         # If secondary, pass too secondary and return
        if secondary and hasattr(self, 'secondary'): 
            if filename is None:
                filename = f'geodat{self.NumberRangeLooks}x' \
                    f'{self.NumberAzimuthLooks}.secondary.geojson'
            self.secondary.writeGeodatGeojson(filename=filename, path=path)
            return
        #
        if not hasattr(self, 'geodatDict'):
            self.genGeodatProperties()
        if not hasattr(self, 'corners'):
            self.getCorners()
        #
        # print(self.corners)
        geoJsonGeometry = geojson.Polygon(
            # [[self.corners[x] for x in ['ll', 'lr', 'ur', 'ul', 'll']]])
            [[self.corners[x][::-1] for x in ['ll', 'ul', 'ur', 'lr', 'll']]])
        self.geodatGeojson = geojson.Feature(geometry=geoJsonGeometry,
                                             properties=self.geodatDict)
        #print(self.NumberRangeLooks, self.NumberAzimuthLooks)
        #print(self.geodatDict)
        # Not secondary so continue
        if filename is None:
            filename = f'geodat{self.NumberRangeLooks}x' \
                    f'{self.NumberAzimuthLooks}.geojson'
        #
        # Write and format the string, then write to the file
        geojsonString = formatGeojson(geojson.dumps(self.geodatGeojson))
        # Remove existing file to avoid problems with links
        if os.path.exists(os.path.join(path, filename)):
            os.remove(os.path.join(path, filename))
        with open(os.path.join(path, filename), 'w') as fpGeojson:
            print(geojsonString, file=fpGeojson)

    def genGeodatProperties(self):
        '''
        Create dictionary with all of the properties in a dict ready
        for geojson output

        Returns
        -------
        None.

        '''
        self.geodatDict = {}
        # These are tags for the geojson properties
        keys = ['ImageName', 'Date', 'NominalTime', 'NumberRangeLooks',
                'NumberAzimuthLooks',
                'MLRangeSize', 'MLAzimuthSize', 'PRF', 'MLNearRange',
                'MLCenterRange', 'MLFarRange', 'RangeErrorCorrection',
                'LookDirection', 'PassType', 'CenterLatLon',
                'TimeToFirstSLCSample', 'SkewOffset', 'Squint',
                'squintAnglePolynomial',
                'EarthRadiusMajor', 'EarthRadiusMinor', 'MLIncidenceCenter',
                'SpaceCraftAltitude', 'CorrectedTime', 'Wavelength',
                'SLCRangePixelSize', 'SLCAzimuthPixelSize', 'deltaT']
        #
        # Get values for these keys
        for key in keys:
            self.geodatDict[key] = getattr(self, key)
        #
        # Flat top-level fields mirroring squintAnglePolynomial, since GDAL's
        # OGR GeoJSON driver (used by mosaic3d's C-side reader) can't read
        # nested objects -- only flat scalars/lists, like state vectors.
        if self.squintAnglePolynomial is not None:
            self.geodatDict['squintCoefficients'] = \
                self.squintAnglePolynomial['coefficients']
            self.geodatDict['squintRefRange'] = \
                self.squintAnglePolynomial['refRange']
            self.geodatDict['squintRefAzimuthTime'] = \
                self.squintAnglePolynomial['refAzimuthTime']
        else:
            self.geodatDict['squintCoefficients'] = None
            self.geodatDict['squintRefRange'] = None
            self.geodatDict['squintRefAzimuthTime'] = None
        #
        self.geodatDict['orbit'] = self.referenceOrbit
        self.geodatDict['frame'] = self.frame
        self.geodatDict['CRID'] = getattr(self, 'CRID', None)
        #
        # Now append the state vectors
        for sv in self.orbit.stateVectors:
            svData = self.orbit.stateVectors[sv]
            # Convert to list
            if type(svData) is np.ndarray:
                svData = list(svData)
            self.geodatDict[sv] = svData
        self.geodatDict['coordOrder'] = 'LonLat'

    def getCorrectedTime(self):
        '''
        Correct time so the first time refers to the first slc pixel since
        GrIMP software uses SLC based times and corrects for ML internally

        Returns
        -------
        None.

        '''
        # Half ML pixel correction
        timeCorrection = 0.5 * (self.NumberAzimuthLooks - 1)/self.PRF
        # This being converted to calendar date, so wrap to obtain valid date
        correctedTime = (self.MLFirstZeroDopplerTime - timeCorrection) % 86400.0
       
        correctedTimeString = str(timedelta(
            seconds=np.around(correctedTime, decimals=6)))
        # get pieces
        x = [float(x) for x in correctedTimeString.split(':')]
        # format as string
        self.CorrectedTime = f'{int(x[0]):02} {int(x[1]):02} {x[2]:.7f}'

    def getRangeErrorCorrection(self):
        '''
        get the range error correction, for now just 0

        Returns
        -------
        None.

        '''
        self.RangeErrorCorrection = 0.0

    def getTimeToFirstSample(self):
        '''
        For now this is zero and is just for compatability with geodat

        Returns
        -------
        None.

        '''
        self.TimeToFirstSLCSample = 0.0

    def getSkewOffsets(self):
        '''
        For now this is zero and is just for compatability with geodat

        Returns
        -------
        None.

        '''
        self.SkewOffset = 0.0

    def getSquint(self):
        '''
        For now this is zero and is just for compatability with geodat

        Returns
        -------
        None.

        '''
        self.Squint = 0.0
        self.squintAnglePolynomial = None

    def getDeltaT(self):
        '''
        Time correction sometimes used for other SARs, but for now is 0.

        Returns
        -------
        None.

        '''
        self.deltaT = 0.0
