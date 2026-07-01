#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 08:47:03 2024

@author: ian
"""
from abc import ABCMeta, abstractmethod
import h5py
import numpy as np
import pyproj
from osgeo import gdal, osr
from scipy import optimize
import os
from nisarhdf.nisarOrbit import nisarOrbit
import geopandas as gpd
import rasterio
from datetime import datetime
from nisarhdf.writeMultiBandVrt import writeMultiBandVrt
from nisarhdf.nisarhdfPlottingTools import autoScaleRange
import gc
# import s3fs
import boto3
import requests
import io
import matplotlib.colors as mcolors
#import psutil
gdal.UseExceptions()

gdalTypes = {'float32': gdal.GDT_Float32,
             'float64': gdal.GDT_Float64,
             'complex64':  gdal.GDT_CFloat32,
             'uint16': gdal.GDT_UInt16,
             'int8': gdal.GDT_Byte,
             'uint8': gdal.GDT_Byte}


class pixelOffsets():
    '''
    Class to hold interogram pixel offsets
    '''

    def __init__(self, h5Object, polarization, product,
                 bands=['alongTrackOffset', 'slantRangeOffset',
                        'correlationSurfacePeak']):
        self.dataFields = []
        for band in bands:
            data = np.array(h5Object[polarization][band])
            setattr(self, band, data)
            self.dataFields.append(band)
        if product in ['RIFG', 'RUNW']:
            self.zeroDopplerTime = h5Object['zeroDopplerTime']
            self.slantRange = h5Object['slantRange']
        elif product in ['GUNW']:
            self.xCoordinate = h5Object[polarization]['xCoordinates']
            self.yCoordinate = h5Object[polarization]['yCoordinates']
        else:
            self.printError(
                f'Invalid product type ({product}) in pixelOffsets')


class nisarBaseHDF():
    '''
    Abstract class to define parser for NISAR HDF products.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, **keywords):
        '''
        Initialize a nisar hdf parser
        Parameters
        ----------
        sar : TYPE, optional
            hdf key. The default is 'LSAR'.
        product : TYPE, optional
            hdf key. The default is 'RUNW'.
        frequency : TYPE, optional
            hdf key.. The default is 'frequencyA'.
        productType : TYPE, optional
            hdf key.. The default is 'interferogram'.
        polarization : TYPE, optional
            hdf key.. The default is 'HH'.
        layer : TYPE, optional
            hdf key.. The default is None.
        productData : TYPE, optional
            hdf key.. The default is 'unwrappedPhase'.

        Returns
        -------
        None.

        '''
        self.initTags(**keywords)
        # Should be set, but just in case
        self.s3cred = None
        # Constants WGS84
        self.EarthRadiusMajor = 6378137.0
        self.EarthRadiusMinor = 6356752.3142
        self.EarthF = 1./298.257223563
        self.f = (self.EarthRadiusMajor -
                  self.EarthRadiusMinor) / self.EarthRadiusMajor
        self.cLight = 2.99792458e8
        self.epsg = None
        self.memory_file = None
        self.h5Full = None
        # this only get set true for GCOV
        self.dB = False
        self.sigma0 = False
        self.noDataLocations = None
        #
        self.commonParams = ['track', 
                             'referenceOrbit',
                             'frame',
                             'datetime',
                             'centerLat',
                             'centerLon',
                             'referenceGranule',
                             'rangeBandwidth',
                             'SLCNearRange', 'SLCFarRange',
                             'SLCFirstZeroDopplerTime',
                             'SLCLastZeroDopplerTime']
        if self.product not in ['GCOV', 'RSLC', 'GSLC']:
            self.commonParams += ['secondaryOrbit', 'secondaryDatetime',
                                  'secondaryGranule']
        self.commonParams += ['LookDirection', 'PassType',
                              'Wavelength', 'PRF', 'epsg']
        #
        self.noDataValuesTiff = {'connectedComponents': 65535,
                                 'coherenceMagnitude': np.nan,
                                 'ionospherePhaseScreen': np.nan,
                                 'ionospherePhaseScreenUncertainty': np.nan,
                                 'unwrappedPhase': np.nan,
                                 'wrappedInterferogram':
                                     np.complex64(np.nan + 1j*np.nan),
                                 'slantRangeOffset': np.nan,
                                 'slantRangeOffsetVariance': np.nan,
                                 'alongTrackOffset': np.nan,
                                 'alongTrackOffsetVariance': np.nan,
                                 'crossOffsetVariance': np.nan,
                                 'correlationSurfacePeak': np.nan,
                                 'snr': np.nan}

        self.noDataValuesBinary = {'connectedComponents': 65535,
                                   'coherenceMagnitude': -2e9,
                                   'ionospherePhaseScreen': -2e9,
                                   'ionospherePhaseScreenUncertainty': -2e9,
                                   'unwrappedPhase': -2e9,
                                   'wrappedInterferogram':
                                       np.complex64(-2e9 + 1j * -2e9),
                                   'slantRangeOffset': -2e9,
                                   'slantRangeOffsetVariance': -2e9,
                                   'alongTrackOffset': -2e9,
                                   'alongTrackOffsetVariance': -2e9,
                                   'crossOffsetVariance': -2e9,
                                   'correlationSurfacePeak': -2e9,
                                   'snr': -2e9}

    #
    # Abtract method placeholders
    #
    @abstractmethod
    def parseParams(**keywords):
        ''' Abstract pointPlot method - define in child class. '''
        pass
    
    @abstractmethod
    def getPolarizations(self, **keywords):
        ''' Abstract pointPlot method - define in child class. '''
        pass

    @abstractmethod
    def setupDataCube(self, *args, **keywords):
        ''' Abstract pointPlot method - define in child class. '''
        pass

    @abstractmethod
    def interpGrid(self, *args, **keywords):
        ''' Abstract pointPlot method - define in child class. '''
        pass

    @abstractmethod
    def getEPSG(**keywords):
        ''' Abstract pointPlot method - define in child class. '''
        pass

    @abstractmethod
    def getExtent(**keywords):
        ''' Abstract pointPlot method - define in child class. '''
        pass

    @abstractmethod
    def getGeoTransform(**keywords):
        ''' Abstract pointPlot method - define in child class. '''
        pass
    #
    # Various "utility" methods for printing, type conversion, coordinate
    # conversion etc.
    #

    def refresh(self, dataFields=None, useNumpy=True):
        '''
        Reload data to undo any modifications

        Parameters
        ----------
        dataFields : list, optional
            List of fields. The default is None, which forces all.

        Returns
        -------
        None.

        '''
        if dataFields is None:
            dataFields = self.dataFields
        if self.product in ['ROFF', 'GOFF']:
            self.getLayers(dataFields)
            if self.product in ['ROFF']:
                self.loadData(['digitalElevationModel'], resetFields=False)
        else:
            self.loadData(dataFields, useNumpy=useNumpy)

    def initTags(self, **kwargs):
        '''
        Initialize tags as attributes

        Parameters
        ----------
        **kwargs : keywords
            Keywords to initialize or overwrite.

        Returns
        -------
        None.

        '''
        for key in kwargs:
            setattr(self, key, kwargs[key])
        #
        

    def ensureNP(self, x):
        '''
        Make sure variable is an np array

        Parameters
        ----------
        x : list or nd.array
            Convert list to numpy array if needed.

        Returns
        -------
        None.

        '''
        if isinstance(x, list):
            x = np.array(x)
        return(x)

    def assembleMeta(self, bands=None):
        '''
        Create a dictionary with all of the meta data to save

        Returns
        -------
        None.

        '''
        self.meta = {}
        for param in self.commonParams + self.levelParams + self.productParams:
            if hasattr(self, param):
                self.meta[param] = getattr(self, param)
        if bands is not None:
            self.meta['bands'] = bands
        # Now append the state vectors
        if self.product in ['RSLC']:
            for sv in self.orbit.stateVectors:
                svData = self.orbit.stateVectors[sv]
                # Use .tolist() so numpy scalars become plain Python floats;
                # list() alone leaves np.float64 elements whose repr changed
                # in NumPy 2.0 to include the 'np.float64(...)' wrapper.
                if isinstance(svData, np.ndarray):
                    svData = svData.tolist()
                self.meta[sv] = svData

    def printParams(self):
        '''
        Print some of the more important parameters

        Returns
        -------
        None.

        '''
        # Print parameters
        print("\n\033[1mSelect Parameters\033[0m")
        for param in self.commonParams + self.levelParams + self.productParams:
            if hasattr(self, param):
                paramValue = getattr(self, param)
                if 'Looks' in param and paramValue < 0:
                    paramValue = 'Variable'
                print(f'{self.product}.{param} = {paramValue} ')
            else:
                print(f'{self.product}.{param} = None')
        if self.product in ['GOFF', 'ROFF']:
            for layer in self.layers:
                print(f'{self.product}.layer{layer} = '
                      f'{self.windowParams[f"layer{layer}"]}')
        # List data fields
        if hasattr(self, 'layers'):
            endString = f'\033[1m for layer(s) {self.layers}\n\033[0m'
        else:
            endString = '\n'
        print("\n\033[1mData Fields\033[0m", end=endString)
        polString = ''
        if self.polarization is not None and self.product not in ['RSLC']:
            polString = f'{self.polarization}.'
        empty = False
        if not hasattr(self, 'dataFields'):
            return
        for field in self.dataFields:
            if hasattr(self, field):
                status = ''
                if getattr(self, field) is None:
                    status = '*'    
                    empty = True
                print(f'{self.product}.{polString}{field}{status}')
            else:
                print(f'{self.product}.{field} = Not loaded, use '
                      f'{self.product}.refresh() to load')
        if empty:
            print('* Indicates data not loaded into memory.')
        #
 

    def printError(self, msg):
        '''
        Print error message
        Parameters
        ----------
        msg : str
            error message.
        Returns
        -------
        None

        '''
        print(f'\n\t\033[1;31m *** {msg} *** \033[0m\n')

    def parseString(self, hdfString):
        '''
        Reformat an hdf string
        '''
        return np.array(hdfString).item().decode()

    def toScalar(self, hdfValue):
        '''
        Convert rscalar hdf value to regular value
        '''
        return np.array(hdfValue).item()

    def parseDateStr(self, dateStr):
        '''
        Parse a date str

        Parameters
        ----------
        dateStr : str
            Date str.

        Returns
        -------
        dateAndTime : datetime
            date and time as datetime.
        date : str
            Date (without time) as string.

        '''
        sLen = min(26, len(dateStr))
        dateStr = dateStr[0:sLen]
        # parse date
        try:
            dateAndTime = datetime.strptime(dateStr, '%Y-%m-%dT%H:%M:%S.%f')
            date = dateAndTime.strftime('%Y-%m-%d')
            return dateAndTime, date
        except Exception:
            self.printError('Could not parse date')
            return None, None

    def getPolarization(self, polarization):
        '''
        Check polarization in product, otherwise pick first available
        '''
        if polarization is not None and polarization in ['HH', 'VV', 'HV', 'VH']:
            self.polarization = polarization
            return
        # pick first like pol polarization
        for pol in self.polarizations:
            if pol in ['HH', 'VV']:
                self.polarization = pol
                return

    def parseRefDate(self):
        '''
        Parse the reference date

        Returns
        -------
        None.

        '''
        image = 'reference'
        if self.isSecondary:
            image = 'secondary'
        if self.product not in ['GCOV', 'RSLC', 'GSLC']:
            key = f'{image}ZeroDopplerStartTime'
        else:
            key = 'zeroDopplerStartTime'
        #
        # print(self.product, key)
        dateStr = self.parseString(
            self.h5['identification'][key])
        #
        self.datetime, self.Date = self.parseDateStr(dateStr)

        if self.product in ['GCOV', 'RSLC', 'GSLC']:
            return
        #
        if not self.isSecondary:
            dateStr = self.parseString(
                self.h5['identification']['secondaryZeroDopplerStartTime'])
            self.secondaryDatetime, self.secondaryDate = self.parseDateStr(
                dateStr)

    def lltoxy(self, lat, lon):
        '''
        Convert to lat/lon to xy
        '''
        if not hasattr(self, 'epsg'):
            self.getEPSG()
        if not hasattr(self, 'lltoxyXform'):
            self.lltoxyXform = pyproj.Transformer.from_crs("EPSG:4326",
                                                           f"EPSG:{self.epsg}")
        return self.lltoxyXform.transform(lat, lon)

    def xytoll(self, x, y):
        '''
        Convert to lat/lon to xy
        '''
        if not hasattr(self, 'epsg'):
            self.getEPSG()
        if not hasattr(self, 'xytollXform'):
            self.xytollXform = pyproj.Transformer.from_crs(f"EPSG:{self.epsg}",
                                                           "EPSG:4326")
        return self.xytollXform.transform(x, y)

    #
    # Open hdf and extract parameters.
    #
    def _getS3cred(self):
        '''
        Get S3 credentials
        '''
        try:
            s = boto3.Session()
            self.s3cred = s.get_credentials()
        except Exception as e:
            print(f"Could not open s3 credentials: {e}")
            return

    def _openHTTPInMemory(self, asf_url, verbose=True):
        """
        Download a NISAR ASF file into memory and open with h5py.
        Does not use ros3 (only sec2), so the file is fully read into RAM.

        Parameters:
            asf_url (str): The original ASF redirect URL
            verbose (bool): Print debug output

        Returns:
            h5py.File object
        """
        if verbose:
            print(f"[INFO] Resolving redirect: {asf_url}")

        # Get presigned S3 URL
        response = requests.get(asf_url, allow_redirects=False)
        if response.status_code not in (301, 302, 303):
            raise RuntimeError(
                f"Unexpected status code: {response.status_code}")
        presigned_url = response.headers['Location']

        if verbose:
            print(f"[INFO] Downloading from: {presigned_url}")

        # Stream into memory
        s3_response = requests.get(presigned_url, stream=True)
        s3_response.raise_for_status()

        self.memory_file = io.BytesIO()
        for chunk in s3_response.iter_content(chunk_size=1048576):
            self.memory_file.write(chunk)
        self.memory_file.seek(0)  # rewind

        if verbose:
            print("[INFO] File downloaded into memory, opening with h5py...")

        return h5py.File(self.memory_file, 'r')

    def _openS3InMemory(self, s3url, verbose=True):
        """
        Download an S3 HDF5 file into memory and open with h5py.
        This avoids using the ros3 driver by using boto3 + BytesIO.

        Parameters:
            s3url (str): e.g., 's3://my-bucket/path/to/file.h5'
            verbose (bool): Show download progress/debug info

        Returns:
            h5py.File object
        """
        if s3url.startswith("s3://"):
            s3url = s3url[5:]
        bucket, key = s3url.split("/", 1)

        if self.s3cred is None:
            self._getS3cred()

        if verbose:
            print(f"[INFO] Downloading s3://{bucket}/{key} into memory...")

        # Set up boto3 session with temp credentials
        session = boto3.Session(
            aws_access_key_id=self.s3cred.access_key,
            aws_secret_access_key=self.s3cred.secret_key,
            aws_session_token=self.s3cred.token,
            region_name=os.getenv("AWS_REGION", "us-west-2"),
        )
        s3 = session.client("s3")
        # Download the file into a BytesIO buffer
        self.memory_file = io.BytesIO()
        s3.download_fileobj(Bucket=bucket, Key=key, Fileobj=self.memory_file)
        self.memory_file.seek(0)

        if verbose:
            print("[INFO] File downloaded into memory, opening with h5py...")

        return h5py.File(self.memory_file, "r")
       
    def _openS3(self, s3link, page_buf_size=2 * 1024**3, verbose=True):
        '''
        Open s3 link using ros3. Leaving to document, but switching to
        _openS3InMemory
        Parameters
        ----------
        s3link : str
            s3 link.
        page_buf_size: int
            page buf size. The default is 2 * 1024**3.
        '''
        if verbose:
            print(f'page_buf_size {page_buf_size/1e6} MB')
        # *** Update one s3 2k token issue fixed ***
        if self.s3cred is None:
            self._getS3cred()
        #
        region = os.getenv("AWS_REGION", "<unknown>")
        # Open the file and return
        if verbose:
            print(f'Opening {s3link} with ros3')
        return h5py.File(name=s3link,
                         mode="r",
                         driver="ros3",
                         page_buf_size=page_buf_size,
                         aws_region=region.encode(),
                         secret_id=self.s3cred.access_key.encode(),
                         secret_key=self.s3cred.secret_key.encode(),
                         session_token=self.s3cred.token.encode())

    def openHDF(self, hdfFile, referenceOrbitXML=None, secondaryOrbitXML=None,
                referenceOrbit=None, secondaryOrbit=None, noLoadData=False,
                closeH5=False, page_buf_size=10*1024**3,
                useRos3=False, fields=None, verbose=False,
                downsampleFactor={'downsampleFactorRow': 1,'downsampleFactorColumn': 1},
                **keywords):
        '''
        Open hdf and save self.h5Full and truncate to self.h5

        Parameters
        ----------
        hdfFile : str
            hdf file.
        referenceOrbitXML : str, optional
            XML orbit file to override HDF statevectors. The default is None.
        secondaryOrbitXML : str, optional
            XML orbit file to override HDF statevectors.. The default is None.
        referenceOrbit : int, optional
            Override secondary orbit in HDF (used in development where missing
            in hdf). The default is None.
        secondaryOrbit : int, optional
            Override secondary orbit in HDF (used in development where missing
            in hdf). The default is None.
        useRos3: Boolean, optional
            Use ros3 driver for s3 files. The default is False
        fields: list
            list of fields to load. The default is None, which forces all.
        **keywords : TYPE
            keywords to pass to parse params.

        Returns
        -------
        None.

        '''
        # Update XMLs
        #
        if type(downsampleFactor) is dict:
            self.downsampleFactorRow = downsampleFactor['downsampleFactorRow']
            self.downsampleFactorColumn = downsampleFactor['downsampleFactorColumn']
        else:
            self.downsampleFactorColumn = downsampleFactor
            self.downsampleFactorRow = downsampleFactor
        #   
        for attr, value in zip(['referenceOrbitXML', 'secondaryOrbitXML'],
                               [referenceOrbitXML, secondaryOrbitXML]):
            if value is not None:
                setattr(self, attr, value)
            else:
                setattr(self, attr, None)
        # Open hdf file
        self.hdfFile = hdfFile
        # Set page_buf_size for NISAR optimized HDFs
        try:
            if 's3' in hdfFile:
                if useRos3:
                    self.h5Full = self._openS3(hdfFile,
                                               page_buf_size=page_buf_size,
                                               verbose=verbose)
                else:
                    self.h5Full = self._openS3InMemory(hdfFile, verbose=verbose)
            elif 'https' in hdfFile:
                self.h5Full = self._openHTTPInMemory(hdfFile, verbose=verbose)
            else:
                if not os.path.exists(hdfFile):
                    self.printError(f'{hdfFile} does not exist')
                    return
                self.h5Full = h5py.File(hdfFile, 'r')
        except Exception:
            print('Could not open with page_buf_size, opening for '
                  'non-optimized access')
        # Truncate to remove tags common to all
        self.h5 = self.h5Full['science'][self.sar]
        #    
        self.getPolarizations()
        # Set image name
        self.ImageName = os.path.basename(hdfFile)
        # Parse primary parameters.
        self.parseParams(referenceOrbit=referenceOrbit,
                         secondaryOrbit=secondaryOrbit,
                         noLoadData=noLoadData, fields=fields,
                         **keywords)   
        #
        if closeH5:
            self.close()

    def getGranuleNames(self):
        '''
        Get the granule names

        Returns
        -------
        None.

        '''
        procInfo = self.h5[self.product]['metadata']['processingInformation']
        if self.product in ['GCOV']:
            self.referenceGranule = \
                self.parseString(procInfo['inputs']['l1SlcGranules'])
            return
        # all other products
        self.referenceGranule = \
            self.parseString(procInfo['inputs']['l1ReferenceSlcGranules'])
        self.secondaryGranule = \
            self.parseString(procInfo['inputs']['l1SecondarySlcGranules'])

    def getLookDirection(self):
        '''
        Get look direction

        Returns
        -------
        None.

        '''
        self.LookDirection = self.parseString(
            self.h5['identification']['lookDirection']).lower()
        self.lookSign = {'right': 1.0, 'left': -1.0}[self.LookDirection]

    def getNumberOfLooks(self):
        '''
        Get number of looks in range and azimuth for multilook product.

        Returns
        -------
        None.

        '''
        # break up long dict
        metadata = self.h5[self.product]['metadata']
        parameters = metadata['processingInformation']['parameters']
        if self.product in ['GCOV']:
            # number of looks is variable
            self.NumberRangeLooks = -1
            self.NumberAzimuthLooks = -1
        else:
            productType = parameters[self.productType]
        #
            self.NumberRangeLooks = self.toScalar(
                productType[self.frequency]['numberOfRangeLooks'])
            self.NumberAzimuthLooks = self.toScalar(
                productType[self.frequency]['numberOfAzimuthLooks'])


    def getRangeBandWidth(self):
        '''
        Get range bandwidth.

        Returns
        -------
        None.

        '''
        # break up long dict
        metadata = self.h5[self.product]['metadata']
        if self.product in ['GCOV', 'RSLC']:
            if self.product == 'GCOV':
                parameters = metadata['sourceData']
            else:
                parameters = self.h5[self.product]
            productType = parameters['swaths']
            self.rangeBandwidth = self.toScalar(
                productType[self.frequency]['acquiredRangeBandwidth'])
            return
        # Other products
        parameters = metadata['processingInformation']['parameters']
        productType = parameters[self.productType]
        #
        self.rangeBandwidth = self.toScalar(
                productType[self.frequency]['rangeBandwidth'])

    def getOrbitAndFrame(self, referenceOrbit=None,
                         secondaryOrbit=None, frame=None, track=None, **kywds):
        '''
        Get product frame, orbit, and track.

        Returns
        -------
        None.

        '''
        #
        if track is None:
            self.track = self.toScalar(
                self.h5['identification']['trackNumber'])
        else:
            self.track = track
        #
        if frame is None:
            self.frame = self.toScalar(
                self.h5['identification']['frameNumber'])
        else:
            self.frame = frame
        #
        if referenceOrbit is None:
            if self.product in ['GCOV']:
                orbit = self.toScalar(
                    self.h5['identification']['absoluteOrbitNumber'])

            else:
                try:
                    orbit = self.toScalar(
                        self.h5['identification']['referenceAbsoluteOrbitNumber'])
                # for backwards compatability
                except Exception:
                    orbit = self.toScalar(
                        self.h5['identification']['absoluteOrbitNumber'])
            self.referenceOrbit = orbit
        else:
            self.referenceOrbit = referenceOrbit
        #
        if self.product in ['GCOV', 'RSLC', 'GLSC']:
            return
        #
        if secondaryOrbit is None:
            # This try/except handles change in key so older products
            # continue to work
            try:
                self.secondaryOrbit = self.toScalar(
                    self.h5['identification']['secondaryAbsoluteOrbitNumber'])
            except Exception:
                self.secondaryOrbit = -1
                print('Can not read secondary orbit for this product version')
        else:
            self.secondaryOrbit = secondaryOrbit

    def effectivePRF(self):
        '''
        Compute an effective PRF for compatibility with other GrIMP software
            given a zero Doppler spacing and number of azimuth looks.

        Returns
        -------
        None.

        '''
        #
        # Compute the effective PRF
        self.PRF = 1. / self.SLCZeroDopplerTimeSpacing

    def getSLCSlantRange(self):
        '''
        Get SLC slantRange information

        Returns
        -------
        None.

        '''
        # save bands to shorten lines
        metadata = self.h5[self.product]['metadata']
        #
        if self.product not in ['GCOV']:
            parameters = metadata['processingInformation']['parameters']
            if not self.isSecondary:
                imageParams = parameters['reference'][self.frequency]
            else:
                imageParams = parameters['secondary'][self.frequency]
        else:
            imageParams = metadata['sourceData']['swaths'][self.frequency]
        #
        self.SLCRangeSize = self.toScalar(imageParams['numberOfRangeSamples'])
        #
        # Get near range and SLC spacing for either reference or seconeary
        self.SLCNearRange = self.toScalar(imageParams['slantRangeStart'])
        self.SLCRangePixelSize = self.toScalar(
            imageParams['slantRangeSpacing'])
        self.SLCNumberOfRangeSamples = self.toScalar(
            imageParams['numberOfRangeSamples'])
        self.SLCFarRange = self.SLCNearRange + \
            (self.SLCRangeSize - 1) * self.SLCRangePixelSize
        self.SLCCenterRange = (self.SLCNearRange + self.SLCFarRange) * 0.5

    def getCubeEpoch(self, secondary=False):
        '''
        Return the datetime epoch ("seconds since <epoch>") for this image's
        own orbit/state-vector time axis (metadata/orbit, or .../reference
        or .../secondary for interferometric products -- the same group
        parseStateVectors() reads to build self.orbit), parsed from that
        group's 'time' dataset 'units' attribute.

        NISAR fixes this epoch at midnight of the day the whole pass
        started, the same for every frame in the pass -- so a frame acquired
        after a UTC-midnight rollover has orbit/cube times that run past
        86400s rather than resetting to 0. Reference and secondary images
        (different acquisitions, different passes) each have their own
        epoch, hence the secondary flag -- using the wrong one here would
        desync SLCFirstZeroDopplerTime from self.orbit's actual time domain
        and break orbit interpolation (RTtoECEF/getSatPositionAndVel).
        Returns None (caller falls back to the acquisition's own
        calendar-day midnight) if unavailable for this product type.
        '''
        try:
            orbitGroup = self.h5[self.product]['metadata']['orbit']
            if self.product not in ['RSLC', 'GCOV', 'GSLC']:
                orbitGroup = orbitGroup['secondary' if secondary
                                        else 'reference']
            units = self.parseString(orbitGroup['time'].attrs['units'])
            epochStr = units.split('seconds since ')[-1]
            return datetime.strptime(epochStr[:19], '%Y-%m-%dT%H:%M:%S')
        except Exception:
            return None

    def getSLCZeroDopplerTime(self, secondary=False):
        '''
        Get Zero doppler start time from the SLC information in the metdaa

        Parameters
        ----------
        isSecondary : booulan, optional
            indicates Secondary. The default is False.

        Returns
        -------
        None.

        '''
        zeroTime = {'hour': 0, 'minute': 0, 'second': 0, 'microsecond': 0}
        metadata = self.h5[self.product]['metadata']
        if self.product not in ['GCOV']:
            parameters = metadata['processingInformation']['parameters']
            SLC = {True: 'secondary', False: 'reference'}[secondary]
            imageParams = parameters[SLC][self.frequency]
        else:
            imageParams = metadata['sourceData']['swaths']
        #
        self.SLCAzimuthSize = self.toScalar(
            imageParams['numberOfAzimuthLines'])
        #
        earlyTime, _ = self.parseDateStr(self.parseString(
            imageParams['zeroDopplerStartTime']))
        #
        # SLC first time, referenced to the same epoch as this image's own
        # orbit/geolocation cube (see getCubeEpoch()) so that orbit
        # interpolation (RTtoECEF) and cube-interpolated quantities derived
        # from MLFirstZeroDopplerTime/MLLastZeroDopplerTime (e.g.
        # getSquintAnglePolynomial, getCenterIncidenceAngle) stay in range
        # for frames acquired just after a UTC-midnight rollover.
        epoch = self.getCubeEpoch(secondary=secondary) or \
            earlyTime.replace(**zeroTime)
        self.SLCFirstZeroDopplerTime = (earlyTime - epoch).total_seconds()
        self.SLCZeroDopplerTimeSpacing = self.toScalar(
            imageParams['zeroDopplerTimeSpacing'])
        self.SLCLastZeroDopplerTime = self.SLCFirstZeroDopplerTime + \
            (self.SLCAzimuthSize - 1) * self.SLCZeroDopplerTimeSpacing
        self.SLCMidZeroDopplerTime = (self.SLCFirstZeroDopplerTime +
                                      self.SLCLastZeroDopplerTime) * 0.5
        # Compute scene center along track spacing
        # will automatically use the correct orbit for computation
        self.SLCSceneCenterAlongTrackSpacing()

    def SLCSceneCenterAlongTrackSpacing(self):
        '''
        Compute self.SLCAzimuthPixelSize directly since sample data products
        had errors and this value was not provided for all data types.

        Returns
        -------
        None.

        '''
        if not hasattr(self, 'orbit'):
            self.orbit = self.parseStateVectors()
        # Compute positions as scent center delta ZDT apart
        p1 = self.RTtoECEF(self.SLCCenterRange,
                           self.SLCMidZeroDopplerTime,
                           0)
        p2 = self.RTtoECEF(self.SLCCenterRange,
                           self.SLCMidZeroDopplerTime +
                           self.SLCZeroDopplerTimeSpacing,
                           0)
        self.SLCAzimuthPixelSize = np.sqrt(np.sum((p1-p2)**2))

    def getOffsetWindowParams(self, layers=['layer1', 'layer2', 'layer3']):
        '''
        Get the window sizes for the individual layers

        Parameters
        ----------
        layers : TYPE, optional
            DESCRIPTION. The default is ['layer1', 'layer2', 'layer3'].

        Returns
        -------
        None.

        '''
        procInfo = self.h5[self.product]['metadata']['processingInformation']
        frequency = procInfo['parameters'][self.productType][self.frequency]
        #
        self.windowParams = {}
        #
        for layer in layers:
            self.windowParams[layer] = {}
            self.windowParams[layer]['alongTrackSearchWindowSize'] = \
                np.array(frequency[layer]['alongTrackSearchWindowSize']).item()
            self.windowParams[layer]['alongTrackWindowSize'] = \
                np.array(frequency[layer]['alongTrackWindowSize']).item()
            self.windowParams[layer]['slantRangeSearchWindowSize'] = \
                np.array(frequency[layer]['slantRangeSearchWindowSize']).item()
            self.windowParams[layer]['slantRangeWindowSize'] = \
                np.array(frequency[layer]['slantRangeWindowSize']).item()

    def getOrbitPassDirection(self):
        '''
        Extract the orbit pass direction (Ascending/Descending)

        Returns
        -------
        None.

        '''
        passType = self.parseString(
            self.h5['identification']['orbitPassDirection']).lower()
        # print(passType)
        self.PassType = {'ascending': 'ascending',
                         'descending': 'descending'
                         }[passType]

    def getWavelength(self):
        '''
        Get the wavelength

        Returns
        -------
        None.

        '''
        # get frequency
        cFreqName = 'centerFrequency'
        if self.product in ['RSLC']:
            bands = self.h5['RSLC']['swaths']
            cFreqName = 'processedCenterFrequency'
        elif self.product in ['GCOV']:
            bands = self.h5[self.product]['metadata']['sourceData']['swaths']
        else:
            bands = self.h5[self.product][self.bands]
        
        #
        self.centerFrequency = self.toScalar(
                bands[self.frequency][cFreqName])
        # And compute the wavelength
        self.Wavelength = self.cLight / self.centerFrequency

    def parseStateVectors(self, XMLOrbit=None, SLC=False):
        '''
        Setup an nisarOrbit instance to contain the state vectors

        Returns
        -------
        orbit  nisarOrbit.
            NISAR orbit instance with state vector information

        '''
        if XMLOrbit is None:
            if not self.isSecondary:
                h5OrbitGroup = \
                    self.h5[self.product]['metadata']['orbit']
                if self.product not in ['RSLC', 'GCOV', 'GSLC']:
                    h5OrbitGroup = h5OrbitGroup['reference']
            else:
                h5OrbitGroup = \
                    self.h5[self.product]['metadata']['orbit']['secondary']
            # print(h5OrbitGroup)
            orbit = nisarOrbit(
                h5OrbitGroup=h5OrbitGroup,
                firstZeroDopplerTime=self.SLCFirstZeroDopplerTime,
                lastZeroDopplerTime=self.SLCLastZeroDopplerTime)
        else:
            orbit = nisarOrbit(
                XMLOrbit=XMLOrbit,
                referenceDate=self.datetime,
                firstZeroDopplerTime=self.SLCFirstZeroDopplerTime,
                lastZeroDopplerTime=self.SLCLastZeroDopplerTime)
            # print('orbit parsed')
        return orbit

    #
    # IO stuff, extract data from hdf, read, and write files.
    #

    def _writeMatchFile(self, filenameRoot, data, layer, matchType, tiff=True):
        '''
        Write file with match type for each pixel (GrIMP only)

        Parameters
        ----------
        fileNameRoot : TYPE
            DESCRIPTION.
        data : TYPE
            DESCRIPTION.
        matchType : TYPE
            DESCRIPTION.
        tiff : bool, optional
            Write data to tiff files. The default is True.
        Returns
        -------
        None.

        '''
        mt = np.zeros(data.shape, 'byte')
        mt[data > -2.e8] = matchType
        self._writeImageData(f'{filenameRoot}.layer{layer}.mt', mt, 'u1',
                             tiff=tiff, grimp=True, noDataValue=0)
        # write the mt vrt
        descriptions = [f'matchType {layer}']
        writeMultiBandVrt(f'{filenameRoot}.layer{layer}.mt.vrt',
                          self.OffsetRangeSize, self.OffsetAzimuthSize,
                          [f'{filenameRoot}.layer{layer}.mt'],
                          descriptions,
                          eType=gdal.GDT_Byte,
                          geoTransform=[-0.5, 1., 0., -0.5, 0., 1.],
                          noDataValue=0,
                          tiff=tiff)

    def getProducts(self, productFields):
        '''
        Load the product specified product fields for offsets if not already
        done.

        Parameters
        ----------
        productFields : list of str
            Product fields to load.

        Returns
        -------
        None.
       '''

        for productField in productFields:
            # Get layers if need
            if not hasattr(self, productField):
                #print(f'getting layers for {productField}')
                self.getLayers([productField])

    def getImageData(self, productField, layer=None, useNumpy=True, power=True):
        '''
        Get image data corresponding to productField and create copy as
        self.productField.

        Parameters
        ----------
        productField : str
            Product field to extract (eg. unwrappedPhase).
        layer : str, optional
            Layer key if needed (mostly for offsets). The default is None.

        Returns
        -------
        None.

        '''
        # get bands group
        bands = self.h5[self.product][self.bands]
        if self.product not in ['GCOV', 'RSLC', 'GSLC']:
            if productField not in ['digitalElevationModel']:
                data = \
                    bands[self.frequency][self.productType][self.polarization]
            else:
                # *** THIS IS FOR BACKWARDS COMPATABILITY, CAN BE REMOVED LATER
                if not hasattr(self, 'digitalElevationModel'):
                    setattr(self, productField,
                            np.zeros((getattr(self,
                                              f'{self.lookType}AzimuthSize'),
                                      getattr(self,
                                              f'{self.lookType}RangeSize'))))
                    return
                data = bands[self.frequency][self.productType]
        else:
            data = bands[self.frequency]
        # Apply layer if needed.
        if layer is not None:
            data = data[layer]
        #
        # set data value
        if useNumpy:
            data = np.array(data[productField])  
        else:
            data = data.get(productField)
        #
        if power and data.dtype in ['complex64', 'c8']:
            data = (np.abs(data).astype(np.float32))**2
        #  
        data = self.downsample(data)
        setattr(self, productField, data)

    def getInterferogramPixelOffsets(self):
        '''
        Get pixel offsets for an interogram product
        '''
        bands = self.h5[self.product][self.bands]
        self.pixelOffsets = \
            pixelOffsets(bands[self.frequency]['pixelOffsets'],
                         self.polarization,
                         self.product)
        # print(type(self.pixelOffsets))

    def getLayers(self, layerTypes, noLoadData=False,
                  layers=['layer1', 'layer2', 'layer3']):
        '''
        Get Layers for a given set layerTypes (e.g, ['alongTrackOffset' ...]).

        Save as self.layerType[numberOfLayers, :, :]

        Parameters
        ----------
        layerTypes : list of str
            Names of the layer variables to extract.
        layers : list, optional
            The layers to extract. The default is ['layer1', 'layer2',
                                                   'layer3'].

        Returns
        -------
        None.

        '''
        #print(layers)
        self.layers = [int(x[-1]) for x in layers]
        h5Data = \
            self.h5[self.product][self.bands][self.frequency][self.productType]
        #print(h5Data[self.polarization].keys())
        self.dataFields = []
        for layerType in layerTypes:
            layerData = []
            self.dataFields.append(layerType)
            if not noLoadData:
                for layer in layers:
                    layerData.append(
                        h5Data[self.polarization][layer][layerType])
                setattr(self, layerType, np.stack(layerData))
            else:
                setattr(self, layerType, None)
        #
        if not noLoadData:
            tmp = np.squeeze(
                getattr(self, layerTypes[0])[len(layers) - 1, :, :])
            self.noDataLocations = np.isnan(tmp)
     
    def loadData(self, fields, useNumpy=True, noLoadData=False,
                 resetFields=True, power=False):
        '''
          Load data to np arrays.

        Parameters
        ----------
        fields : list of str
            Field names to sayve.
        useNumpy : bool, optional
            Save data as numpy instead of h5 data. The default is True.
        noLoadData : boold optional
            Don't actually load the data (ie. meta only). The default is False.
        resetFields : bool optional
            Set True to start new field list. Set False when using with after
            getLayers. The default is True.
        Returns
        -------
        None.

        '''
        if resetFields:
            self.dataFields = []
        for field in fields:
            self.dataFields.append(field)
            if noLoadData:
                setattr(self, field, None)
            else:
                self.getImageData(field, useNumpy=useNumpy, power=power)
            #
        #if len(fields) > 0 and not noLoadData:
         #   self.noDataLocations = np.isnan(getattr(self, fields[0]))

    def findNoDataValue(self, band, tiff):
        '''
        Determine no data value from either predefined values or default
        to np.nan (tiff=True) or -2e9 (tiff=False)

        Parameters
        ----------
        band : stra
            band name.
        tiff : bool
            True for tiff file.

        Returns
        -------
        number of appropriate type
            no data value.
        '''
        if tiff:
            if band in self.noDataValuesTiff:
                return self.noDataValuesTiff[band]
            else:
                return np.nan  # Default if not defined
        else:
            if band in self.noDataValuesBinary:
                return self.noDataValuesBinary[band]
            else:
                return -2.e9

    def readGeodatAsDataFrame(self, filename=None):
        '''
        Read Geodat files a pandas data frme

        Parameters
        ----------
        filename : str, optional
            Name of geodat file. The default is None, which then uses
            geodatNumberRangeLooksxNumberAzimuthLooks.geojson

        Returns
        -------
        None.

        '''
        if filename is None:
            filename = f'geodat{self.NumberRangeLooks}x' \
                    f'{self.NumberAzimuthLooks}.geojson'
        #
        self.geoDF = gpd.read_file(filename)

    def writeData(self, filenameRoot, bands=None, tiff=True,
                  byteOrder='LSB', grimp=False,
                  suffixes=None, layers=None,
                  matchTypes=[1, 2, 3], saveMatch=False,
                  scaleToPixels=False,  geojsonName=None,
                  geojsonNameSecondary=None,
                  noSuffix=False, driverName='COG',
                  quickLook=False,
                  sigma0=False,
                  vrtFile=None,
                  scale=1.,
                  vrtOnly=False):
        '''
        Write data to binary or tiff file for all data types. Non offset
        results are saved as individual files (filenameRoot.band[.tif] which
        can be read via a single vrt file (filenameRoot.vrt). If save as tiffs,
        .tif is appended.

        For offsets, files are saved as filenameRoot.layer.suffix[.tif], where
        suffixes are determined by the band to suffix dictionary:
        {'slantRangeOffset': '.dr','slantRangeOffsetVariance': '.vr',
         'alongTrackOffset': '.da', 'alongTrackOffsetVariance': '.va',
         'crossOffsetVariance': '.vc', 'correlationSurfacePeak': '.cc',
         'snr': '.snr'}.

        For either offsets or non-offset data, interpolated fields are saved
        as filenameRoot.fieldName[.tif] (e.g. filenameRoot.incidenceAngle)

        Parameters
        ----------
        filenameRoot : str
            Each band will be save as filenameRoot.band(.tif) along with
            filenameRoot.vrt.
        bands : list of str, optional
            Bands to save (e.g., ['unwrappedPhase]. The default is None,
            which will dump all of the data fields.
        tiff : bool, optional
            Write data to tiff files. The default is True.
        byteOrder : str, optional
            Allow binary file to be written with 'LSB' or 'MSB'. The default
            is 'LSB'.
        grimp : bool, optional
            For GrIMP compatability, force binary files with geocoded data to
            use a lower-left corner geotransform. The default is False.
        noSuffix : bool, optional
            Don't append filename suffix (1 band only). The default is False.
        vrtFile : str, optional
            Override default name for non-offset data. The default is (filenameRoot.vrt).
        Offset Only
        ------------
        suffixes : list of str, optional
            suffixes for output offset fields . The default is ['.da', '.va',
            '.dr', '.va', '.cc', '.snr'].
        layers : list of ints, optional
            Layers to save. The default is [1, 2, 3].
        saveMatch : boolean, optional
            Save a binary match type (mt) file for GrIMP workflow. The
            default is False.
        scaleToPixels : boolean, optional
            Scale offsets to pixel values for GrIMP workflow. The
            default is False.
        geojsonName : str optional
            geojson for redference image for GrIMP workflow. The default is
            None.
        geojsonNameSecondary : str, optional
            geojson for secondary image for GrIMP workflow. The default is
            None.
        driverName : str, optional
            Gdal driver name if writing to tiff. COG or GTiff. The default
            is 'COG'
       quickLook : bool, optional
            write a quickLook png for some image types.            
        Returns
        -------
        None.

        '''
        if bands is None:
            bands = self.dataFields
        if layers is None and hasattr(self, 'layers'):
            layers = self.layers
        # This section makes sure that interpolated fields are written as
        # regular data while offsets with layers are treated specially.

        isData, isDerived = False, False
        for band in bands:
            if band in self.dataFields:
                isData = True
            else:
                isDerived = True
        if isDerived and isData and self.product in ['ROFF', 'GOFF']:
            self.printError('Cannot write offset data and interplated results'
                            ' at the same time')
            return
        # Now write with appropriate routine
        if self.product not in ['ROFF', 'GOFF'] or isDerived:
            self._writeNonOffsetData(filenameRoot,
                                     bands=bands,
                                     tiff=tiff,
                                     grimp=grimp,
                                     byteOrder=byteOrder,
                                     noSuffix=noSuffix,
                                     driverName=driverName,
                                     quickLook=quickLook,
                                     vrtFile=vrtFile, scale=scale,
                                     vrtOnly=vrtOnly)
        else:
            if 'digitalElevationModel' in bands:
                self._writeNonOffsetData(filenameRoot,
                                         bands=['digitalElevationModel'],
                                         tiff=tiff,
                                         grimp=grimp,
                                         byteOrder=byteOrder,
                                         noSuffix=noSuffix,
                                         driverName=driverName,
                                         quickLook=False, 
                                         scale=scale)
            self._writeOffsets(filenameRoot,
                               bands=bands,
                               tiff=tiff,
                               byteOrder=byteOrder,
                               grimp=grimp,
                               suffixes=suffixes,
                               layers=layers,
                               matchTypes=matchTypes,
                               saveMatch=saveMatch,
                               scaleToPixels=scaleToPixels,
                               geojsonName=geojsonName,
                               geojsonNameSecondary=geojsonNameSecondary,
                               noSuffix=noSuffix)

    def _writeOffsets(self, filenameRoot,
                      bands=None,
                      tiff=True,
                      byteOrder='LSB',
                      suffixes=None,
                      grimp=False,
                      layers=None,
                      matchTypes=[1, 2, 3],
                      saveMatch=False,
                      scaleToPixels=False,
                      geojsonName=None,
                      geojsonNameSecondary=None,
                      noSuffix=False):
        '''
        Save offsets as individual layer files with vrt for each layer that
        combines all of the bands.

        Parameters
        ----------
        filenameRoot : str
            Basename for offsets (e.g., offsets.dr, .da...).

        bands : list of str, optional
            Product Fields to save. The default is all.
        tiff : bool, optional
            Write data to tiff files. The default is True.
        byteOrder : str, optional
            Allow binary file to be written with 'LSB' or 'MSB'. The default
            is 'LSB'.
        grimp : bool, optional
            For GrIMP compatability, force binary files with geocoded data to
            use a lower-left corner geotransform. The default is False.
        suffixes : list of str, optional
            suffixes for output offset fields . The default is ['.da', '.va',
            '.dr', '.va', '.cc', '.snr'].
        layers : list of ints, optional
            Layers to save. The default is [1, 2, 3].
        saveMatch : boolean, optional
            Save a binary match type (mt) file for GrIMP workflow. The
            default is False.
        scaleToPixels : boolean, optional
            Scale offsets to pixel values for GrIMP workflow. The
            default is False.
        geojsonName : str optional
            geojson for redference image for GrIMP workflow. The default is
            None.
        geojsonNameSecondary : str, optional
            geojson for secondary image for GrIMP workflow. The default is
            None.
        noSuffix : bool, optional
            Don't append filename suffix (1 band only). The default is False.

        Returns
        -------
        None.

        '''
        # Setup meta data
        if bands is None:
            bands = self.dataFields
        if layers is None:
            layers = self.layers
        self.assembleMeta()
        meta = self.meta.copy()
        meta['bands'] = bands
        meta['ByteOrder'] = byteOrder
        if grimp:
            meta['geo1'] = geojsonName
            meta['geo2'] = geojsonNameSecondary
       
        meta['offsetUnits'] = {True: 'pixels',
                               False: 'meters'}[scaleToPixels]
        # Make sure tiff appended to vrt files.
        tiffSuffix = {True: '.tif', False: ''}[tiff]
        #
        # Loop over layers with and create a vrt for each layer
        for layer, matchType in zip(layers, matchTypes):
            sourceFiles, descriptions, dataTypes, noDataValues = [], [], [], []
            #
            # Lo
            for band in bands:
                if band in ['digitalElevationModel']:
                    continue
                # If one of bands, use predefined suffies otherwise use band

                if band in self.suffixes:
                    suffix = self.suffixes[band]
                else:
                    suffix = band
                # Filename
                if noSuffix:
                    sourceFile = f'{filenameRoot}'
                else:
                    sourceFile = \
                            f'{filenameRoot}.layer{layer}{suffix}{tiffSuffix}' 
                sourceFiles.append(sourceFile)
                descriptions.append(f'{band}')
                # condition data
                data = getattr(self, band)
                # Scale from meters to pixels if requested
                scaleFactor = 1.
                if scaleToPixels and band in self.scaleFactors:
                    scaleFactor = self.scaleFactors[band]
                
                bandData = (np.squeeze(data[layer-1, :, :]) * scaleFactor).astype(np.float32)
                #
                dataTypes.append(gdalTypes[str(bandData.dtype)])
                noDataValues.append(self.findNoDataValue(band, tiff))
                self._writeImageData(sourceFile, bandData,
                                     tiff=tiff,
                                     noDataValue=noDataValues[-1],
                                     dataType=dataTypes[-1],
                                     byteOrder=byteOrder,
                                     grimp=grimp)
                #
                # If along track offsets save match type
                if band == 'alongTrackOffset' and saveMatch:
                    self._writeMatchFile(filenameRoot, bandData, layer,
                                         matchType, tiff=tiff)
            # Save the vrt
            sy, sx = bandData.shape
            epsg = None
            if self.coordType == 'GEO':
                epsg = self.epsg
            #print('Desc', descriptions)
            metaLayer = meta.copy()
            for key in self.windowParams[f'layer{layer}']:
                metaLayer[key] = self.windowParams[f'layer{layer}'][key]
            #print(meta,'\n\n', metaLayer)
            writeMultiBandVrt(f'{filenameRoot}.layer{layer}.vrt',
                              sx, sy,
                              sourceFiles,
                              descriptions,
                              eType=dataTypes,
                              geoTransform=self.getGeoTransform(grimp=grimp,
                                                                tiff=tiff),
                              noDataValue=noDataValues, metaData=metaLayer,
                              byteOrder=byteOrder, tiff=tiff, epsg=epsg)
            #
            if grimp and not tiff:
                self.writeOffsetsDatFile(f'{filenameRoot}.layer{layer}.dat',
                                         geodat1=geojsonName,
                                         geodat2=geojsonNameSecondary)

    def _writeNonOffsetData(self, filenameRoot, bands=None, tiff=True,
                            byteOrder='LSB', grimp=False, noSuffix=False,
                            driverName='COG', quickLook=False,
                            sigma0=False, vrtFile=None, scale=1.,
                            vrtOnly=False):
        '''
        Write non-offset data to binary or tiff file.

        Parameters
        ----------
        filenameRoot : str
            Each band will be save as filenameRoot.band(.tif) along with
            filenameRoot.vrt.
        bands : list of str, optional
            Bands to save (e.g., ['unwrappedPhase]. The default is None,
            which will dump all of the data fields.
        tiff : bool, optional
            Write data to tiff files. The default is True.
        byteOrder : str, optional
            Allow binary file to be written with 'LSB' or 'MSB'. The default
            is 'LSB'.
        grimp : bool, optional
            For GrIMP compatability, force binary files with geocoded data to
            use a lower-left corner geotransform. The default is False.
        quickLook : bool, optional
            Write a quick look png. Overrides tiff flag
        driverName : str, optional
            Gdal driver name if writing to tiff. COG or GTiff. The default
            is 'COG'
        vrtFile : str, optional
            Override default name. The default is (filenameRoot.vrt).
        Returns
        -------
        None.

        '''
        #
        if vrtOnly:
            from nisarhdf.writeVrtOnly import writeVrtOnly as _writeVrtOnly
            if bands is None:
                bands = self.dataFields
            _writeVrtOnly(self, filenameRoot, bands, vrtFile=vrtFile)
            return
        #
        if tiff and byteOrder != 'LSB':
            self.printError(f'Byte order {byteOrder} not supported for tiffs')
            return
        # Add '.tif' suffix to diff files
        if not quickLook:
            suffix = {True: '.tif', False: ''}[tiff]
        else:
            suffix = '.png'
        # Default to all fields as bands
        if bands is None:
            bands = self.dataFields
        self.assembleMeta()
        meta = self.meta.copy()
        meta['bands'] = bands
        meta['ByteOrder'] = byteOrder
        # Results used to create vrt
        sourceFiles, descriptions, dataTypes, noDataValues = [], [], [], []
        # Loop and write each field
        for band in bands:
            filename = {True: f'{filenameRoot}',
                        False: f'{filenameRoot}.{band}{suffix}'}[noSuffix]
            data = getattr(self, band)
            geoTransform = self.getGeoTransform(grimp=grimp, tiff=tiff)
      
            # Save bands nam, file, and data type
            descriptions.append(band)
            sourceFiles.append(filename)
            dataTypes.append(gdalTypes[str(data.dtype)])
            # Get no data value, generally np.nan for tiff and -2e9 for binary
            noDataValue = self.findNoDataValue(band, tiff)
            noDataValues.append(noDataValue)
            # Write the data to a binary or tiff file.
            self._writeImageData(filename, data,
                                 tiff=tiff,
                                 quickLook=quickLook,
                                 noDataValue=noDataValue,
                                 dataType=str(data.dtype),
                                 byteOrder=byteOrder,
                                 grimp=grimp,
                                 driverName=driverName,
                                 geoTransform=geoTransform,
                                 band=band, scale=scale)
        # Write a vrt
        sy, sx = data.shape
        if vrtFile is None:
            vrtFile = f'{filenameRoot}.vrt'
        epsg = None
        if self.coordType == 'GEO':
            epsg = self.epsg
        writeMultiBandVrt(vrtFile,
                          sx, sy,
                          sourceFiles,
                          descriptions,
                          eType=dataTypes,
                          geoTransform=geoTransform,
                          noDataValue=noDataValues, metaData=meta,
                          byteOrder=byteOrder, tiff=tiff, epsg=epsg)

    def _writeImageData(self, filename, data,  byteOrder="LSB",
                        dataType=None, grimp=False, tiff=True, quickLook=False,
                        noDataValue=None, driverName='COG', geoTransform=None,
                        band=None, scale=1.):
        '''
        Call routine to either write data as a tiff or flat binary file.

        Parameters
        ----------
        filename : TYPE
            File name for the data.
        data : np.array()
            Data to save to filename.
        byteOrder : str, optional
            Allow binary file to be written with 'LSB' or 'MSB'. The default
            is 'LSB'.
        grimp : bool, optional
            For GrIMP compatability, force binary files with geocoded data to
            use a lower-left corner geotransform. The default is False.
        tiff : bool, optional
            Write to tiff if True, otherwise flat binary. The default is True.
        noDataValue : numpber, optional
            The no data vaalue. The default is None.
        driverName : str
            Gdal driver name. COG or GTiff

        Returns
        -------
        None.

        '''
        data1 = data
        if scale != 1.:
            print(f'scaling data by {scale}')
            data1 = data * scale
        if not tiff and not quickLook:
            self._writeBinaryImageData(filename, data1,
                                       byteOrder=byteOrder,
                                       dataType=None,
                                       noDataValue=noDataValue,
                                       grimp=grimp)
            # This write geolocation info file for GrIMP workflows
            if grimp and self.coordType == 'GEO':
                self.writeGeodat(f'{filename}.geodat')
        elif tiff and not quickLook:
            self._writeTiffImageData(filename, data1,
                                     dataType=None,
                                     noDataValue=noDataValue,
                                     driverName=driverName,
                                     geoTransform=geoTransform)
        elif quickLook:
            self._writePNG(filename,
                           data1,
                           band=band)
        else:
            print('invalid image format')

    def _writeTiffImageData(self, filename, data, dataType=None,
                            noDataValue=-2.e9, driverName='COG',
                            geoTransform=None, **keywords):
        '''
        Write data as a tiff  file.

        Parameters
        ----------
        filename : TYPE
            File name for the data.
        data : np.array()
            Data to save to filename.
        dataType : str, optional
            Data type (e.g., 'float32'). The default is none, which will
            determine the type directly from the data.
        noDataValue : number, optional
            The no data value. The default is None.
        driverName : str
            Gdal driver name. COG or GTiff
        Returns
        -------
        None.

        '''
        if dataType is None:
            dataType = str(data.dtype)
        # Tiff options
        options = ['COMPRESS=LZW', 'GEOTIFF_VERSION=1.1']
        if self.lookType in ['SLC']:
            options.append('BIGTIFF=YES')
        else:
            options.append('BIGTIFF=NO')
        if dataType != 'complex64' and driverName == 'COG':
            options.append('PREDICTOR=YES')
        if driverName == 'COG':
            options.append('RESAMPLING=AVERAGE')
        # To create COG, first create a version in memory.
        driver = gdal.GetDriverByName('MEM')
        nRows, nColumns = data.shape
        #
        if geoTransform is None:
            geoTransform = self.getGeoTransform(tiff=True)
        dst_ds = driver.Create('', nColumns, nRows, 1, gdalTypes[dataType])
        dst_ds.SetGeoTransform(geoTransform)
        # set projection
        if self.coordType == 'GEO':
            sr = osr.SpatialReference()
            sr.ImportFromEPSG(self.epsg)
            dst_ds.SetProjection(sr.ExportToWkt())
        else:
            data = np.flipud(data)
        #
        # driver specific stuff to initalize band
        dst_ds.FlushCache()
        dst_ds.GetRasterBand(1).WriteArray(data)
        if dataType not in ['complex64']:
            dst_ds.GetRasterBand(1).SetNoDataValue(noDataValue)
        #
        # Create copy for the COG.
        driver = gdal.GetDriverByName(driverName)
        dst_ds2 = driver.CreateCopy(filename, dst_ds, options=options)
        dst_ds2.FlushCache()
        dst_ds.FlushCache()
        del(dst_ds)
        del(dst_ds2)
        gc.collect()
        
    def write_dataset_to_bin(self, data, out_filename,
                         dtype=None, byte_order='LSB',
                         no_data_value=None,
                         chunk_bytes=512 * 1024 * 1024):  # 512 MB chunks
        """
        Efficiently stream an already-open h5py dataset (`data`) to a binary file,
        reading in blocks that are aligned to the dataset's chunk boundaries.
        """

    
        nrows = data.shape[0]
        row_size = np.prod(data.shape[1:]) * data.dtype.itemsize
        dtype = np.dtype(dtype or data.dtype)
    
        # --- Determine the row chunk size from the dataset ---
        if data.chunks is not None:
            row_chunk = data.chunks[0]
        else:
            row_chunk = 1  # contiguous datasets have no chunks
    
        # --- Compute rows_per_chunk aligned to chunk boundaries ---
        rows_per_chunk = max(1, chunk_bytes // row_size)
        # Align to a multiple of row_chunk
        rows_per_chunk = max(row_chunk, (rows_per_chunk // row_chunk) * row_chunk)
        print(f"Writing {nrows} rows in blocks of {rows_per_chunk} rows (aligned to {row_chunk})...")
        with open(out_filename, 'wb') as fout:
            for start in range(0, nrows, rows_per_chunk):
                end = min(nrows, start + rows_per_chunk)
    
                # Adjust start and end to stay on chunk boundaries if possible
                # (except for the last block)
                if end < nrows:
                    end = (end // row_chunk) * row_chunk
    
                block = data[start:end]  # h5py reads efficiently by chunk
                # Convert dtype if needed
                if block.dtype != dtype:
                    block = block.astype(dtype, copy=False)
    
                # Replace NaNs if requested
                if no_data_value is not None and np.issubdtype(dtype, np.floating):
                    np.nan_to_num(block, copy=False, nan=no_data_value)
    
                # Swap byte order if requested
                if byte_order == 'MSB':
                    block.byteswap(True)
                    block = block.view(block.dtype.newbyteorder('='))
    
                # Write binary block
                block.tofile(fout)

                #net_after = psutil.net_io_counters()    
                #bytes_sent = net_after.bytes_sent - net_before.bytes_sent
                #bytes_recv = net_after.bytes_recv - net_before.bytes_recv
                
                
                print(f"\rWrote rows {start}-{end} ({(end / nrows) * 100:.1f}%)", end='', flush=True)
                
        print("\nDone.")


    def _writeBinaryImageData(self, filename, data, dataType=None,
                              byteOrder="LSB", noDataValue=-2.e9, grimp=False, **keywords):
        '''
        write data as a flat binary image with specfied data type.

        Parameters
        ----------
        filename : str
            File name of file data is written to.
        data : np.array()
            The single band of data to save.
        dataType : str, optional
            Data type (e.g., 'float32'). The default is none, which will
            determine the type directly from the data.
        byteOrder : str, optional
            Allow binary file to be written with 'LSB' or 'MSB'. The default
            is 'LSB'.
        noDataValue : TYPE, optional
            DESCRIPTION. The default is -2.e9.
        grimp : bool, optional
            For GrIMP compatability, force binary files with geocoded data to
            use a lower-left corner geotransform so flipud before writing.
            The default is False.
        Returns
        -------
        None.

        '''
        if dataType is None:
            dataType = str(data.dtype)
        #
        if grimp and self.coordType == 'GEO':
            # Flip geocoded images for grimp
            data = np.flipud(data)
            self.writeGeodat(f'{filename}.geodat')
        #
        # Open output
        # Numpy, so just dump whole array
        if isinstance(data, np.ndarray):
            with open(filename, 'wb') as fpOut:
            
                if dataType not in ['complex64']:
                   data[np.isnan(data)] = noDataValue  
                if byteOrder == "MSB":
                    data.byteswap(True)
                    data = data.view(data.dtype.newbyteorder('='))
                data.tofile(fpOut)
            # h5, so write line by line to avoid loading a large h5 all t once
            # This is slow but uses little mem
        else:
            self.write_dataset_to_bin(data,
                                      filename,
                                      dtype=dataType,
                                      byte_order=byteOrder,
                                      no_data_value=noDataValue,
                                      chunk_bytes=2048 * 1024 * 1024)  # GB chunks
              

    def _writePNG(self, filename, data, band=None):
        '''
        Save a numpy array as a PNG using GDAL.
    
        Parameters
        ----------
        filename : str
            Output PNG filename.
        data : np.ndarray
            Input array (2D, float).
        '''
        if data.ndim != 2:
            raise ValueError("Input array must be 2D")
        # If complex use phase for image
        if data.dtype == np.complex64:
           self._writeComplexPNG(filename, data)
        else:
            self._writeScalarPNG(filename, data, band=band)
 
    def _writeComplexPNG(self, filename, data, sqrt=True):
        '''
        Helper for _writePNG to handle complex images.
        
        Parameters
        ----------
        filename : str
            Output PNG filename.
        data : np.ndarray
            Input array (2D, float).
        '''
        # Create alpha channel for no data
        alpha = np.full(data.shape, 255, dtype=np.uint8) 
        alpha[np.isnan(data)] = 0
        # hue from phase
        phase = np.angle(data).astype(np.float32)
        phase[np.isnan(phase)] = 0
        hue = (phase + np.pi) / (2 * np.pi)
        # value from magnitude
        mag = np.abs(data).astype(np.float32)
        if sqrt:
            mag = np.sqrt(mag)
        min_val, max_val = autoScaleRange(mag, 97)
        mag = np.clip(mag, min_val, max_val)
        mag[np.isnan(mag)] = 0
        value = mag/(max_val - min_val)
        # create hsv -> rgb
        hsv = np.stack((hue, np.ones_like(mag), value), axis=-1)
        rgb = (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)
        #
        H, W, _ = rgb.shape
        # PNG needs intermediate MEM driver
        mem_driver = gdal.GetDriverByName("MEM")
        mem_ds = mem_driver.Create("", W, H, 4, gdal.GDT_Byte)
        for i in range(3):
            mem_ds.GetRasterBand(i+1).WriteArray(rgb[:,:,i])
        mem_ds.GetRasterBand(4).WriteArray(alpha)
        # Now translate in-memory to PNG on disk
        png_driver = gdal.GetDriverByName("PNG")
        png_ds = png_driver.CreateCopy(filename, mem_ds)
        png_ds.FlushCache()
        png_ds = None
        mem_ds = None

    def _writeScalarPNG(self, filename, data, band=None):
        '''
        Write a scaler array as PNG.
    
        Parameters
        ----------
        filename : str
            Output PNG filename.
        data : np.ndarray
            Input array (2D, float).
        '''
        arr = data.astype(np.float32)
        # Build alpha channel: transparent for nodata, opaque otherwise
        alpha = np.full(arr.shape, 255, dtype=np.uint8)
        alpha[np.isnan(data)] = 0
        # Scale based on band type
        if band in ['coherenceMagnitude', 'HHHH', 'VVVV', 'HVHV', 'VHVH',
                    'HH', 'VV', 'HV', 'VH']:
            min_val, max_val = autoScaleRange(arr, 97)
        else:
            min_val, max_val = autoScaleRange(arr, 100)
        arr = np.clip(arr, min_val, max_val)
        arr = 255 * (arr - min_val) / (max_val - min_val)
        # Avoid warning with nans to uint8
        arr[np.isnan(data)] = 0
        arr = arr.astype(np.uint8)
        gdal_type = gdal.GDT_Byte
        # Create an in-memory dataset
        mem_driver = gdal.GetDriverByName("MEM")
        mem_ds = mem_driver.Create("", arr.shape[1], arr.shape[0], 2, gdal_type)
        mem_ds.GetRasterBand(1).WriteArray(arr)
        mem_ds.GetRasterBand(2).WriteArray(alpha)
        # Write to PNG using CreateCopy
        png_driver = gdal.GetDriverByName("PNG")
        png_driver.CreateCopy(filename, mem_ds, strict=0)
        mem_ds = None
        #print(f"Saved {filename} (type={gdal_type}, nodata={noDataValue})")

    def removeOutlierOffsets(self, filterField,
                             thresholds=[0.07, 0.05, 0.025],
                             layers=[1, 2, 3]):
        '''
        Remove outliners based on correlation or snr threshold

        Parameters
        ----------
        filterField : str
            Field to use to discard outliners ('correlationSurfacePeak' or
                                               'snr'.
        thresholds : TYPE, optional
            Thresholds use to discard outlier.
            The default is [0.07, 0.05, 0.025],
            which are defaults for correlation.
        layers : list, optional
            List of layers to process. The default is [1, 2, 3].

        Returns
        -------
        None.

        '''
        if self.product not in ['ROFF', 'GOFF']:
            print(f'Product type is {self.product} but Method only works with'
                  'ROFF and GOFF')
            return
        # Load the product fields if not already done
        # Loop over layers
        qaData = getattr(self, filterField)
        azOffsets = getattr(self, 'alongTrackOffset')
        rgOffsets = getattr(self, 'slantRangeOffset')
        for layer in layers:
            qaLayer = qaData[layer-1]
            bad = qaLayer < thresholds[layer-1]
            azOffsets[layer-1][bad] = np.nan
            rgOffsets[layer-1][bad] = np.nan

    def maskPhase(self, largest=True):
        '''
        Mask phase using connected components mask so all zero components
        are discarded.
        largest : Bool
            Retain only the largest connected component.
        Returns
        -------
        None.

        '''
        if self.product not in ['RUNW', 'GUNW']:
            print(f'Product type is {self.product} but Method only works with'
                  'ROFF and GOFF')
            return
        #
        # Ensure data have been extracted
        if not hasattr(self, 'connectedComponents'):
            self.getImageData('connectedComponents')
        if not hasattr(self, 'unwrappedPhase'):
            self.getImageData('unwrappedPhase')
        #
        #  Make a copy
        if not hasattr(self, 'maskedUnwrappedPhase'):
            self.maskedUnwrappedPhase = self.unwrappedPhase.copy()
        # Eliminate phases with connected component values of 0
        print(np.sum(self.connectedComponents < 1))
        #
        self.maskedUnwrappedPhase[self.connectedComponents < 1] = np.nan
        #
        # Retain only the larged connected compnent.
        if largest:
            # Get the nonzero compponent labels
            labels = np.unique(self.connectedComponents)
            labels = labels[np.logical_and(labels > 0, labels < 65535)]
            # Count for each label value
            count = np.array([np.sum(self.connectedComponents == x)
                              for x in labels])
            # find the max
            smallLabels = labels[count < np.max(count)]
            print('smallLabels', smallLabels)
            #
            # null out all the smaller
            for label in smallLabels:
                self.maskedUnwrappedPhase[label ==
                                          self.connectedComponents] = np.nan

    def applyMask(self, maskFile):
        '''
        Apply an external mask (e.g., icemask) file

        Parameters
        ----------
        maskFile : str
            Path to mask file.

        Returns
        -------
        None.

        '''
        if not os.path.exists(maskFile):
            self.printError('applyMask: file ({maskFile} does not exist')
            return

        if not hasattr(self, 'maskedUnwrappedPhase'):
            self.maskedUnwrappedPhase = self.unwrappedPhase.copy()
        #
        mask = rasterio.open(maskFile).read()[0]
        self.maskedUnwrappedPhase[mask < 1] = np.nan

    def ionosphereCorrectPhase(self, masked=False):
        '''
        Correct phase for ionosphere

        Parameters
        ----------
        masked : bool, optional
            Apply the correction to the masked phase. The default is False.

        Returns
        -------
        None.

        '''
        if masked:
            self.ionosphereCorrectedPhase = \
                self.maskedUnwrappedPhase - self.ionospherePhaseScreen
        else:
            self.ionosphereCorrectedPhase = \
                self.unwrappedPhase - self.ionospherePhaseScreen
#
# Date cubes
#

    def incidenceAngleCube(self, x, y, z, maskNoData=None, save=False):
        '''
        Interpolate incidenceAngleCube data cube
        Parameters
        ----------
        x or slantTrange : nparray
            x (G) or slant (R) coordinates.
        y : nparray
            y (G) or slant (ZD) coordinates.
        z : nparray
            Elevation coordinates.
        save: bool, optional
            If true, save the result as self.result (e.g.,
            self.incidenceAngle). The default is False.
        Returns
        -------
        incidenceAngleCube : nparrays
            incidenceAngleCube at desired points

        '''
        if self.product in ['RSLC']:
            print('incidence angle data  not provided with RSLC')
            return
        # Setup interpolators if needed.
        if not hasattr(self, 'incidenceAngleInterpolator'):
            self.setupDataCube('incidenceAngle', zGridPts=None)
        # interpolate
        incidenceAngle = self.interpGrid(self.incidenceAngleInterpolator,
                                         x, y, z, maskNoData=maskNoData)
        # save result for ouput
        if save:
            self.incidenceAngle = incidenceAngle.astype(np.float32)
            self.incidenceAngle[np.isnan(self.incidenceAngle)] = np.nan

        return incidenceAngle

    def elevationAngleCube(self, x, y, z, maskNoData=None, save=False):
        '''
        Interpolate elevationAngle data cube
        Parameters
        ----------
        x or slantTrange : nparray
            x (G) or slant (R) coordinates.
        y : nparray
            y (G) or slant (ZD) coordinates.
        z : nparray
            Elevation coordinates.
        save: bool, optional
            If true, save the result as self.result (e.g.,
            self.elevationAngle). The default is False.
        Returns
        -------
        elevationAngle : nparrays
            elevationAngle at desired points

        '''
        if self.product in ['RSLC']:
            print('elevation angle data  not provided with RSLC')
            return
        # Setup interpolators if needed.
        if not hasattr(self, 'elevationAngleInterpolator'):
            self.setupDataCube('elevationAngle', zGridPts=None)
        # interpolate
        elevationAngle = self.interpGrid(self.elevationAngleInterpolator,
                                         x, y, z, maskNoData=maskNoData)
        # save result for ouput
        if save:
            self.elevationAngle = elevationAngle.astype(np.float32)
        return elevationAngle

    def groundTrackVelocityCube(self, x, y, z, maskNoData=None, save=False):
        '''
        Interpolate groundTrackVelocity data cube
        Parameters
        ----------
        x or slantTrange : nparray
            x (G) or slant (R) coordinates.
        y : nparray
            y (G) or slant (ZD) coordinates.
        z : nparray
            Elevation coordinates.
        save: bool, optional
            If true, save the result as self.result (e.g.,
            self.groundTrackVelocity). The default is False.
        Returns
        -------
        groundTrackVelocity : nparrays
            groundTrackVelocity at desired points

        '''
        if self.product in ['RSLC']:
            print('groundTrackVelocit data  not provided with RSLC')
            return
        # Setup interpolators if needed.
        if not hasattr(self, 'groundTrackVelocityInterpolator'):
            self.setupDataCube('groundTrackVelocity', zGridPts=None)
        # interpolate
        groundTrackVelocity = self.interpGrid(
            self.groundTrackVelocityInterpolator, x, y, z,
            maskNoData=maskNoData)
        # save result for ouput
        if save:
            self.groundTrackVelocity = groundTrackVelocity
        return groundTrackVelocity.astype(np.float32)

    def baselineCube(self, x, y, z, maskNoData=None, save=False):
        '''
        Interpolate baseline data cube

        Parameters
        ----------
        x or slantTrange : nparray
            x (G) or slant (R) coordinates.
        y : nparray
            y (G) or slant (ZD) coordinates.
        z : nparray
            Elevation coordinates.
        save: bool, optional
            If true, save the result as self.result (e.g., self.bn and
            self.bp). The default is False.
        Returns
        -------
        bn, bp : nparrays
            perpendicular and parallel components of baseline

        '''
        if self.product in ['RSLC', 'GCOV']:
            print('baseline data  not provided with RSLC')
            return
        # Setup interpolators if needed.
        if not hasattr(self, 'perpendicularBaselineInterpolator'):
            self.setupDataCube('perpendicularBaseline', zGridPts=[0, -1])
        if not hasattr(self, 'parallelBaselineInterpolator'):
            self.setupDataCube('parallelBaseline', zGridPts=[0, -1])
        # interpolate
        bn = self.interpGrid(self.perpendicularBaselineInterpolator, x, y, z,
                             maskNoData=maskNoData)
        bp = self.interpGrid(self.parallelBaselineInterpolator, x, y, z,
                             maskNoData=maskNoData)
        # save result for ouput
        if save:
            self.bn = bn.astype(np.float32)
            self.bp = bp.astype(np.float32)
        return bn, bp

    def alongTrackUnitVectorCube(self, x, y, z, maskNoData=None, save=False):
        '''
        Interpolate alongTrackUnitVector data cube

        Parameters
        ----------
        x or slantTrange : nparray
            x (G) or slant (R) coordinates.
        y : nparray
            y (G) or slant (ZD) coordinates.
        z : nparray
            Elevatio coordinates.
        save: bool, optional
            If true, save the result as self.result (e.g.,
            self.alongTrackUnitVectorX and self.alongTrackUnitVectorY). The
            default is False.
        Returns
        -------
        ATx, ATY : nparrays
            x and y components of the along track unit vector

        '''
        if self.product in ['RSLC']:
            print('alongTrackUnitVector data  not provided with RSLC')
            return
        # Setup interpolators if needed.
        if not hasattr(self, 'alongTrackUnitVectorXInterpolator'):
            self.setupDataCube('alongTrackUnitVectorX', zGridPts=None)
        if not hasattr(self, 'alongTrackUnitVectorYInterpolator'):
            self.setupDataCube('alongTrackUnitVectorY', zGridPts=None)
        # interpolate
        ATx = self.interpGrid(self.alongTrackUnitVectorXInterpolator, x, y, z,
                              maskNoData=maskNoData)
        ATy = self.interpGrid(self.alongTrackUnitVectorYInterpolator, x, y, z,
                              maskNoData=maskNoData)
        # save result for ouput
        if save:
            self.alongTrackUnitVectorX = ATx.astype(np.float32)
            self.alongTrackUnitVectorY = ATy.astype(np.float32)
        return ATx, ATy

    def losUnitVectorCube(self, x, y, z, maskNoData=None, save=False):
        '''
        Interpolate losUnitVector data cube

        Parameters
        ----------
        x or slantTrange : nparray
            x (G) or slant (R) coordinates.
        y : nparray
            y (G) or slant (ZD) coordinates.
        z : nparray
            Elevation coordinates.
        save: bool, optional
            If true, save the result as self.result (e.g., self.losUnitVectorX,
            self.losUnitVectorY). The default is False.

        Returns
        -------
        LOSx, LOSy : nparrays
             x and y components of the los unit vector

        '''
        if self.product in ['RSLC']:
            print('losUnitVector data  not provided with RSLC')
            return
        # Setup interpolators if needed.
        if not hasattr(self, 'losUnitVectorXInterpolator'):
            self.setupDataCube('losUnitVectorX', zGridPts=None)
        if not hasattr(self, 'losUnitVectorYInterpolator'):
            self.setupDataCube('losUnitVectorY', zGridPts=None)
        # interpolate
        LOSx = self.interpGrid(self.losUnitVectorXInterpolator, x, y, z,
                               maskNoData=maskNoData)
        LOSy = self.interpGrid(self.losUnitVectorYInterpolator, x, y, z,
                               maskNoData=maskNoData)
        # save result for ouput
        if save:
            self.losUnitVectorX = LOSx.astype(np.float32)
            self.losUnitVectorY = LOSy.astype(np.float32)
        return LOSx, LOSy

    def getSquintAnglePolynomial(self, nRangeSamples=9, nAzimuthSamples=9):
        '''
        Fit squint(r, a) = c0 + c1*r' + c2*a' + c3*a'^2 + c4*r'*a' +
        c5*r'^2 (r' = slantRange - MLCenterRange, a' = zeroDopplerTime -
        MLMidZeroDopplerTime) from the geolocationGrid cube's
        losUnitVector/alongTrackUnitVector, and save the result as
        self.squintAnglePolynomial = {'coefficients': [c0..c5],
        'refRange': ..., 'refAzimuthTime': ...}.

        Returns
        -------
        None.

        '''
        # Sampling grid over this sub-frame's range/azimuth extent
        rangeSamples = np.linspace(self.MLNearRange, self.MLFarRange,
                                   nRangeSamples)
        azimuthSamples = np.linspace(self.MLFirstZeroDopplerTime,
                                     self.MLLastZeroDopplerTime,
                                     nAzimuthSamples)
        R, A = np.meshgrid(rangeSamples, azimuthSamples, indexing='ij')
        # Squint's height-sensitivity is negligible (see
        # nisarErrors/Documents/plotSquintError.md), so sample at a single
        # fixed height -- the geolocationGrid cube's own mid-level height,
        # which is guaranteed in-bounds for the interpolator.
        heightAboveEllipsoid = np.array(
            self.h5[self.product]['metadata']['geolocationGrid'][
                'heightAboveEllipsoid'])
        Z = np.full(R.shape,
                   heightAboveEllipsoid[len(heightAboveEllipsoid) // 2])
        # Sample los/along-track unit vectors and compute squint (deviation
        # of the los heading from broadside relative to the along-track
        # heading; L flips the nominal +/-90 deg offset for look direction)
        losX, losY = self.losUnitVectorCube(R, A, Z)
        atX, atY = self.alongTrackUnitVectorCube(R, A, Z)
        losHeading = np.degrees(np.arctan2(losX, losY)) % 360
        atHeading = np.degrees(np.arctan2(atX, atY)) % 360
        diff = ((losHeading - atHeading + 180) % 360) - 180
        L = 1.0 if self.LookDirection.lower() == 'left' else -1.0
        squint = diff - 90.0 * L
        # Center range/azimuth on the frame's existing reference point for
        # numerical conditioning (raw slant range ~1e6 m, raw zero-Doppler
        # time ~1e4 s of day are poorly conditioned for a quadratic fit)
        rPrime = (R - self.MLCenterRange).flatten()
        aPrime = (A - self.MLMidZeroDopplerTime).flatten()
        designMatrix = np.array([rPrime * 0 + 1, rPrime, aPrime, aPrime**2,
                                 rPrime * aPrime, rPrime**2]).T
        coefficients, _, _, _ = np.linalg.lstsq(
            designMatrix, squint.flatten(), rcond=None)
        self.squintAnglePolynomial = {
            'coefficients': list(coefficients),
            'refRange': self.MLCenterRange,
            'refAzimuthTime': self.MLMidZeroDopplerTime,
            }

    # Gelocaton code that should not be needed given geolocation cubes
    # Most is GrIMP specfic except what is need for
    # self.SLCSceneCenterAlongTrackSpacing

    def computeAngles(self, slantRange, zeroDopplerTime, zWGS84,
                      degrees=False):
        '''
        Compute elevation angle for slantRange, zeroDoppler, zWGS84.

        Note this elevation angle is in the zero-Doppler plane orthogonal to
        the satellite track (the dataCube elevation angle includes the squint)

        Parameters
        ----------
        slantRange : float
            slant range.
        zeroDopplerTime : floate
            zero Doppler time.
        z : float
            Elevation (WGS84 ellipsoid) of point .
        degrees : TYPE, optional
            Return values degrees. The default is False.

        Returns
        -------
        elevationAngle : float
            Elevation angle for point.
        incidenceAngle : float
            Incidence angle for point.

        '''
        lat, lon, z = self.RTtoLatLon(slantRange,  zeroDopplerTime, zWGS84)
        position, velocity = self.getSatPositionAndVel(zeroDopplerTime)
        ReH = np.linalg.norm(position)
        #
        if not hasattr(self, 'LLTOECEF'):
            self.LLtoECEF = pyproj.Transformer.from_crs("EPSG:4326",
                                                        "EPSG:4978",).transform
        #
        ptRadius = np.linalg.norm(self.LLtoECEF(lat, lon, z))
        #
        elevationAngle = np.arccos((slantRange**2 + ReH**2 - ptRadius**2) /
                                   (2.0 * slantRange * ReH))
        incidenceAngle = np.arcsin(np.sin(elevationAngle) * ReH/ptRadius)
        #
        if degrees:
            return np.degrees(elevationAngle), np.degrees(incidenceAngle)
        return elevationAngle, incidenceAngle

    def adjust_zdTime(self, zdTime):
        zd = np.asarray(zdTime)
    
        first = zd.item() if zd.ndim == 0 else zd[0]
    
        if first < 10000 and self.orbit.TimeOfFirstStateVector > 80000:
            zd = zd + 86400
    
        if np.isscalar(zdTime):
            return zd.item()
        elif isinstance(zdTime, list):
            return zd.tolist()
        else:
            return zd
        
    def getSatPositionAndVel(self, zdTime):
        '''
        Get satellite position for given zdTime

        Parameters
        ----------
        zdTime : float
            Zero Doppler time

        Returns
        -------
        None.
        '''
        # State vectors that straddle midnight exceed 86400, while the 
        # zdTime might start just after midnight, this will force correct
        # interpolation in this case
        zdTime1 = self.adjust_zdTime(zdTime)
        position = [self.orbit.xsv([zdTime1])[0],
                    self.orbit.ysv([zdTime1])[0],
                    self.orbit.zsv([zdTime1])[0]]
        velocity = [self.orbit.vxsv([zdTime1])[0],
                    self.orbit.vysv([zdTime1])[0],
                    self.orbit.vzsv([zdTime1])[0]]
        return position, velocity

    def getSatelliteHeight(self, zdTime):
        '''
        Compute satellite height along track

        Parameters
        ----------
        zdTime : list or np.array of points
            Zero Doppler times at which to compute the satellite height.

        Returns
        -------
        hSat : list or np.array of points
            Heights corresponding to zero Doppler times.

        '''
        if not hasattr(self, 'ECEFtoLL'):
            self.ECEFtoLL = pyproj.Transformer.from_crs("EPSG:4978",
                                                        "EPSG:4326").transform
        zdTime1 = self.adjust_zdTime(zdTime)
        # Get the ECEF coords
        x = self.orbit.xsv(zdTime1)
        y = self.orbit.ysv(zdTime1)
        z = self.orbit.zsv(zdTime1)
        # Convert to geodetic lat, lon, height above ellipsoid
        latSat, lonSat, hSat = self.ECEFtoLL(x, y, z)
        return hSat

    def RTtoLatLon(self, slantRange, zdTime, z):
        '''
        Convert R/D coords to lat lon at a point. In most cases shouldn't be
        needed because this can be acomplished faster with the data cube.

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
        if not hasattr(self, 'ECEFtoLL'):
            self.ECEFtoLL = pyproj.Transformer.from_crs("EPSG:4978",
                                                        "EPSG:4326").transform
        #
        # User position/velocity state vector interpolation for time
        position, velocity = self.getSatPositionAndVel(zdTime)
        # Return lat/lon/z for slantRange, z
        return self.SMLocateZD(position, velocity, slantRange, z)

    def RTtoECEF(self, slantRange, zdTime, z):
        '''
        Convert R/D coords to lat lon at a point. In most cases shouldn't be
        needed because this can be acomplished faster with the data cube.

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
        if not hasattr(self, 'ECEFtoLL'):
            self.ECEFtoLL = pyproj.Transformer.from_crs("EPSG:4978",
                                                        "EPSG:4326").transform
        #
        # User position/velocity state vector interpolation for time
        position, velocity = self.getSatPositionAndVel(zdTime)
        # Return lat/lon/z for slantRange, z
        return self.SMLocateZDECEF(position, velocity, slantRange, z)

    def SMLocateZDECEF(self, position, velocity, slantRange, z):
        '''
        Based on old Soren Madsen algorithm to compute lat/lon from slant
        range and altitude given sat positoin

        Parameters
        ----------
        position : 3 element vector
            Satellite ECEF coordinates.
        velocity : 3 element vector
            Satellite velocity.
        slantRange : float
            Slant range.
        z : float
            WGS84 elevation

        Returns
        -------
        x, y, z
            geocentric coordinates

        '''
        #
        # Calculate the vertical unit vector
        Rsat = np.linalg.norm(position)
        nhat = position / Rsat
        #
        # Calculate the horizontal component of the unit pointing vector
        # which is orthogonal to the velocity vector
        ph = np.cross(velocity, nhat)
        phhat = ph / np.linalg.norm(ph)
        #
        # Calculate the vertical component of the unit pointing vector
        # which is perpendicular to the velocity vector
        pv = np.cross(velocity, phhat)
        pvhat = pv / np.linalg.norm(pv)
        #
        #  Calculate effective earth radius and satellite altitude
        posxy, posz = position[0:2], position[2]
        reff = np.sqrt(Rsat**2 /
                       (np.dot(posxy, posxy) / self.EarthRadiusMajor**2 +
                        (posz**2) / self.EarthRadiusMinor**2))
        #
        # Compute the approximate look angle using reff at target loc
        temp = (Rsat**2.0 + slantRange**2 - (reff + z)**2) / \
            (2.0 * Rsat * slantRange)
        elook = self.lookSign * np.arccos(temp)
        #
        # Minimize look error function
        ReMajorZ2 = (self.EarthRadiusMajor + z)**2
        ReMinorZ2 = (self.EarthRadiusMinor + z)**2
        # Function to optimize
        def lookOptimize(elook, phhat, pvhat, position, slantRange,
                         ReMajorZ2, ReMinorZ2):
            return self._computeLookError(elook, phhat, pvhat, position,
                                          slantRange, ReMajorZ2, ReMinorZ2)[0]
        #
        elook = optimize.leastsq(lookOptimize, elook,
                                 args=(phhat, pvhat, position, slantRange,
                                       ReMajorZ2, ReMinorZ2))[0][0]
        #
        # Target location using latest final pointing vector
        # compute final values
        p = np.sin(elook) * phhat + np.cos(elook) * pvhat
        t = position + slantRange * p
        # return location
        return t

    def SMLocateZD(self, position, velocity, slantRange, z):
        '''
        Based on old Soren Madsen algorithm to compute lat/lon from slant
        range and altitude given sat positoin

        Parameters
        ----------
        position : 3 element vector
            Satellite ECEF coordinates.
        velocity : 3 element vector
            Satellite velocity.
        slantRange : float
            Slant range.
        z : float
            WGS84 elevation

        Returns
        -------
        lat, lon, z
             coordinates

        '''
        t = self.SMLocateZDECEF(position, velocity, slantRange, z)
        return self.ECEFtoLL(*list(t))

    def downsample(self, arr):
        '''
        Downsample arr by factor, which will be forced to be integere

        Parameters
        ----------
        arr : np.array
            array to downsample.
        factor : int
            Downsample factor.

        Returns
        -------
        np.array
            Downsampled array.

        '''
        if self.downsampleFactorColumn == 1 and self.downsampleFactorRow == 1:
            return arr
        m, n = arr.shape
        # print(m, n)
        mNew, nNew = m // self.downsampleFactorRow, n // self.downsampleFactorColumn
        #
        # print(mNew, nNew)
        return arr[:mNew*self.downsampleFactorRow,
                   :nNew*self.downsampleFactorColumn,].reshape(mNew, self.downsampleFactorRow,
                                                            nNew, self.downsampleFactorColumn).mean(axis=(1, 3))

    def rescale_geoTransform(self, gt, rowFactor, colFactor):
        """
        Modify a GDAL GeoTransform after downsampling by an integer factor.

        Parameters
        ----------
        gt : tuple or list
            Original geotransform (6 elements)
        factor : int
            Downsampling factor
        Returns
        -------
        tuple
            Updated geotransform
        """
        gt0, gt1, gt2, gt3, gt4, gt5 = gt
        return (
            gt0,           # origin x stays the same
            gt1 * colFactor,  # pixel width increases
            gt2,           # rotation unchanged
            gt3,           # origin y stays the same
            gt4,           # rotation unchanged
            gt5 * rowFactor   # pixel height increases (still neg. for north-up)
        )

    def _computeLookError(self, elook, ph, pv, position, slantRange,
                          ReMajorZ2, ReMinorZ2):
        '''
        Compute look angle error for lat/lon determination

        Parameters
        ----------
        elook : float
            current look angle.
        ph : 3 element vector
            horizontal component of the unit pointing vector.
        pv : 3 element vector
            vertical component of the unit pointing vector..
        position : 3 element vector
            satellite position
        slantRange : float
            Slant range.
        ReMajorZ2 : float
            (ReMajor + z)**2.
        ReMinorZ2 : float
            (ReMajor + z)**2.
        Returns
        -------
        delook : float
            Look angle error.
        t : 3 element vector
            target location.
        dt : 3 element vector
            Differential position.

        '''
        #
        # Latest pointing vector
        p = np.sin(elook) * ph + np.cos(elook) * pv
        #
        # Target location using latest look pointing vector
        t = position + slantRange * p
        txy, tz = t[0:2], t[2]
        #
        # Error to minimize
        error = 1.0 - np.dot(txy, txy) / ReMajorZ2 - (tz**2) / ReMinorZ2
        return error, t

    def close(self):
        try:
            self.h5Full.close()
        except Exception:
            pass
        try:
            if self.memory_file is not None:
                self.memory_file.close()
        except Exception:
            pass
        self.h5Full = None
        self.memory_file = None

    def __del__(self):
        self.close()  # Safe way to ensure cleanup if forgotten
