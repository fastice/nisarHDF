#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:44:13 2024

@author: ian
"""
from nisarhdf.nisarBaseRangeDopplerHDF import nisarBaseRangeDopplerHDF
import os
import numpy as np
from scipy import ndimage

from nisarhdf import writeMultiBandVrt, formatGeojson

class nisarRUNWHDF(nisarBaseRangeDopplerHDF):
    '''
    This class creates objects to work with nisar RUNWHDF images.
    '''

    def __init__(self,  sar='LSAR', frequency='frequencyA',
                 polarization=None, isSecondary=False, 
                 productType='interferogram',
                 referenceOrbitXML=None, secondaryOrbitXML=None, debug=False):
        '''
       sar : str, optional
            SAR Idendtifier (always LSAR for now). The default is 'LSAR'.
        frequency : str, optional
            frequency band to extract. The default is 'frequencyA'.
        polarization : str, optional
            Polarization. The default is 'HH'.
        isSecondary : Boolean, optional
            For internal use only. The default is False.
        referenceOrbitXML : str, optional
            XML file to override orbit in hdf The default is None.
        secondaryOrbitXML : TYPE, optional
            XML file for secondary orbit. The default is None

        Returns
        -------
        None.

        '''
        # print('ref orbit', referenceOrbitXML)
        nisarBaseRangeDopplerHDF.__init__(self,
                                          sar=sar,
                                          product='RUNW',
                                          frequency=frequency,
                                          productType=productType,
                                          polarization=polarization,
                                          layer=None,
                                          productData='unwrappedPhase',
                                          bands='swaths',
                                          isSecondary=isSecondary,
                                          referenceOrbitXML=referenceOrbitXML,
                                          secondaryOrbitXML=secondaryOrbitXML,
                                          debug=debug)
        self.lookType = 'ML'
        self.productParams = ['NumberRangeLooks', 'NumberAzimuthLooks']
        for param in self.RDParams:
            self.productParams.append(f'{self.lookType}{param}')

    def parseParams(self, fields=None, productType='interferogram', 
                    polarization=None, secondary=False, 
                    noLoadData=False, **keywords):
        '''
        Parse all the params needed to make a geodatNRxNA.geojson file

        Returns
        -------
        None.

        '''
        if not secondary:
            self.getPolarization(polarization)
        self.getGranuleNames()
        self.getOrbitAndFrame(**keywords)
        self.getLookDirection()
        self.parseRefDate()
        self.getNumberOfLooks()
        self.getSLCSlantRange()
        self.getSLCZeroDopplerTime(secondary=secondary)
        self.getMLSize()
        self.getMLSlantRange()
        self.effectivePRF()
        self.getRangeErrorCorrection()
        if not secondary:
            self.getInterferogramPixelOffsets()
        else:
            self.ImageName = 'secondary'
        self.getMLZeroDopplerTime(secondary=secondary)
            #
        self.getCorrectedTime()
        # Note if secondary, it will have been passed the reference (see below)
        self.orbit = self.parseStateVectors(XMLOrbit=self.referenceOrbitXML)
        #
        self.getOrbitPassDirection()
        self.getCorners()
        self.getRangeErrorCorrection()
        self.getTimeToFirstSample()
        self.getSkewOffsets()
        self.getCenterIncidenceAngle()
        self.getSquint(secondary=secondary)
        self.getDeltaT()
        self.getCenterLatLon()
        self.getSceneCenterSatelliteHeight()
        #self.getOrbitPassDirection()
        self.getWavelength()
        #self.getSingleLookPixelSize()
        self.getExtent()
        #
        # If not secondary, create the secondary and parse
        if not secondary:
            self.secondary = \
                nisarRUNWHDF(isSecondary=True,
                             referenceOrbitXML=self.secondaryOrbitXML,
                             debug=self.debug)
            self.secondary.h5 = self.h5
            self.secondary.CRID = getattr(self, 'CRID', None)
            self.secondary.parseParams(secondary=True, referenceOrbit=self.secondaryOrbit)
            #self.secondaryDatetime = self.secondary.datetime
            #self.secondaryDate = self.secondary.datetime
            self.dT = np.round((self.secondaryDatetime -
                               self.datetime).total_seconds()/86400.,
                               decimals=3)
            self.secondary.referenceOrbit = self.secondaryOrbit
            self.secondary.frame = self.frame
            self.genGeodatProperties()
            if fields is None:
                fields = ['coherenceMagnitude',
                          'connectedComponents',
                          'ionospherePhaseScreen',
                          'ionospherePhaseScreenUncertainty',
                          'unwrappedPhase',
                          'digitalElevationModel']
            self.loadData(fields, noLoadData=noLoadData)

    def getSquint(self, secondary=False):
        '''
        Leave the legacy scalar Squint field untouched (older code uses it
        for a deskew time delay to zero Doppler -- a different quantity
        from the residual squint *angle* fit below, so it must not be
        repurposed) and, for the reference image only, fit the real
        squintAnglePolynomial from the geolocationGrid cube. The secondary
        image shares the reference's HDF5 file and geolocationGrid cube
        (see self.secondary.h5 = self.h5 above), whose zero-Doppler time
        axis only spans the reference pass's absolute time -- sampling it
        at the secondary's own (different-day) azimuth times would fall
        outside the cube and return NaN, so the secondary keeps
        squintAnglePolynomial=None.

        Returns
        -------
        None.

        '''
        nisarBaseRangeDopplerHDF.getSquint(self)
        if secondary:
            return
        self.getSquintAnglePolynomial()

    def cleanIonosphere(self, edge_mask_px=80, iterations_low=250,
                        downsample=8, sigma_az=150, sigma_rg=10):
        """
        Complete pipeline: 
        1. Masking & Erosion
        2. Multi-scale Inpainting (removes blocks/gaps)
        3. Anisotropic Gaussian Smoothing (removes glacier leakage)
        """
        arr = np.asanyarray(self.ionospherePhaseScreen).astype(np.float32) 
        # 1. Internal Mask Generation
        # Valid = not NoData, not NaN
        mask = np.isfinite(arr) & (arr != -2000000000)
     
        if edge_mask_px > 0:
            print(f"Eroding {edge_mask_px} pixels from edges...")
            mask = ndimage.binary_erosion(mask, iterations=edge_mask_px)  
        # if erosion removes to much proceed with uneroded mask
        if not np.any(mask):
            mask = np.isfinite(arr) & (arr != -2000000000)
        # 2. Low-Resolution Pass (Global Trend)
        print(f"Downsampling by {downsample}x for global continuity...")
        low_arr = ndimage.zoom(arr, 1.0/downsample, order=1)
        low_mask = ndimage.zoom(mask.astype(np.float32), 
                                1.0/downsample, 
                                order=0) > 0.5
        
        # Initialize low-res gaps with median
        fill_val = np.median(low_arr[low_mask])
        filled_low = np.where(low_mask, low_arr, fill_val)
        
        kernel = np.array([[0.25, 0.5, 0.25], [0.5, 0, 0.5], [0.25, 0.5, 0.25]])
        kernel /= kernel.sum()
        
        for _ in range(iterations_low):
            smoothed = ndimage.convolve(filled_low, kernel, mode='reflect')
            filled_low = np.where(low_mask, low_arr, smoothed)
        
        # 3. Upsample and Refine
        print("Upsampling and refining at full resolution...")
        zoom_factors = [o / l for o, l in zip(arr.shape, filled_low.shape)]
        smart_fill = ndimage.zoom(filled_low, zoom_factors, order=1)
        
        # Ensure exact shape match
        smart_fill = smart_fill[:arr.shape[0], :arr.shape[1]]
        
        # Start high-res with the 'smart' background
        final_filled = np.where(mask, arr, smart_fill)
        
        # Quick polish to stitch edges perfectly
        for _ in range(30):
            smoothed = ndimage.convolve(final_filled, kernel, mode='reflect')
            final_filled = np.where(mask, arr, smoothed)
        
        # 4. Anisotropic Gaussian Smoothing
        # This specifically targets the 'sinuous' glacier leakage
        print(f"Applying directional blur (Az={sigma_az}, Rg={sigma_rg})...")
        self.ionosphereCleaned = ndimage.gaussian_filter(final_filled, 
                                                         sigma=(sigma_az,
                                                                sigma_rg))

