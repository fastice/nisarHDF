#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 15:38:36 2025

@author: ian
"""
import argparse
import nisarhdf
from datetime import datetime
import sys
import psutil
import tracemalloc
import atexit

productTypes = ['RSLC', 'ROFF', 'RIFG', 'RUNW', 'GCOV', 'GUNW', 'GOFF' ]

# Dictionary with information on other fields
fieldsDict = {'ROFF': {'pixelOffsets': ['slantRangeOffset',
                                        'slantRangeOffsetVariance',
                                        'alongTrackOffset', 
                                        'alongTrackOffsetVariance',
                                        'crossOffsetVariance', 
                                        'correlationSurfacePeak',
                                        'snr', 
                                        'digitalElevationModel']},
              'RIFG': {'interferogram': ['wrappedInterferogram',
                                         'coherenceMagnitude']},
              'RUNW': {'interferogram': ['unwrappedPhase',
                                         'coherenceMagnitude',
                                         'connectedComponents', 
                                         'ionospherePhaseScreen',
                                         'ionospherePhaseScreenUncertainty',
                                         'digitalElevationModel']},
              'RSLC': {None: ['HH', 
                              'VV',
                              'HV', 
                              'VH']},
              'GCOV': {None: ['HHHH', 
                              'VVVV',
                              'HVHV', 
                              'VHVH',
                              'mask',
                              'numberOfLooks',
                              'rtcGammaToSigmaFactor']},
              'GUNW': {'unwrappedInterferogram': ['unwrappedPhase',
                                                  'coherenceMagnitude',
                                                  'connectedComponents',
                                                  'ionospherePhaseScreen',
                                                  'ionospherePhaseScreenUncertainty'], 
                       'wrappedInterferogram': ['wrappedInterferogram',
                                                'coherenceMagnitude']}, 
              'GOFF': {'pixelOffsets': ['slantRangeOffset',
                                        'slantRangeOffsetVariance',
                                        'alongTrackOffset',
                                        'alongTrackOffsetVariance',
                                        'crossOffsetVariance', 
                                        'correlationSurfacePeak',
                                        'snr']}
              }

productTypeDict = {'ROFF': ['pixelOffsets'],
                    'RIFG': ['interferogram'],
                    'RUNW': ['interferogram'],
                    'RSLC': [None],
                    'GCOV':[None],
                    'GUNW': ['unwrappedInterferogram', 'wrappedInterferogram'],  
                    'GOFF': ['pixelOffsets']
                    }

defaultFieldsDict = {'RSLC': {None: ['HH', 'VV', 'HV', 'VH']},
                     'ROFF': {'pixelOffsets': ['slantRangeOffset', 
                                               'alongTrackOffset']},
                     'RIFG': {'interferogram': ['wrappedInterferogram']},
                     'RUNW': {'interferogram': ['unwrappedPhase']},
                     'GCOV': {None: ['HHHH', 'VVVV', 'HVHV', 'VHVH']},
                     'GUNW': {'unwrappedInterferogram': ['unwrappedPhase'], 
                              'wrappedInterferogram': ['wrappedInterferogram']}, 
                     'GOFF': {'pixelOffsets': ['slantRangeOffset',
                                               'alongTrackOffset']}
                     
                    }

quickLookDefaultDownsampleFactors = {'RSLC': 20,
                                     'ROFF': 1, # not used
                                     'RIFG': 8,
                                     'RUNW': 2,
                                     'GCOV': 8,
                                     'GUNW': 2, 
                                     'GOFF': 2} # not use
    
def specialKeywords(myArgs, applicableProducts, specialArgs, args, 
                    default=False):
    '''
    Setup product specific arguments

    Parameters
    ----------
    productType : TYPE
        DESCRIPTION.
    applicableProducts : TYPE
        DESCRIPTION.
    keyword : TYPE
        DESCRIPTION.
    args : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    for arg in specialArgs:
        if myArgs['product'] in applicableProducts:
            myArgs[arg] = getattr(args, arg)
        else:
            myArgs[arg] = default
            #print(f'--{arg} only for {applicableProducts}, ignoring')
            

def myerror(message,myLogger=None):
    """ print error and exit with a FAILING status (1).

    Local copy of utilities.myerror; was a bare sys.exit(), i.e. status 0,
    which made fatal errors look like success to any caller checking a
    return code.
    """
    print('\n\t\033[1;31m *** ',message,' *** \033[0m\n')
    if myLogger != None :
        myLogger.logError(message)
    sys.exit(1)

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be an integer > 0")
    return ivalue

def human_size(num_bytes):
    '''
    Convert num_bytes to format like ls -h
    '''
    for unit in ["B","K","M","G","T","P","E"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:6.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:7.1f}Z"  # fall back for huge numbers
    
def parseCommandLine():
    '''
    Handle command line args
    '''
    parser = argparse.ArgumentParser(
        description='\n\n\033[1mConvert NISAR H5 product to Tiff(s), COG(s) or binary files'
        '\033[0m\n\n',
        epilog='Part of the nisarhdf package.')
    parser.add_argument('productName', type=str,
                        help='NISAR h5 product (file path, s3, or https)')
    parser.add_argument('output', type=str, nargs="?",
                        help='Root name for ouput. Band (e.g., .unwrappedPhase) and file type (e.g., .tiff) will be appended. (Not required for --info)')
    parser.add_argument('--info', action="store_true",
                        help='Print summary info only')
    parser.add_argument('--quickLook', action="store_true",
                        help='Write a quick look PNG file (GCOV )')
    parser.add_argument('--ros3', action="store_true",
                        help='Use ros3 rather than reading full file to memory for s3 and some https. '
                        'Slower but uses less memory')  
    parser.add_argument('--conserveMem', action="store_true",
                        help='Save memory by not converting HDF to numpy at the expense of speed')  
    parser.add_argument('--productFamily', type=str, default='',
                        help='NISAR product type (e.g., RUNW, GUNW etc) '
                        '[parse from product path]',
                        choices=productTypes)
    parser.add_argument('--frequencyB', action="store_true",
                        help='Select frequencyB [frequencyA] ')    
    parser.add_argument('--dB', action="store_true",
                        help='Output results in dB (GCOV only)')  
    parser.add_argument('--noSuffix', action="store_true",
                        help='Do not include suffix if only one field')  
    parser.add_argument('--sigma0', action="store_true",
                        help='Output results as sigma0 (GCOV only) [gamma0]')  
    parser.add_argument('--scaleToPixels', action="store_true",
                        help='Scale offsets to pixels (ROFF, GOFF only)')  
    parser.add_argument('--scale', type=float, default=1.,
                        help='Scale Factor')  
    parser.add_argument('--wrapped', action="store_true",
                        help='Wrapped interferogram (GUNW only)')  
    parser.add_argument('--MSB', action="store_true",
                        help='Write data with MSB byte order')  
    parser.add_argument('--polarization', type=str, default=None,
                        choices=['HH', 'VV', 'HV', 'VH'],
                        help='Polarization for \n\n\033[1mnon-GCOV\n\n\033[0m '
                        'products [first like pol]')
    parser.add_argument('--downsampleFactor', type=positive_int, default=None,
                        help='Reduce resolution by 1/subsampleFactor (integer > 0) [full res]')
    parser.add_argument('--fields', type=str, default=[], nargs='+',                       
                        help='Select fields including GCOV covariance terms (e.g., '
                        'HHHH, HVHV), use all for everything. '
                        'Use --info to see defaults for product type [defaults]')
    parser.add_argument('--layers', type=str, default=['1', '2', '3'], nargs='+',
                        choices=['1', '2', '3'],                    
                        help='For offsets only, layer numbers to access [1-3]')
    parser.add_argument('--outputFormat', type=str, default='COG',
                        choices=['COG', 'GTiff', 'binary'],
                        help='Output format [COG: cloud optimized geotiff]')
    parser.add_argument('--vrtOnly', action="store_true",
                        help='Write a VRT pointing directly to HDF5 bands; '
                             'no data is extracted [False]')
    #
    args = parser.parse_args()
    #
    myArgs = {}
    #
    if args.output is None and not args.info:
        myerror('Unless --info set, specify an output file name')
    #
    myArgs['product'] = getProductType(args.productFamily, 
                                       args.productName)
    #
    specialKeywords(myArgs, ['GCOV'], ['dB', 'sigma0'], args)
    specialKeywords(myArgs, ['ROFF', 'GOFF'], ['scaleToPixels'], args)
    specialKeywords(myArgs, ['GUNW'], ['wrapped'], args)
    myArgs['frequency'] = {False: 'frequencyA',
                           True: 'frequencyB'}[args.frequencyB]
    myArgs['byteOrder'] = {False: 'LSB',
                           True: 'MSB'}[args.MSB]
    #
    processHDFOpenKeywords(args, myArgs)
    #
    myArgs['productType'] = productTypeDict[myArgs['product']][0]
    if myArgs['wrapped']:
        myArgs['productType'] = productTypeDict[myArgs['product']][1]
    #
    myArgs['fields'] = reconcileFields(args, myArgs)
    #
    # args that need no checking
    for arg in ['productName', 'output', 'polarization',
                'outputFormat', 'info', 'ros3',
                'downsampleFactor', 'quickLook', 'noSuffix', 'scale',
                'vrtOnly']:
        myArgs[arg] = getattr(args, arg)
    # Force 'ros3' for info
    if myArgs['info']:
        myArgs['ros3'] = True
    #
    myArgs['power'] = False
    if myArgs['quickLook']:
        if  myArgs['product'] in ['RSLC']:
            myArgs['power'] = True
        # No quick look support yet for these products
        if myArgs['product'] in ['ROFF', 'GOFF']:
            print(f'quickLooks not supported for ROFF and GOFF')
            myArgs['quickLook'] = False
        # If not specified, use default
        elif myArgs['downsampleFactor'] is None:
            myArgs['downsampleFactor'] = quickLookDefaultDownsampleFactors[myArgs['product']]
            # for GUNW wrapped use 4x large
            if myArgs['product'] in ['GUNW'] and myArgs['wrapped']:
                 myArgs['downsampleFactor'] *= 4
    # None quicklook default
    elif myArgs['downsampleFactor'] is None:
        # If not set by one of the defaults, default to 1
        myArgs['downsampleFactor'] = 1
    #
    myArgs['downsampleFactor'] = {'downsampleFactorRow': myArgs['downsampleFactor'],
                                  'downsampleFactorColumn': myArgs['downsampleFactor']}    
    #
    return myArgs

def  processHDFOpenKeywords(args, myArgs):
    '''
    Process the set of keywords that are passed to the hdfOpen method
    '''
    myArgs['hdfOpenKeywords'] = {}
    # keywords that only apply to GCOV
    if myArgs['product'] in ['GCOV']:
        myArgs['hdfOpenKeywords']['dB'] = myArgs['dB']
        myArgs['hdfOpenKeywords']['sigma0'] = myArgs['sigma0']
        if args.polarization is not None:
            myerror('--polarization is not valid for GCOV. \n'
                    'Polarization specified through fields (e.g --fields HHHH HVHV)')
    elif myArgs['product'] not in ['RSLC']:
            myArgs['hdfOpenKeywords']['polarization'] = args.polarization
    elif myArgs['product'] in ['RSLC'] and args.quickLook:
        # If just quickLook, compute power
        myArgs['hdfOpenKeywords']['power'] = True
    #
    # Only offset products have layers
    if myArgs['product'] in ['GOFF', 'ROFF']:
        myArgs['hdfOpenKeywords']['layers'] = [f'layer{x}' for x in args.layers]
    #
    if args.conserveMem:
        myArgs['hdfOpenKeywords']['useNumpy'] = False
    else:
        myArgs['hdfOpenKeywords']['useNumpy'] = True


def reconcileFields(args, myArgs):
    '''
    Use defaults and command line options to produce final set of fields
    '''
    #
    # Reconcile fields
    fields = args.fields
    product = myArgs['product']
    if len(fields) == 0:
        fields = defaultFieldsDict[product][myArgs['productType'] ]
    elif 'all' in fields:
        fields = fieldsDict[product][myArgs['productType']]
    else:
        for field in fields:
            if field not in fieldsDict[product][myArgs['productType']]:
                myerror(f'{field} not in '
                        f'{fieldsDict[product][myArgs["productType"]] }') 
    return fields

def printError(msg):
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


def getProductType(productType, productPath):
    '''
    Return either passed in productType if or find type from productPath
    '''
    if productType in productTypes:
        return productType
    # Find product Type
    for productType in productTypes:
        if productType in productPath.upper():
            return productType
    printError('Product type not specified and cannot parse from product path')
    
def outputData(myArgs, myProduct, noSuffix=False, scale=1):
    '''
    Output data

    Parameters
    ----------
    myArgs : dict
        Parameters.
    myProduct : nisarhdf obj
        The product.

    Returns
    -------
    None.

    '''
    keywords = {}
    if myArgs['productType'] in ['GOFF', 'ROFF']:
        keywords = {'scaleToPixels': myArgs['scaleToPixels']}
    #if myArgs['product'] in ['GCOV']: 
     #   keywords = {'dB': myArgs['dB'], 'sigma0': myArgs['sigma0']}
        #print(keywords)
    #
    #for myArgs[]
    tiff, driverName = {'GTiff': [True, 'GTiff'],
                        'COG': [True, 'COG'],
                        'binary': [False, '']}[myArgs['outputFormat']]
    # write data
    myProduct.writeData(myArgs['output'],
                        tiff=tiff,
                        quickLook=myArgs['quickLook'],
                        driverName=driverName, bands=None,
                        noSuffix=noSuffix,
                        byteOrder=myArgs['byteOrder'],
                        scale=scale,
                        vrtOnly=myArgs.get('vrtOnly', False),
                        **keywords)
    return
 

def printInfo(myArgs, myProduct):
    '''

    Parameters
    ----------
    myArgs : dict
        CLI parameters.
    myProduct : nisarhdf object
        The current product.
    Returns
    -------
    None.

    '''
    #
    # List defaults
    if myArgs["product"] in ['GCOV', 'RSLC']:
        print("\n\033[1mDefault polarization:\033[0m")
        for pol in myProduct.polarizations:
            print(pol, end=' ')
        print('')
        print('\033[1mAvailable fields: \033[0m', end='\n')
        if myArgs["product"] in ['GCOV']:
            prodSpecTerms = myProduct.covTerms
        else:
            prodSpecTerms = myProduct.polarizations
        for pol in prodSpecTerms:
            print(f'{pol}')
        for field in fieldsDict[myArgs["product"]][None]:
            if field[0:2] not in ['HH', 'VV', 'HV', 'VH']:
                print(f'{field}')
    else: 
        for productType in defaultFieldsDict[myArgs["product"]]:
            prodName = {True: myArgs["product"],
                        False: myArgs["productType"]}[productType is None]
            print(f'\033[1m\nDefaults for {prodName}: \033[0m', end='\n')
            for field in defaultFieldsDict[myArgs["product"]][productType]:
                print(f'{field}')
            print(f'\033[1mAvailable fields for {productType}: \033[0m', end='\n')
            for field in fieldsDict[myArgs["product"]][productType]:
                print(f'{field}')
    #
    myProduct.printParams()



tracemalloc.start()

def report_peak():
    current, peak = tracemalloc.get_traced_memory()
    print('\n\033[1mMemory Usage: \033[0m', end='\n')
    print(f"Current memory usage: {human_size(current)}")
    print(f"Peak memory usage: {human_size(peak)}")

atexit.register(report_peak)

def run():
    ''' 
    Main driver for program
    '''
    tracemalloc.start()
    myArgs = parseCommandLine()
    net_before = psutil.net_io_counters()
    # Debug
    if False:
        for key in myArgs:
            print(key, myArgs[key])
    #sys.exit()
    #
    #
    myProduct = getattr(nisarhdf,
                        f'nisar{myArgs["product"]}HDF')(
                            frequency=myArgs['frequency'])
    # read data
    start = datetime.now()
    myProduct.openHDF(myArgs['productName'],
                      productType=myArgs['productType'],
                      noLoadData=myArgs['info'] or myArgs['vrtOnly'],
                      fields=myArgs['fields'],
                      useRos3=myArgs['ros3'],
                      page_buf_size=2*1024**3,
                      downsampleFactor=myArgs['downsampleFactor'],
                      **myArgs['hdfOpenKeywords']
                      )
    noSuffix = False
    nFields = 0
    for field in myArgs['fields']:
        if hasattr(myProduct, field):
            nFields += 1
    if myArgs['noSuffix']:
        noSuffix = True
    print('noSuffix', noSuffix, nFields)
    read = datetime.now()
    #
    print('Data loaded')
    # print info and return for --info
    if myArgs['info']:
        printInfo(myArgs, myProduct)
    else:
        outputData(myArgs, myProduct, noSuffix=noSuffix, scale=myArgs['scale'])  
    write = datetime.now()
    print('\n\033[1mRun time: \033[0m', end='\n')
    print(f'Load data: {read-start}')
    print(f'Write data: {write-read}')
    print('Total', write-start)
    #
    print('\n\033[1mNetwork traffic: \033[0m', end='\n')
    net_after = psutil.net_io_counters()    
    bytes_sent = net_after.bytes_sent - net_before.bytes_sent
    bytes_recv = net_after.bytes_recv - net_before.bytes_recv
    print(f"Bytes sent: {human_size(bytes_sent)}")
    print(f"Bytes received: {human_size(bytes_recv)}")

if __name__ == "__main__":
    run()
    
