"""
- *RDT Image if the scan is successful* [FileName: Barcode_RDTScan]
- *Enhanced (high contrast) image if the scan is successful* [FileName: Barcode_EnhancedScan]
- *RDT Image if a photo is taken manually* [FileName: Barcode_ManualPhoto]
"""

S3_URL_BASE_PATH = 'https://s3-us-west-2.amazonaws.com/fileshare.auderenow.io/public/rdt-reader-photos/'
TYPE = 'NONE'
BARCODES = [11111111,
            11223344,
            12345600,
            12345601,
            12345602,
            12345603,
            22222222,
            62462611,
            63524073]

RDT_SCAN = '_RDTScan'
ENHANCED_SCAN = '_EnhancedScan'
MANUAL_PHOTO = '_ManualPhoto'
