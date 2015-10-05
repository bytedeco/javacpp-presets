/*====================================================================*
 -  Copyright (C) 2001 Leptonica.  All rights reserved.
 -
 -  Redistribution and use in source and binary forms, with or without
 -  modification, are permitted provided that the following conditions
 -  are met:
 -  1. Redistributions of source code must retain the above copyright
 -     notice, this list of conditions and the following disclaimer.
 -  2. Redistributions in binary form must reproduce the above
 -     copyright notice, this list of conditions and the following
 -     disclaimer in the documentation and/or other materials
 -     provided with the distribution.
 -
 -  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 -  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 -  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 -  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL ANY
 -  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 -  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 -  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 -  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 -  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 -  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *====================================================================*/

#ifndef  LEPTONICA_ALLHEADERS_H
#define  LEPTONICA_ALLHEADERS_H


#define LIBLEPT_MAJOR_VERSION   1
#define LIBLEPT_MINOR_VERSION   72

#include "alltypes.h"

#ifndef NO_PROTOS
/*
 *  These prototypes were autogen'd by xtractprotos, v. 1.5
 */
#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */

LEPT_DLL extern BOXA * boxaRotate ( BOXA *boxas, l_float32 xc, l_float32 yc, l_float32 angle );
LEPT_DLL extern PIX * pixReduceRankBinaryCascade ( PIX *pixs, l_int32 level1, l_int32 level2, l_int32 level3, l_int32 level4 );
LEPT_DLL extern BOX * boxCreate ( l_int32 x, l_int32 y, l_int32 w, l_int32 h );
LEPT_DLL extern void boxDestroy ( BOX **pbox );
LEPT_DLL extern l_int32 boxGetGeometry ( BOX *box, l_int32 *px, l_int32 *py, l_int32 *pw, l_int32 *ph );
LEPT_DLL extern BOXA * boxaCreate ( l_int32 n );
LEPT_DLL extern void boxaDestroy ( BOXA **pboxa );
LEPT_DLL extern l_int32 boxaAddBox ( BOXA *boxa, BOX *box, l_int32 copyflag );
LEPT_DLL extern l_int32 boxaGetCount ( BOXA *boxa );
LEPT_DLL extern BOX * boxaGetBox ( BOXA *boxa, l_int32 index, l_int32 accessflag );
LEPT_DLL extern l_int32 boxaGetBoxGeometry ( BOXA *boxa, l_int32 index, l_int32 *px, l_int32 *py, l_int32 *pw, l_int32 *ph );
LEPT_DLL extern l_int32 boxaReplaceBox ( BOXA *boxa, l_int32 index, BOX *box );
LEPT_DLL extern BOXAA * boxaaCreate ( l_int32 n );
LEPT_DLL extern void boxaaDestroy ( BOXAA **pbaa );
LEPT_DLL extern l_int32 boxaaAddBoxa ( BOXAA *baa, BOXA *ba, l_int32 copyflag );
LEPT_DLL extern l_int32 boxaaRemoveBoxa ( BOXAA *baa, l_int32 index );
LEPT_DLL extern l_int32 boxaaAddBox ( BOXAA *baa, l_int32 index, BOX *box, l_int32 accessflag );
LEPT_DLL extern BOX * boxBoundingRegion ( BOX *box1, BOX *box2 );
LEPT_DLL extern PIX * pixDrawBoxa ( PIX *pixs, BOXA *boxa, l_int32 width, l_uint32 val );
LEPT_DLL extern l_int32 boxaGetExtent ( BOXA *boxa, l_int32 *pw, l_int32 *ph, BOX **pbox );
LEPT_DLL extern PIXCMAP * pixcmapCreate ( l_int32 depth );
LEPT_DLL extern l_int32 pixcmapAddColor ( PIXCMAP *cmap, l_int32 rval, l_int32 gval, l_int32 bval );
LEPT_DLL extern BOXA * pixConnComp ( PIX *pixs, PIXA **ppixa, l_int32 connectivity );
LEPT_DLL extern l_int32 pixCountConnComp ( PIX *pixs, l_int32 connectivity, l_int32 *pcount );
LEPT_DLL extern l_int32 pixSeedfill ( PIX *pixs, L_STACK *stack, l_int32 x, l_int32 y, l_int32 connectivity );
LEPT_DLL extern PIX * pixBlockconv ( PIX *pix, l_int32 wc, l_int32 hc );
LEPT_DLL extern l_int32 pixRenderBox ( PIX *pix, BOX *box, l_int32 width, l_int32 op );
LEPT_DLL extern l_int32 pixRenderBoxArb ( PIX *pix, BOX *box, l_int32 width, l_uint8 rval, l_uint8 gval, l_uint8 bval );
LEPT_DLL extern l_int32 pixRenderPolyline ( PIX *pix, PTA *ptas, l_int32 width, l_int32 op, l_int32 closeflag );
LEPT_DLL extern l_int32 pixRenderPolylineArb ( PIX *pix, PTA *ptas, l_int32 width, l_uint8 rval, l_uint8 gval, l_uint8 bval, l_int32 closeflag );
LEPT_DLL extern PIX * pixErodeGray ( PIX *pixs, l_int32 hsize, l_int32 vsize );
LEPT_DLL extern PIX * pixThresholdToBinary ( PIX *pixs, l_int32 thresh );
LEPT_DLL extern l_int32 pixWriteJpeg ( const char *filename, PIX *pix, l_int32 quality, l_int32 progressive );
LEPT_DLL extern char * getImagelibVersions (  );
LEPT_DLL extern PIX * pixDilate ( PIX *pixd, PIX *pixs, SEL *sel );
LEPT_DLL extern PIX * pixErode ( PIX *pixd, PIX *pixs, SEL *sel );
LEPT_DLL extern PIX * pixOpen ( PIX *pixd, PIX *pixs, SEL *sel );
LEPT_DLL extern PIX * pixClose ( PIX *pixd, PIX *pixs, SEL *sel );
LEPT_DLL extern PIX * pixDilateBrick ( PIX *pixd, PIX *pixs, l_int32 hsize, l_int32 vsize );
LEPT_DLL extern PIX * pixErodeBrick ( PIX *pixd, PIX *pixs, l_int32 hsize, l_int32 vsize );
LEPT_DLL extern PIX * pixOpenBrick ( PIX *pixd, PIX *pixs, l_int32 hsize, l_int32 vsize );
LEPT_DLL extern PIX * pixCloseBrick ( PIX *pixd, PIX *pixs, l_int32 hsize, l_int32 vsize );
LEPT_DLL extern PIXA * pixaMorphSequenceByRegion ( PIX *pixs, PIXA *pixam, const char *sequence, l_int32 minw, l_int32 minh );
LEPT_DLL extern PIX * pixMorphCompSequence ( PIX *pixs, const char *sequence, l_int32 dispsep );
LEPT_DLL extern void numaDestroy ( NUMA **pna );
LEPT_DLL extern l_int32 numaGetCount ( NUMA *na );
LEPT_DLL extern l_int32 numaGetIValue ( NUMA *na, l_int32 index, l_int32 *pival );
LEPT_DLL extern PIX * pixGenHalftoneMask ( PIX *pixs, PIX **ppixtext, l_int32 *phtfound, l_int32 debug );
LEPT_DLL extern l_int32 pixaConvertToPdf ( PIXA *pixa, l_int32 res, l_float32 scalefactor, l_int32 type, l_int32 quality, const char *title, const char *fileout );
LEPT_DLL extern l_int32 l_generateCIDataForPdf ( const char *fname, PIX *pix, l_int32 quality, L_COMP_DATA **pcid );
LEPT_DLL extern l_int32 l_generateCIData ( const char *fname, l_int32 type, l_int32 quality, l_int32 ascii85, L_COMP_DATA **pcid );
LEPT_DLL extern l_int32 pixGenerateCIData ( PIX *pixs, l_int32 type, l_int32 quality, l_int32 ascii85, L_COMP_DATA **pcid );
LEPT_DLL extern void l_CIDataDestroy ( L_COMP_DATA **pcid );
LEPT_DLL extern PIX * pixCreate ( l_int32 width, l_int32 height, l_int32 depth );
LEPT_DLL extern PIX * pixCreateTemplate ( PIX *pixs );
LEPT_DLL extern PIX * pixCreateHeader ( l_int32 width, l_int32 height, l_int32 depth );
LEPT_DLL extern PIX * pixClone ( PIX *pixs );
LEPT_DLL extern void pixDestroy ( PIX **ppix );
LEPT_DLL extern PIX * pixCopy ( PIX *pixd, PIX *pixs );
LEPT_DLL extern l_int32 pixSizesEqual ( PIX *pix1, PIX *pix2 );
LEPT_DLL extern l_int32 pixGetWidth ( PIX *pix );
LEPT_DLL extern l_int32 pixGetHeight ( PIX *pix );
LEPT_DLL extern l_int32 pixGetDepth ( PIX *pix );
LEPT_DLL extern l_int32 pixGetDimensions ( PIX *pix, l_int32 *pw, l_int32 *ph, l_int32 *pd );
LEPT_DLL extern l_int32 pixGetSpp ( PIX *pix );
LEPT_DLL extern l_int32 pixSetSpp ( PIX *pix, l_int32 spp );
LEPT_DLL extern l_int32 pixGetWpl ( PIX *pix );
LEPT_DLL extern l_int32 pixGetXRes ( PIX *pix );
LEPT_DLL extern l_int32 pixSetXRes ( PIX *pix, l_int32 res );
LEPT_DLL extern l_int32 pixGetYRes ( PIX *pix );
LEPT_DLL extern l_int32 pixSetYRes ( PIX *pix, l_int32 res );
LEPT_DLL extern l_int32 pixSetInputFormat ( PIX *pix, l_int32 informat );
LEPT_DLL extern l_int32 pixSetText ( PIX *pix, const char *textstring );
LEPT_DLL extern PIXCMAP * pixGetColormap ( PIX *pix );
LEPT_DLL extern l_int32 pixSetColormap ( PIX *pix, PIXCMAP *colormap );
LEPT_DLL extern l_uint32 * pixGetData ( PIX *pix );
LEPT_DLL extern l_int32 pixSetData ( PIX *pix, l_uint32 *data );
LEPT_DLL extern l_int32 pixGetPixel ( PIX *pix, l_int32 x, l_int32 y, l_uint32 *pval );
LEPT_DLL extern l_int32 pixSetPixel ( PIX *pix, l_int32 x, l_int32 y, l_uint32 val );
LEPT_DLL extern l_int32 pixSetAll ( PIX *pix );
LEPT_DLL extern l_int32 pixSetAllArbitrary ( PIX *pix, l_uint32 val );
LEPT_DLL extern l_int32 pixClearInRect ( PIX *pix, BOX *box );
LEPT_DLL extern l_int32 pixSetInRect ( PIX *pix, BOX *box );
LEPT_DLL extern l_int32 pixSetInRectArbitrary ( PIX *pix, BOX *box, l_uint32 val );
LEPT_DLL extern l_int32 pixSetOrClearBorder ( PIX *pixs, l_int32 left, l_int32 right, l_int32 top, l_int32 bot, l_int32 op );
LEPT_DLL extern PIX * pixAddBorder ( PIX *pixs, l_int32 npix, l_uint32 val );
LEPT_DLL extern l_int32 composeRGBPixel ( l_int32 rval, l_int32 gval, l_int32 bval, l_uint32 *ppixel );
LEPT_DLL extern l_int32 pixEndianByteSwap ( PIX *pixs );
LEPT_DLL extern l_int32 pixEndianTwoByteSwap ( PIX *pixs );
LEPT_DLL extern l_int32 pixSetMasked ( PIX *pixd, PIX *pixm, l_uint32 val );
LEPT_DLL extern PIX * pixInvert ( PIX *pixd, PIX *pixs );
LEPT_DLL extern PIX * pixOr ( PIX *pixd, PIX *pixs1, PIX *pixs2 );
LEPT_DLL extern PIX * pixAnd ( PIX *pixd, PIX *pixs1, PIX *pixs2 );
LEPT_DLL extern PIX * pixXor ( PIX *pixd, PIX *pixs1, PIX *pixs2 );
LEPT_DLL extern PIX * pixSubtract ( PIX *pixd, PIX *pixs1, PIX *pixs2 );
LEPT_DLL extern l_int32 pixZero ( PIX *pix, l_int32 *pempty );
LEPT_DLL extern l_int32 pixForegroundFraction ( PIX *pix, l_float32 *pfract );
LEPT_DLL extern l_int32 pixCountPixels ( PIX *pix, l_int32 *pcount, l_int32 *tab8 );
LEPT_DLL extern NUMA * pixCountPixelsByRow ( PIX *pix, l_int32 *tab8 );
LEPT_DLL extern l_int32 pixCountPixelsInRow ( PIX *pix, l_int32 row, l_int32 *pcount, l_int32 *tab8 );
LEPT_DLL extern PIX * pixClipRectangle ( PIX *pixs, BOX *box, BOX **pboxc );
LEPT_DLL extern l_int32 pixClipBoxToForeground ( PIX *pixs, BOX *boxs, PIX **ppixd, BOX **pboxd );
LEPT_DLL extern PIXA * pixaCreate ( l_int32 n );
LEPT_DLL extern void pixaDestroy ( PIXA **ppixa );
LEPT_DLL extern l_int32 pixaAddPix ( PIXA *pixa, PIX *pix, l_int32 copyflag );
LEPT_DLL extern l_int32 pixaAddBox ( PIXA *pixa, BOX *box, l_int32 copyflag );
LEPT_DLL extern l_int32 pixaGetCount ( PIXA *pixa );
LEPT_DLL extern PIX * pixaGetPix ( PIXA *pixa, l_int32 index, l_int32 accesstype );
LEPT_DLL extern l_int32 pixaReplacePix ( PIXA *pixa, l_int32 index, PIX *pix, BOX *box );
LEPT_DLL extern l_int32 pixaInsertPix ( PIXA *pixa, l_int32 index, PIX *pixs, BOX *box );
LEPT_DLL extern l_int32 pixaRemovePix ( PIXA *pixa, l_int32 index );
LEPT_DLL extern PIXAA * pixaaCreate ( l_int32 n );
LEPT_DLL extern void pixaaDestroy ( PIXAA **ppaa );
LEPT_DLL extern l_int32 pixaaAddPixa ( PIXAA *paa, PIXA *pixa, l_int32 copyflag );
LEPT_DLL extern l_int32 pixaaAddPix ( PIXAA *paa, l_int32 index, PIX *pix, BOX *box, l_int32 copyflag );
LEPT_DLL extern l_int32 pixaaAddBox ( PIXAA *paa, BOX *box, l_int32 copyflag );
LEPT_DLL extern PIX * pixaDisplay ( PIXA *pixa, l_int32 w, l_int32 h );
LEPT_DLL extern PIX * pixaDisplayTiled ( PIXA *pixa, l_int32 maxwidth, l_int32 background, l_int32 spacing );
LEPT_DLL extern PIX * pixaDisplayTiledInRows ( PIXA *pixa, l_int32 outdepth, l_int32 maxwidth, l_float32 scalefactor, l_int32 background, l_int32 spacing, l_int32 border );
LEPT_DLL extern PIX * pixRemoveColormap ( PIX *pixs, l_int32 type );
LEPT_DLL extern PIX * pixConvertRGBToLuminance ( PIX *pixs );
LEPT_DLL extern PIX * pixConvertTo8 ( PIX *pixs, l_int32 cmapflag );
LEPT_DLL extern PIX * pixConvertTo32 ( PIX *pixs );
LEPT_DLL extern l_int32 pixWriteStreamPng ( FILE *fp, PIX *pix, l_float32 gamma );
LEPT_DLL extern PTA * ptaCreate ( l_int32 n );
LEPT_DLL extern void ptaDestroy ( PTA **ppta );
LEPT_DLL extern l_int32 ptaAddPt ( PTA *pta, l_float32 x, l_float32 y );
LEPT_DLL extern PIX * pixRead ( const char *filename );
LEPT_DLL extern PIX * pixReadStream ( FILE *fp, l_int32 hint );
LEPT_DLL extern l_int32 findFileFormat ( const char *filename, l_int32 *pformat );
LEPT_DLL extern l_int32 findFileFormatBuffer ( const l_uint8 *buf, l_int32 *pformat );
LEPT_DLL extern PIX * pixReadMem ( const l_uint8 *data, size_t size );
LEPT_DLL extern l_int32 pixRasterop ( PIX *pixd, l_int32 dx, l_int32 dy, l_int32 dw, l_int32 dh, l_int32 op, PIX *pixs, l_int32 sx, l_int32 sy );
LEPT_DLL extern PIX * pixRotate ( PIX *pixs, l_float32 angle, l_int32 type, l_int32 incolor, l_int32 width, l_int32 height );
LEPT_DLL extern PIX * pixRotate90 ( PIX *pixs, l_int32 direction );
LEPT_DLL extern PIX * pixFlipLR ( PIX *pixd, PIX *pixs );
LEPT_DLL extern PIX * pixFlipTB ( PIX *pixd, PIX *pixs );
LEPT_DLL extern PIX * pixScale ( PIX *pixs, l_float32 scalex, l_float32 scaley );
LEPT_DLL extern PIX * pixScaleToSize ( PIX *pixs, l_int32 wd, l_int32 hd );
LEPT_DLL extern PIX * pixExpandReplicate ( PIX *pixs, l_int32 factor );
LEPT_DLL extern PIX * pixSeedfillBinary ( PIX *pixd, PIX *pixs, PIX *pixm, l_int32 connectivity );
LEPT_DLL extern PIX * pixDistanceFunction ( PIX *pixs, l_int32 connectivity, l_int32 outdepth, l_int32 boundcond );
LEPT_DLL extern SEL * selCreate ( l_int32 height, l_int32 width, const char *name );
LEPT_DLL extern SEL * selCreateBrick ( l_int32 h, l_int32 w, l_int32 cy, l_int32 cx, l_int32 type );
LEPT_DLL extern l_int32 selFindMaxTranslations ( SEL *sel, l_int32 *pxp, l_int32 *pyp, l_int32 *pxn, l_int32 *pyn );
LEPT_DLL extern PIX * pixDeskew ( PIX *pixs, l_int32 redsearch );
LEPT_DLL extern PIX * pixReadTiff ( const char *filename, l_int32 n );
LEPT_DLL extern PIX * pixReadStreamTiff ( FILE *fp, l_int32 n );
LEPT_DLL extern l_int32 pixWriteTiff ( const char *filename, PIX *pix, l_int32 comptype, const char *modestring );
LEPT_DLL extern PIX * pixReadMemTiff ( const l_uint8 *cdata, size_t size, l_int32 n );
LEPT_DLL extern l_int32 stringLength ( const char *src, size_t size );
LEPT_DLL extern void * reallocNew ( void **pindata, l_int32 oldsize, l_int32 newsize );
LEPT_DLL extern FILE * fopenReadStream ( const char *filename );
LEPT_DLL extern void lept_free ( void *ptr );
LEPT_DLL extern char * getLeptonicaVersion (  );
LEPT_DLL extern void l_getCurrentTime ( l_int32 *sec, l_int32 *usec );
LEPT_DLL extern char * l_getFormattedDate (  );
LEPT_DLL extern l_int32 pixWrite ( const char *filename, PIX *pix, l_int32 format );
LEPT_DLL extern l_int32 pixWriteStream ( FILE *fp, PIX *pix, l_int32 format );
LEPT_DLL extern l_int32 pixWriteMem ( l_uint8 **pdata, size_t *psize, PIX *pix, l_int32 format );
LEPT_DLL extern l_int32 pixDisplay ( PIX *pixs, l_int32 x, l_int32 y );
LEPT_DLL extern l_int32 pixDisplayWrite ( PIX *pixs, l_int32 reduction );
LEPT_DLL extern l_uint8 * zlibCompress ( l_uint8 *datain, size_t nin, size_t *pnout );
LEPT_DLL extern PIX * pixReadStreamBmp ( FILE *fp );
LEPT_DLL extern l_int32 pixWriteStreamBmp ( FILE *fp, PIX *pix );
LEPT_DLL extern PIX * pixReadMemBmp ( const l_uint8 *cdata, size_t size );
LEPT_DLL extern l_int32 pixWriteMemBmp ( l_uint8 **pdata, size_t *psize, PIX *pix );
LEPT_DLL extern PIX * pixReadStreamGif ( FILE *fp );
LEPT_DLL extern l_int32 pixWriteStreamGif ( FILE *fp, PIX *pix );
LEPT_DLL extern PIX * pixReadMemGif ( const l_uint8 *cdata, size_t size );
LEPT_DLL extern l_int32 pixWriteMemGif ( l_uint8 **pdata, size_t *psize, PIX *pix );
LEPT_DLL extern PIX * pixReadJpeg ( const char *filename, l_int32 cmapflag, l_int32 reduction, l_int32 *pnwarn, l_int32 hint );
LEPT_DLL extern l_int32 pixWriteJpeg ( const char *filename, PIX *pix, l_int32 quality, l_int32 progressive );
LEPT_DLL extern l_int32 pixWriteStreamJpeg ( FILE *fp, PIX *pixs, l_int32 quality, l_int32 progressive );
LEPT_DLL extern PIX * pixReadMemJpeg ( const l_uint8 *data, size_t size, l_int32 cmflag, l_int32 reduction, l_int32 *pnwarn, l_int32 hint );
LEPT_DLL extern PIX * pixReadStreamPng ( FILE *fp );
LEPT_DLL extern PIX * pixReadMemPng ( const l_uint8 *cdata, size_t size );
LEPT_DLL extern l_int32 pixWriteMemPng ( l_uint8 **pdata, size_t *psize, PIX *pix, l_float32 gamma );
LEPT_DLL extern l_int32 pixWritePng ( const char *filename, PIX *pix, l_float32 gamma );
LEPT_DLL extern PIX * pixReadMem ( const l_uint8 *data, size_t size );
LEPT_DLL extern PIX * pixRead ( const char *filename );
LEPT_DLL extern PIX * pixReadWithHint ( const char *filename, l_int32 hint );
LEPT_DLL extern PIX * pixReadStream ( FILE *fp, l_int32 hint );
LEPT_DLL extern PIX * pixReadTiff ( const char *filename, l_int32 n );
LEPT_DLL extern PIX * pixReadStreamTiff ( FILE *fp, l_int32 n );
LEPT_DLL extern l_int32 pixWriteTiff ( const char *filename, PIX *pix, l_int32 comptype, const char *modestring );
LEPT_DLL extern l_int32 pixWriteMemTiff ( l_uint8 **pdata, size_t *psize, PIX *pix, l_int32 comptype );
LEPT_DLL extern PIX * pixReadStreamWebP ( FILE *fp );
LEPT_DLL extern PIX * pixReadMemWebP ( const l_uint8 *filedata, size_t filesize );
LEPT_DLL extern l_int32 pixWriteWebP ( const char *filename, PIX *pixs, l_int32 quality, l_int32 lossless );
LEPT_DLL extern l_int32 pixWriteStreamWebP ( FILE *fp, PIX *pixs, l_int32 quality, l_int32 lossless );
LEPT_DLL extern l_int32 pixWriteMemWebP ( l_uint8 **pencdata, size_t *pencsize, PIX *pixs, l_int32 quality, l_int32 lossless );

#ifdef __cplusplus
}
#endif  /* __cplusplus */
#endif /* NO_PROTOS */


#endif /* LEPTONICA_ALLHEADERS_H */

