// Targeted by JavaCPP version 0.8-2-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

public class FlyCapture2_C extends org.bytedeco.javacpp.presets.FlyCapture2_C {
    static { Loader.load(); }

// Parsed from <FlyCapture2Defs_C.h>

//=============================================================================
// Copyright � 2008 Point Grey Research, Inc. All Rights Reserved.
//
// This software is the confidential and proprietary information of Point
// Grey Research, Inc. ("Confidential Information").  You shall not
// disclose such Confidential Information and shall use it only in
// accordance with the terms of the license agreement you entered into
// with PGR.
//
// PGR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
// SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, OR NON-INFRINGEMENT. PGR SHALL NOT BE LIABLE FOR ANY DAMAGES
// SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
// THIS SOFTWARE OR ITS DERIVATIVES.
//=============================================================================
//=============================================================================
// $Id: FlyCapture2Defs_C.h,v 1.74 2010-12-13 23:58:00 mgara Exp $
//=============================================================================

// #ifndef PGR_FC2_FLYCAPTURE2DEFS_C_H
// #define PGR_FC2_FLYCAPTURE2DEFS_C_H

// #include <stdlib.h>

//=============================================================================
// C definitions header file for FlyCapture2. 
//
// This file defines the C enumerations, typedefs and structures for FlyCapture2
//
// Please see FlyCapture2Defs.h or the API documentation for full details
// of the various enumerations and structures.
//=============================================================================

// #ifdef __cplusplus
// #endif

//=============================================================================
// Typedefs
//=============================================================================  

// #ifndef FALSE
public static final int FALSE =               0;
// #endif

// #ifndef TRUE
public static final int TRUE =                1;
// #endif

// #ifndef FULL_32BIT_VALUE
public static final int FULL_32BIT_VALUE = 0x7FFFFFFF;
// #endif 

public static final int MAX_STRING_LENGTH =   512;

/**
 * A context to the FlyCapture2 C library. It must be created before
 * performing any calls to the library.
 */ 
@Opaque public static class fc2Context extends Pointer {
    public fc2Context() { }
    public fc2Context(Pointer p) { super(p); }
}

/**
 * A context to the FlyCapture2 C GUI library. It must be created before
 * performing any calls to the library.
 */ 
@Opaque public static class fc2GuiContext extends Pointer {
    public fc2GuiContext() { }
    public fc2GuiContext(Pointer p) { super(p); }
}

/**
 * An internal pointer used in the fc2Image structure.
 */ 
@Opaque public static class fc2ImageImpl extends Pointer {
    public fc2ImageImpl() { }
    public fc2ImageImpl(Pointer p) { super(p); }
}

/**
 * A context referring to the AVI recorder object.
 */ 
@Opaque public static class fc2AVIContext extends Pointer {
    public fc2AVIContext() { }
    public fc2AVIContext(Pointer p) { super(p); }
}

/**
 * A context referring to the ImageStatistics object.
 */ 
@Opaque public static class fc2ImageStatisticsContext extends Pointer {
    public fc2ImageStatisticsContext() { }
    public fc2ImageStatisticsContext(Pointer p) { super(p); }
}

/**
 * A GUID to the camera.  It is used to uniquely identify a camera.
 */ 
public static class fc2PGRGuid extends Pointer {
    static { Loader.load(); }
    public fc2PGRGuid() { allocate(); }
    public fc2PGRGuid(int size) { allocateArray(size); }
    public fc2PGRGuid(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2PGRGuid position(int position) {
        return (fc2PGRGuid)super.position(position);
    }

    public native @Cast("unsigned int") int value(int i); public native fc2PGRGuid value(int i, int value);
    @MemberGetter public native @Cast("unsigned int*") IntPointer value();

}

//=============================================================================
// Enumerations
//=============================================================================

/** enum fc2Error */
public static final int
    /** Undefined */
    FC2_ERROR_UNDEFINED = -1,
    /** Function returned with no errors. */
    FC2_ERROR_OK = 0,
    /** General failure. */
    FC2_ERROR_FAILED = 1,
    /** Function has not been implemented. */
    FC2_ERROR_NOT_IMPLEMENTED = 2,
    /** Could not connect to Bus Master. */
    FC2_ERROR_FAILED_BUS_MASTER_CONNECTION = 3,
    /** Camera has not been connected. */
    FC2_ERROR_NOT_CONNECTED = 4,
    /** Initialization failed. */
    FC2_ERROR_INIT_FAILED = 5, 
    /** Camera has not been initialized. */
    FC2_ERROR_NOT_INTITIALIZED = 6,
    /** Invalid parameter passed to function. */
    FC2_ERROR_INVALID_PARAMETER = 7,
    /** Setting set to camera is invalid. */
    FC2_ERROR_INVALID_SETTINGS = 8,         
    /** Invalid Bus Manager object. */
    FC2_ERROR_INVALID_BUS_MANAGER = 9,
    /** Could not allocate memory. */
    FC2_ERROR_MEMORY_ALLOCATION_FAILED = 10, 
    /** Low level error. */
    FC2_ERROR_LOW_LEVEL_FAILURE = 11,
    /** Device not found. */
    FC2_ERROR_NOT_FOUND = 12,
    /** GUID failure. */
    FC2_ERROR_FAILED_GUID = 13,
    /** Packet size set to camera is invalid. */
    FC2_ERROR_INVALID_PACKET_SIZE = 14,
    /** Invalid mode has been passed to function. */
    FC2_ERROR_INVALID_MODE = 15,
    /** Error due to not being in Format7. */
    FC2_ERROR_NOT_IN_FORMAT7 = 16,
    /** This feature is unsupported. */
    FC2_ERROR_NOT_SUPPORTED = 17,
    /** Timeout error. */
    FC2_ERROR_TIMEOUT = 18,
    /** Bus Master Failure. */
    FC2_ERROR_BUS_MASTER_FAILED = 19,
    /** Generation Count Mismatch. */
    FC2_ERROR_INVALID_GENERATION = 20,
    /** Look Up Table failure. */
    FC2_ERROR_LUT_FAILED = 21,
    /** IIDC failure. */
    FC2_ERROR_IIDC_FAILED = 22,
    /** Strobe failure. */
    FC2_ERROR_STROBE_FAILED = 23,
    /** Trigger failure. */
    FC2_ERROR_TRIGGER_FAILED = 24,
    /** Property failure. */
    FC2_ERROR_PROPERTY_FAILED = 25,
    /** Property is not present. */
    FC2_ERROR_PROPERTY_NOT_PRESENT = 26,
    /** Register access failed. */
    FC2_ERROR_REGISTER_FAILED = 27,
    /** Register read failed. */
    FC2_ERROR_READ_REGISTER_FAILED = 28,
    /** Register write failed. */
    FC2_ERROR_WRITE_REGISTER_FAILED = 29,
    /** Isochronous failure. */
    FC2_ERROR_ISOCH_FAILED = 30,
    /** Isochronous transfer has already been started. */
    FC2_ERROR_ISOCH_ALREADY_STARTED = 31,
    /** Isochronous transfer has not been started. */
    FC2_ERROR_ISOCH_NOT_STARTED = 32,
    /** Isochronous start failed. */
    FC2_ERROR_ISOCH_START_FAILED = 33,
    /** Isochronous retrieve buffer failed. */
    FC2_ERROR_ISOCH_RETRIEVE_BUFFER_FAILED = 34,
    /** Isochronous stop failed. */
    FC2_ERROR_ISOCH_STOP_FAILED = 35,
    /** Isochronous image synchronization failed. */
    FC2_ERROR_ISOCH_SYNC_FAILED = 36,
    /** Isochronous bandwidth exceeded. */
    FC2_ERROR_ISOCH_BANDWIDTH_EXCEEDED = 37,
    /** Image conversion failed. */
    FC2_ERROR_IMAGE_CONVERSION_FAILED = 38,
    /** Image library failure. */
    FC2_ERROR_IMAGE_LIBRARY_FAILURE = 39,
    /** Buffer is too small. */
    FC2_ERROR_BUFFER_TOO_SMALL = 40,
    /** There is an image consistency error. */
    FC2_ERROR_IMAGE_CONSISTENCY_ERROR = 41,
    FC2_ERROR_FORCE_32BITS =  FULL_32BIT_VALUE;  

/** enum fc2BusCallbackType */
public static final int
    FC2_BUS_RESET = 0,
    FC2_ARRIVAL = 1,
    FC2_REMOVAL = 2,
    FC2_CALLBACK_TYPE_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2GrabMode */
public static final int
    FC2_DROP_FRAMES = 0,
    FC2_BUFFER_FRAMES = 1,
    FC2_UNSPECIFIED_GRAB_MODE = 2,
    FC2_GRAB_MODE_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2GrabTimeout */
public static final int
    FC2_TIMEOUT_NONE = 0,
    FC2_TIMEOUT_INFINITE = -1,
    FC2_TIMEOUT_UNSPECIFIED = -2,
    FC2_GRAB_TIMEOUT_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2BandwidthAllocation */
public static final int
    FC2_BANDWIDTH_ALLOCATION_OFF = 0,
    FC2_BANDWIDTH_ALLOCATION_ON = 1,
    FC2_BANDWIDTH_ALLOCATION_UNSUPPORTED = 2,
    FC2_BANDWIDTH_ALLOCATION_UNSPECIFIED = 3,
    FC2_BANDWIDTH_ALLOCATION_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2InterfaceType */
public static final int        
    FC2_INTERFACE_IEEE1394 = 0,
    FC2_INTERFACE_USB_2 = 1,
    FC2_INTERFACE_USB_3 = 2,
	FC2_INTERFACE_GIGE = 3,
    FC2_INTERFACE_UNKNOWN = 4,
    FC2_INTERFACE_TYPE_FORCE_32BITS =  FULL_32BIT_VALUE;

/** Types of low level drivers that flycapture uses. */
/** enum fc2DriverType */
public static final int        
    /** PGRCam.sys. */
    FC2_DRIVER_1394_CAM = 0,
    /** PGR1394.sys. */
    FC2_DRIVER_1394_PRO = 1,
    /** firewire_core. */
    FC2_DRIVER_1394_JUJU = 2,
    /** video1394. */
    FC2_DRIVER_1394_VIDEO1394 = 3,
    /** raw1394. */
    FC2_DRIVER_1394_RAW1394 = 4,
    /** No usb driver used just BSD stack. (Linux only) */
    FC2_DRIVER_USB_NONE = 5,
    /** PGRUsbCam.sys. */
    FC2_DRIVER_USB_CAM = 6,
    /** PGRXHCI.sys. */
    FC2_DRIVER_USB3_PRO = 7,
    /** no gige drivers used,MS/BSD stack. */
    FC2_DRIVER_GIGE_NONE = 8,
    /** PGRGigE.sys. */
    FC2_DRIVER_GIGE_FILTER = 9,
    /** PGRGigEPro.sys. */
    FC2_DRIVER_GIGE_PRO = 10,
    /** Unknown driver type. */
    FC2_DRIVER_UNKNOWN = -1,
    FC2_DRIVER_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2PropertyType */
public static final int
    FC2_BRIGHTNESS = 0,
    FC2_AUTO_EXPOSURE = 1,
    FC2_SHARPNESS = 2,
    FC2_WHITE_BALANCE = 3,
    FC2_HUE = 4,
    FC2_SATURATION = 5,
    FC2_GAMMA = 6,
    FC2_IRIS = 7,
    FC2_FOCUS = 8,
    FC2_ZOOM = 9,
    FC2_PAN = 10,
    FC2_TILT = 11,
    FC2_SHUTTER = 12,
    FC2_GAIN = 13,
    FC2_TRIGGER_MODE = 14,
    FC2_TRIGGER_DELAY = 15,
    FC2_FRAME_RATE = 16,
    FC2_TEMPERATURE = 17,
    FC2_UNSPECIFIED_PROPERTY_TYPE = 18,
    FC2_PROPERTY_TYPE_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2FrameRate */
public static final int
    /** 1.875 fps. */
    FC2_FRAMERATE_1_875 = 0,       
    /** 3.75 fps. */
    FC2_FRAMERATE_3_75 = 1,   
    /** 7.5 fps. */
    FC2_FRAMERATE_7_5 = 2,    
    /** 15 fps. */
    FC2_FRAMERATE_15 = 3,   
    /** 30 fps. */
    FC2_FRAMERATE_30 = 4,     
    /** 60 fps. */
    FC2_FRAMERATE_60 = 5,     
    /** 120 fps. */
    FC2_FRAMERATE_120 = 6,
    /** 240 fps. */
    FC2_FRAMERATE_240 = 7,
    /** Custom frame rate for Format7 functionality. */
    FC2_FRAMERATE_FORMAT7 = 8,
    /** Number of possible camera frame rates. */
    FC2_NUM_FRAMERATES = 9,
    FC2_FRAMERATE_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2VideoMode */
public static final int
    /** 160x120 YUV444. */
    FC2_VIDEOMODE_160x120YUV444 = 0,
    /** 320x240 YUV422. */
    FC2_VIDEOMODE_320x240YUV422 = 1,
    /** 640x480 YUV411. */
    FC2_VIDEOMODE_640x480YUV411 = 2,
    /** 640x480 YUV422. */
    FC2_VIDEOMODE_640x480YUV422 = 3,
    /** 640x480 24-bit RGB. */
    FC2_VIDEOMODE_640x480RGB = 4,
    /** 640x480 8-bit. */
    FC2_VIDEOMODE_640x480Y8 = 5,
    /** 640x480 16-bit. */
    FC2_VIDEOMODE_640x480Y16 = 6,
    /** 800x600 YUV422. */
    FC2_VIDEOMODE_800x600YUV422 = 7,
    /** 800x600 RGB. */
    FC2_VIDEOMODE_800x600RGB = 8,
    /** 800x600 8-bit. */
    FC2_VIDEOMODE_800x600Y8 = 9,
    /** 800x600 16-bit. */
    FC2_VIDEOMODE_800x600Y16 = 10,
    /** 1024x768 YUV422. */
    FC2_VIDEOMODE_1024x768YUV422 = 11,
    /** 1024x768 RGB. */
    FC2_VIDEOMODE_1024x768RGB = 12,
    /** 1024x768 8-bit. */
    FC2_VIDEOMODE_1024x768Y8 = 13,
    /** 1024x768 16-bit. */
    FC2_VIDEOMODE_1024x768Y16 = 14,
    /** 1280x960 YUV422. */
    FC2_VIDEOMODE_1280x960YUV422 = 15,
    /** 1280x960 RGB. */
    FC2_VIDEOMODE_1280x960RGB = 16,
    /** 1280x960 8-bit. */
    FC2_VIDEOMODE_1280x960Y8 = 17,
    /** 1280x960 16-bit. */
    FC2_VIDEOMODE_1280x960Y16 = 18,
    /** 1600x1200 YUV422. */
    FC2_VIDEOMODE_1600x1200YUV422 = 19,
    /** 1600x1200 RGB. */
    FC2_VIDEOMODE_1600x1200RGB = 20,
    /** 1600x1200 8-bit. */
    FC2_VIDEOMODE_1600x1200Y8 = 21,
    /** 1600x1200 16-bit. */
    FC2_VIDEOMODE_1600x1200Y16 = 22,
    /** Custom video mode for Format7 functionality. */
    FC2_VIDEOMODE_FORMAT7 = 23,
    /** Number of possible video modes. */
    FC2_NUM_VIDEOMODES = 24,
    FC2_VIDEOMODE_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2Mode */
public static final int
    FC2_MODE_0 = 0,
    FC2_MODE_1 = 1,
    FC2_MODE_2 = 2,
    FC2_MODE_3 = 3,
    FC2_MODE_4 = 4,
    FC2_MODE_5 = 5,
    FC2_MODE_6 = 6,
    FC2_MODE_7 = 7,
    FC2_MODE_8 = 8,
    FC2_MODE_9 = 9,
    FC2_MODE_10 = 10,
    FC2_MODE_11 = 11,
    FC2_MODE_12 = 12,
    FC2_MODE_13 = 13,
    FC2_MODE_14 = 14,
    FC2_MODE_15 = 15,
    FC2_MODE_16 = 16,
    FC2_MODE_17 = 17,
    FC2_MODE_18 = 18,
    FC2_MODE_19 = 19,
    FC2_MODE_20 = 20,
    FC2_MODE_21 = 21,
    FC2_MODE_22 = 22,
    FC2_MODE_23 = 23,
    FC2_MODE_24 = 24,
    FC2_MODE_25 = 25,
    FC2_MODE_26 = 26,
    FC2_MODE_27 = 27,
    FC2_MODE_28 = 28,
    FC2_MODE_29 = 29,
    FC2_MODE_30 = 30,
    FC2_MODE_31 = 31,
    /** Number of modes */
    FC2_NUM_MODES = 32,
    FC2_MODE_FORCE_32BITS =  FULL_32BIT_VALUE;  

/** enum fc2PixelFormat */
public static final int
    /** 8 bits of mono information. */
    FC2_PIXEL_FORMAT_MONO8			=  0x80000000,
    /** YUV 4:1:1. */
    FC2_PIXEL_FORMAT_411YUV8		=  0x40000000,
    /** YUV 4:2:2. */
    FC2_PIXEL_FORMAT_422YUV8		=  0x20000000,
    /** YUV 4:4:4. */
    FC2_PIXEL_FORMAT_444YUV8		=  0x10000000,
    /** R = G = B = 8 bits. */
    FC2_PIXEL_FORMAT_RGB8			=  0x08000000,
    /** 16 bits of mono information. */
    FC2_PIXEL_FORMAT_MONO16			=  0x04000000,
    /** R = G = B = 16 bits. */
    FC2_PIXEL_FORMAT_RGB16			=  0x02000000,
    /** 16 bits of signed mono information. */
    FC2_PIXEL_FORMAT_S_MONO16		=  0x01000000,
    /** R = G = B = 16 bits signed. */
    FC2_PIXEL_FORMAT_S_RGB16		=  0x00800000,
    /** 8 bit raw data output of sensor. */
    FC2_PIXEL_FORMAT_RAW8			=  0x00400000,
    /** 16 bit raw data output of sensor. */
    FC2_PIXEL_FORMAT_RAW16			=  0x00200000,
    /** 12 bits of mono information. */
    FC2_PIXEL_FORMAT_MONO12			=  0x00100000,
    /** 12 bit raw data output of sensor. */
    FC2_PIXEL_FORMAT_RAW12			=  0x00080000,
    /** 24 bit BGR. */
    FC2_PIXEL_FORMAT_BGR			=  0x80000008,
    /** 32 bit BGRU. */
    FC2_PIXEL_FORMAT_BGRU			=  0x40000008,
    /** 24 bit RGB. */
    FC2_PIXEL_FORMAT_RGB			=  FC2_PIXEL_FORMAT_RGB8,
    /** 32 bit RGBU. */
    FC2_PIXEL_FORMAT_RGBU			=  0x40000002,
    /** R = G = B = 16 bits. */
    FC2_PIXEL_FORMAT_BGR16			=  0x02000001,
	/** 64 bit BGRU. */
	FC2_PIXEL_FORMAT_BGRU16			=  0x02000002,
    /** JPEG compressed stream. */
    FC2_PIXEL_FORMAT_422YUV8_JPEG	=  0x40000001,
    /** Number of pixel formats. */
    FC2_NUM_PIXEL_FORMATS			= 20,
    /** Unspecified pixel format. */
    FC2_UNSPECIFIED_PIXEL_FORMAT	= 0;

/** enum fc2BusSpeed */
public static final int
    /** 100Mbits/sec. */
    FC2_BUSSPEED_S100 = 0,
    /** 200Mbits/sec. */
    FC2_BUSSPEED_S200 = 1,
    /** 400Mbits/sec. */
    FC2_BUSSPEED_S400 = 2,
    /** 480Mbits/sec. Only for USB2 cameras. */
    FC2_BUSSPEED_S480 = 3,
    /** 800Mbits/sec. */
    FC2_BUSSPEED_S800 = 4,
    /** 1600Mbits/sec. */
    FC2_BUSSPEED_S1600 = 5,
    /** 3200Mbits/sec. */
    FC2_BUSSPEED_S3200 = 6,
    /** 5000Mbits/sec. Only for USB3 cameras. */
    FC2_BUSSPEED_S5000 = 7,
    /** 10Base-T. Only for GigE cameras. */
    FC2_BUSSPEED_10BASE_T = 8,
    /** 100Base-T.  Only for GigE cameras.*/
    FC2_BUSSPEED_100BASE_T = 9,
    /** 1000Base-T (Gigabit Ethernet).  Only for GigE cameras. */
    FC2_BUSSPEED_1000BASE_T = 10,
    /** 10000Base-T.  Only for GigE cameras. */
    FC2_BUSSPEED_10000BASE_T = 11,
    /** The fastest speed available. */
    FC2_BUSSPEED_S_FASTEST = 12,
    /** Any speed that is available. */
    FC2_BUSSPEED_ANY = 13,
    /** Unknown bus speed. */
    FC2_BUSSPEED_SPEED_UNKNOWN = -1,
    FC2_BUSSPEED_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2PCIeBusSpeed */
public static final int
	FC2_PCIE_BUSSPEED_2_5 = 0, /** 2.5 Gb/s */
	FC2_PCIE_BUSSPEED_5_0 = 1, /** 5.0 Gb/s */
	FC2_PCIE_BUSSPEED_UNKNOWN = -1, /** Speed is unknown */
	FC2_PCIE_BUSSPEED_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2ColorProcessingAlgorithm */
public static final int
    FC2_DEFAULT = 0,
    FC2_NO_COLOR_PROCESSING = 1,
    FC2_NEAREST_NEIGHBOR_FAST = 2,
    FC2_EDGE_SENSING = 3,
    FC2_HQ_LINEAR = 4,
    FC2_RIGOROUS = 5,
    FC2_IPP = 6,
    FC2_DIRECTIONAL = 7,
    FC2_COLOR_PROCESSING_ALGORITHM_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2BayerTileFormat */
public static final int
    /** No bayer tile format. */
    FC2_BT_NONE = 0,
    /** Red-Green-Green-Blue. */
    FC2_BT_RGGB = 1,
    /** Green-Red-Blue-Green. */
    FC2_BT_GRBG = 2,
    /** Green-Blue-Red-Green. */
    FC2_BT_GBRG = 3,
    /** Blue-Green-Green-Red. */
    FC2_BT_BGGR = 4,
    FC2_BT_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2ImageFileFormat */
public static final int
    /** Determine file format from file extension. */
    FC2_FROM_FILE_EXT = -1,
    /** Portable gray map. */
    FC2_PGM = 0,
    /** Portable pixmap. */
    FC2_PPM = 1,
    /** Bitmap. */
    FC2_BMP = 2,
    /** JPEG. */
    FC2_JPEG = 3,
    /** JPEG 2000. */
    FC2_JPEG2000 = 4,
    /** Tagged image file format. */
    FC2_TIFF = 5,
    /** Portable network graphics. */
    FC2_PNG = 6,
    /** Raw data. */
    FC2_RAW = 7,
    FC2_IMAGE_FILE_FORMAT_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2GigEPropertyType */
public static final int
    FC2_HEARTBEAT = 0,
    FC2_HEARTBEAT_TIMEOUT = 1;

/** enum fc2StatisticsChannel */
public static final int
    FC2_STATISTICS_GREY = 0,
    FC2_STATISTICS_RED = 1,
    FC2_STATISTICS_GREEN = 2,
    FC2_STATISTICS_BLUE = 3,
    FC2_STATISTICS_HUE = 4,
    FC2_STATISTICS_SATURATION = 5,
    FC2_STATISTICS_LIGHTNESS = 6,
    FC2_STATISTICS_FORCE_32BITS =  FULL_32BIT_VALUE;


/** enum fc2OSType */
public static final int
	FC2_WINDOWS_X86 = 0,
	FC2_WINDOWS_X64 = 1,
	FC2_LINUX_X86 = 2,
	FC2_LINUX_X64 = 3,
	FC2_MAC = 4,
	FC2_UNKNOWN_OS = 5,
    FC2_OSTYPE_FORCE_32BITS =  FULL_32BIT_VALUE;

/** enum fc2ByteOrder */
public static final int
	FC2_BYTE_ORDER_LITTLE_ENDIAN = 0,
	FC2_BYTE_ORDER_BIG_ENDIAN = 1,
    FC2_BYTE_ORDER_FORCE_32BITS =  FULL_32BIT_VALUE;

//=============================================================================
// Structures
//=============================================================================

//
// Description:
//	 An image. It is comparable to the Image class in the C++ library.
//   The fields in this structure should be considered read only.
//
public static class fc2Image extends Pointer {
    static { Loader.load(); }
    public fc2Image() { allocate(); }
    public fc2Image(int size) { allocateArray(size); }
    public fc2Image(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2Image position(int position) {
        return (fc2Image)super.position(position);
    }

    public native @Cast("unsigned int") int rows(); public native fc2Image rows(int rows);
    public native @Cast("unsigned int") int cols(); public native fc2Image cols(int cols);
    public native @Cast("unsigned int") int stride(); public native fc2Image stride(int stride);
    public native @Cast("unsigned char*") BytePointer pData(); public native fc2Image pData(BytePointer pData);
    public native @Cast("unsigned int") int dataSize(); public native fc2Image dataSize(int dataSize);
    public native @Cast("unsigned int") int receivedDataSize(); public native fc2Image receivedDataSize(int receivedDataSize);
    public native @Cast("fc2PixelFormat") int format(); public native fc2Image format(int format);
    public native @Cast("fc2BayerTileFormat") int bayerFormat(); public native fc2Image bayerFormat(int bayerFormat);

    public native @ByRef fc2ImageImpl imageImpl(); public native fc2Image imageImpl(fc2ImageImpl imageImpl);

}

public static class fc2SystemInfo extends Pointer {
    static { Loader.load(); }
    public fc2SystemInfo() { allocate(); }
    public fc2SystemInfo(int size) { allocateArray(size); }
    public fc2SystemInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2SystemInfo position(int position) {
        return (fc2SystemInfo)super.position(position);
    }

	public native @Cast("fc2OSType") int osType(); public native fc2SystemInfo osType(int osType);
	public native @Cast("char") byte osDescription(int i); public native fc2SystemInfo osDescription(int i, byte osDescription);
	@MemberGetter public native @Cast("char*") BytePointer osDescription();
	public native @Cast("fc2ByteOrder") int byteOrder(); public native fc2SystemInfo byteOrder(int byteOrder);
	public native @Cast("size_t") long sysMemSize(); public native fc2SystemInfo sysMemSize(long sysMemSize);
	public native @Cast("char") byte cpuDescription(int i); public native fc2SystemInfo cpuDescription(int i, byte cpuDescription);
	@MemberGetter public native @Cast("char*") BytePointer cpuDescription();
	public native @Cast("size_t") long numCpuCores(); public native fc2SystemInfo numCpuCores(long numCpuCores);
	public native @Cast("char") byte driverList(int i); public native fc2SystemInfo driverList(int i, byte driverList);
	@MemberGetter public native @Cast("char*") BytePointer driverList();
	public native @Cast("char") byte libraryList(int i); public native fc2SystemInfo libraryList(int i, byte libraryList);
	@MemberGetter public native @Cast("char*") BytePointer libraryList();
	public native @Cast("char") byte gpuDescription(int i); public native fc2SystemInfo gpuDescription(int i, byte gpuDescription);
	@MemberGetter public native @Cast("char*") BytePointer gpuDescription();
	public native @Cast("size_t") long screenWidth(); public native fc2SystemInfo screenWidth(long screenWidth);
	public native @Cast("size_t") long screenHeight(); public native fc2SystemInfo screenHeight(long screenHeight);
    public native @Cast("unsigned int") int reserved(int i); public native fc2SystemInfo reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2Version extends Pointer {
    static { Loader.load(); }
    public fc2Version() { allocate(); }
    public fc2Version(int size) { allocateArray(size); }
    public fc2Version(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2Version position(int position) {
        return (fc2Version)super.position(position);
    }

    public native @Cast("unsigned int") int major(); public native fc2Version major(int major);
    public native @Cast("unsigned int") int minor(); public native fc2Version minor(int minor);
    public native @Cast("unsigned int") int type(); public native fc2Version type(int type);
    public native @Cast("unsigned int") int build(); public native fc2Version build(int build);
}

public static class fc2Config extends Pointer {
    static { Loader.load(); }
    public fc2Config() { allocate(); }
    public fc2Config(int size) { allocateArray(size); }
    public fc2Config(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2Config position(int position) {
        return (fc2Config)super.position(position);
    }

    public native @Cast("unsigned int") int numBuffers(); public native fc2Config numBuffers(int numBuffers);
    public native @Cast("unsigned int") int numImageNotifications(); public native fc2Config numImageNotifications(int numImageNotifications);
    public native @Cast("unsigned int") int minNumImageNotifications(); public native fc2Config minNumImageNotifications(int minNumImageNotifications);
    public native int grabTimeout(); public native fc2Config grabTimeout(int grabTimeout);
    public native @Cast("fc2GrabMode") int grabMode(); public native fc2Config grabMode(int grabMode); 
    public native @Cast("fc2BusSpeed") int isochBusSpeed(); public native fc2Config isochBusSpeed(int isochBusSpeed);
    public native @Cast("fc2BusSpeed") int asyncBusSpeed(); public native fc2Config asyncBusSpeed(int asyncBusSpeed);
    public native @Cast("fc2BandwidthAllocation") int bandwidthAllocation(); public native fc2Config bandwidthAllocation(int bandwidthAllocation);
	public native @Cast("unsigned int") int registerTimeoutRetries(); public native fc2Config registerTimeoutRetries(int registerTimeoutRetries);
	public native @Cast("unsigned int") int registerTimeout(); public native fc2Config registerTimeout(int registerTimeout);
    public native @Cast("unsigned int") int reserved(int i); public native fc2Config reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

@Name("fc2TriggerDelayInfo") public static class fc2PropertyInfo extends Pointer {
    static { Loader.load(); }
    public fc2PropertyInfo() { allocate(); }
    public fc2PropertyInfo(int size) { allocateArray(size); }
    public fc2PropertyInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2PropertyInfo position(int position) {
        return (fc2PropertyInfo)super.position(position);
    }

    public native @Cast("fc2PropertyType") int type(); public native fc2PropertyInfo type(int type);
    public native @Cast("BOOL") int present(); public native fc2PropertyInfo present(int present);
    public native @Cast("BOOL") int autoSupported(); public native fc2PropertyInfo autoSupported(int autoSupported);
    public native @Cast("BOOL") int manualSupported(); public native fc2PropertyInfo manualSupported(int manualSupported);
    public native @Cast("BOOL") int onOffSupported(); public native fc2PropertyInfo onOffSupported(int onOffSupported);
    public native @Cast("BOOL") int onePushSupported(); public native fc2PropertyInfo onePushSupported(int onePushSupported);
    public native @Cast("BOOL") int absValSupported(); public native fc2PropertyInfo absValSupported(int absValSupported);
    public native @Cast("BOOL") int readOutSupported(); public native fc2PropertyInfo readOutSupported(int readOutSupported);
    public native @Cast("unsigned int") int min(); public native fc2PropertyInfo min(int min);
    public native @Cast("unsigned int") int max(); public native fc2PropertyInfo max(int max);
    public native float absMin(); public native fc2PropertyInfo absMin(float absMin);
    public native float absMax(); public native fc2PropertyInfo absMax(float absMax);
    public native @Cast("char") byte pUnits(int i); public native fc2PropertyInfo pUnits(int i, byte pUnits);
    @MemberGetter public native @Cast("char*") BytePointer pUnits();
    public native @Cast("char") byte pUnitAbbr(int i); public native fc2PropertyInfo pUnitAbbr(int i, byte pUnitAbbr);
    @MemberGetter public native @Cast("char*") BytePointer pUnitAbbr();
    public native @Cast("unsigned int") int reserved(int i); public native fc2PropertyInfo reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}   

@Name("fc2TriggerDelay") public static class fc2Property extends Pointer {
    static { Loader.load(); }
    public fc2Property() { allocate(); }
    public fc2Property(int size) { allocateArray(size); }
    public fc2Property(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2Property position(int position) {
        return (fc2Property)super.position(position);
    }

    public native @Cast("fc2PropertyType") int type(); public native fc2Property type(int type);
    public native @Cast("BOOL") int present(); public native fc2Property present(int present);
    public native @Cast("BOOL") int absControl(); public native fc2Property absControl(int absControl);
    public native @Cast("BOOL") int onePush(); public native fc2Property onePush(int onePush);
    public native @Cast("BOOL") int onOff(); public native fc2Property onOff(int onOff);
    public native @Cast("BOOL") int autoManualMode(); public native fc2Property autoManualMode(int autoManualMode);
    public native @Cast("unsigned int") int valueA(); public native fc2Property valueA(int valueA);  
    public native @Cast("unsigned int") int valueB(); public native fc2Property valueB(int valueB); //Applies only to the white balance blue value. Use
    //Value A for the red value.
    public native float absValue(); public native fc2Property absValue(float absValue);
    public native @Cast("unsigned int") int reserved(int i); public native fc2Property reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

    // For convenience, trigger delay is the same structure
    // used in a separate function along with trigger mode.

}

public static class fc2TriggerModeInfo extends Pointer {
    static { Loader.load(); }
    public fc2TriggerModeInfo() { allocate(); }
    public fc2TriggerModeInfo(int size) { allocateArray(size); }
    public fc2TriggerModeInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2TriggerModeInfo position(int position) {
        return (fc2TriggerModeInfo)super.position(position);
    }

    public native @Cast("BOOL") int present(); public native fc2TriggerModeInfo present(int present);
    public native @Cast("BOOL") int readOutSupported(); public native fc2TriggerModeInfo readOutSupported(int readOutSupported);
    public native @Cast("BOOL") int onOffSupported(); public native fc2TriggerModeInfo onOffSupported(int onOffSupported);
    public native @Cast("BOOL") int polaritySupported(); public native fc2TriggerModeInfo polaritySupported(int polaritySupported);
    public native @Cast("BOOL") int valueReadable(); public native fc2TriggerModeInfo valueReadable(int valueReadable);
    public native @Cast("unsigned int") int sourceMask(); public native fc2TriggerModeInfo sourceMask(int sourceMask);
    public native @Cast("BOOL") int softwareTriggerSupported(); public native fc2TriggerModeInfo softwareTriggerSupported(int softwareTriggerSupported);
    public native @Cast("unsigned int") int modeMask(); public native fc2TriggerModeInfo modeMask(int modeMask);
    public native @Cast("unsigned int") int reserved(int i); public native fc2TriggerModeInfo reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2TriggerMode extends Pointer {
    static { Loader.load(); }
    public fc2TriggerMode() { allocate(); }
    public fc2TriggerMode(int size) { allocateArray(size); }
    public fc2TriggerMode(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2TriggerMode position(int position) {
        return (fc2TriggerMode)super.position(position);
    }
      
    public native @Cast("BOOL") int onOff(); public native fc2TriggerMode onOff(int onOff);
    public native @Cast("unsigned int") int polarity(); public native fc2TriggerMode polarity(int polarity);
    public native @Cast("unsigned int") int source(); public native fc2TriggerMode source(int source);
    public native @Cast("unsigned int") int mode(); public native fc2TriggerMode mode(int mode);
    public native @Cast("unsigned int") int parameter(); public native fc2TriggerMode parameter(int parameter);      
    public native @Cast("unsigned int") int reserved(int i); public native fc2TriggerMode reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2StrobeInfo extends Pointer {
    static { Loader.load(); }
    public fc2StrobeInfo() { allocate(); }
    public fc2StrobeInfo(int size) { allocateArray(size); }
    public fc2StrobeInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2StrobeInfo position(int position) {
        return (fc2StrobeInfo)super.position(position);
    }

    public native @Cast("unsigned int") int source(); public native fc2StrobeInfo source(int source);
    public native @Cast("BOOL") int present(); public native fc2StrobeInfo present(int present);
    public native @Cast("BOOL") int readOutSupported(); public native fc2StrobeInfo readOutSupported(int readOutSupported);
    public native @Cast("BOOL") int onOffSupported(); public native fc2StrobeInfo onOffSupported(int onOffSupported);
    public native @Cast("BOOL") int polaritySupported(); public native fc2StrobeInfo polaritySupported(int polaritySupported);
    public native float minValue(); public native fc2StrobeInfo minValue(float minValue);
    public native float maxValue(); public native fc2StrobeInfo maxValue(float maxValue);
    public native @Cast("unsigned int") int reserved(int i); public native fc2StrobeInfo reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2StrobeControl extends Pointer {
    static { Loader.load(); }
    public fc2StrobeControl() { allocate(); }
    public fc2StrobeControl(int size) { allocateArray(size); }
    public fc2StrobeControl(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2StrobeControl position(int position) {
        return (fc2StrobeControl)super.position(position);
    }
      
    public native @Cast("unsigned int") int source(); public native fc2StrobeControl source(int source);
    public native @Cast("BOOL") int onOff(); public native fc2StrobeControl onOff(int onOff);
    public native @Cast("unsigned int") int polarity(); public native fc2StrobeControl polarity(int polarity);
    public native float delay(); public native fc2StrobeControl delay(float delay);
    public native float duration(); public native fc2StrobeControl duration(float duration);
    public native @Cast("unsigned int") int reserved(int i); public native fc2StrobeControl reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2Format7ImageSettings extends Pointer {
    static { Loader.load(); }
    public fc2Format7ImageSettings() { allocate(); }
    public fc2Format7ImageSettings(int size) { allocateArray(size); }
    public fc2Format7ImageSettings(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2Format7ImageSettings position(int position) {
        return (fc2Format7ImageSettings)super.position(position);
    }

    public native @Cast("fc2Mode") int mode(); public native fc2Format7ImageSettings mode(int mode);
    public native @Cast("unsigned int") int offsetX(); public native fc2Format7ImageSettings offsetX(int offsetX);
    public native @Cast("unsigned int") int offsetY(); public native fc2Format7ImageSettings offsetY(int offsetY);
    public native @Cast("unsigned int") int width(); public native fc2Format7ImageSettings width(int width);
    public native @Cast("unsigned int") int height(); public native fc2Format7ImageSettings height(int height);
    public native @Cast("fc2PixelFormat") int pixelFormat(); public native fc2Format7ImageSettings pixelFormat(int pixelFormat);
    public native @Cast("unsigned int") int reserved(int i); public native fc2Format7ImageSettings reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2Format7Info extends Pointer {
    static { Loader.load(); }
    public fc2Format7Info() { allocate(); }
    public fc2Format7Info(int size) { allocateArray(size); }
    public fc2Format7Info(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2Format7Info position(int position) {
        return (fc2Format7Info)super.position(position);
    }

    public native @Cast("fc2Mode") int mode(); public native fc2Format7Info mode(int mode);

    public native @Cast("unsigned int") int maxWidth(); public native fc2Format7Info maxWidth(int maxWidth);
    public native @Cast("unsigned int") int maxHeight(); public native fc2Format7Info maxHeight(int maxHeight);
    public native @Cast("unsigned int") int offsetHStepSize(); public native fc2Format7Info offsetHStepSize(int offsetHStepSize);
    public native @Cast("unsigned int") int offsetVStepSize(); public native fc2Format7Info offsetVStepSize(int offsetVStepSize);
    public native @Cast("unsigned int") int imageHStepSize(); public native fc2Format7Info imageHStepSize(int imageHStepSize);
    public native @Cast("unsigned int") int imageVStepSize(); public native fc2Format7Info imageVStepSize(int imageVStepSize);
    public native @Cast("unsigned int") int pixelFormatBitField(); public native fc2Format7Info pixelFormatBitField(int pixelFormatBitField);
    public native @Cast("unsigned int") int vendorPixelFormatBitField(); public native fc2Format7Info vendorPixelFormatBitField(int vendorPixelFormatBitField);
    public native @Cast("unsigned int") int packetSize(); public native fc2Format7Info packetSize(int packetSize);
    public native @Cast("unsigned int") int minPacketSize(); public native fc2Format7Info minPacketSize(int minPacketSize);
    public native @Cast("unsigned int") int maxPacketSize(); public native fc2Format7Info maxPacketSize(int maxPacketSize);
    public native float percentage(); public native fc2Format7Info percentage(float percentage);
    public native @Cast("unsigned int") int reserved(int i); public native fc2Format7Info reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2Format7PacketInfo extends Pointer {
    static { Loader.load(); }
    public fc2Format7PacketInfo() { allocate(); }
    public fc2Format7PacketInfo(int size) { allocateArray(size); }
    public fc2Format7PacketInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2Format7PacketInfo position(int position) {
        return (fc2Format7PacketInfo)super.position(position);
    }

    public native @Cast("unsigned int") int recommendedBytesPerPacket(); public native fc2Format7PacketInfo recommendedBytesPerPacket(int recommendedBytesPerPacket);
    public native @Cast("unsigned int") int maxBytesPerPacket(); public native fc2Format7PacketInfo maxBytesPerPacket(int maxBytesPerPacket);
    public native @Cast("unsigned int") int unitBytesPerPacket(); public native fc2Format7PacketInfo unitBytesPerPacket(int unitBytesPerPacket);
    public native @Cast("unsigned int") int reserved(int i); public native fc2Format7PacketInfo reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2IPAddress extends Pointer {
    static { Loader.load(); }
    public fc2IPAddress() { allocate(); }
    public fc2IPAddress(int size) { allocateArray(size); }
    public fc2IPAddress(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2IPAddress position(int position) {
        return (fc2IPAddress)super.position(position);
    }

    public native @Cast("unsigned char") byte octets(int i); public native fc2IPAddress octets(int i, byte octets);
    @MemberGetter public native @Cast("unsigned char*") BytePointer octets();
}

public static class fc2MACAddress extends Pointer {
    static { Loader.load(); }
    public fc2MACAddress() { allocate(); }
    public fc2MACAddress(int size) { allocateArray(size); }
    public fc2MACAddress(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2MACAddress position(int position) {
        return (fc2MACAddress)super.position(position);
    }

    public native @Cast("unsigned char") byte octets(int i); public native fc2MACAddress octets(int i, byte octets);
    @MemberGetter public native @Cast("unsigned char*") BytePointer octets();
}

public static class fc2GigEProperty extends Pointer {
    static { Loader.load(); }
    public fc2GigEProperty() { allocate(); }
    public fc2GigEProperty(int size) { allocateArray(size); }
    public fc2GigEProperty(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2GigEProperty position(int position) {
        return (fc2GigEProperty)super.position(position);
    }

    public native @Cast("fc2GigEPropertyType") int propType(); public native fc2GigEProperty propType(int propType);        
    public native @Cast("BOOL") int isReadable(); public native fc2GigEProperty isReadable(int isReadable);
    public native @Cast("BOOL") int isWritable(); public native fc2GigEProperty isWritable(int isWritable);
    public native @Cast("unsigned int") int min(); public native fc2GigEProperty min(int min);
    public native @Cast("unsigned int") int max(); public native fc2GigEProperty max(int max);
    public native @Cast("unsigned int") int value(); public native fc2GigEProperty value(int value);

    public native @Cast("unsigned int") int reserved(int i); public native fc2GigEProperty reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();
}

public static class fc2GigEStreamChannel extends Pointer {
    static { Loader.load(); }
    public fc2GigEStreamChannel() { allocate(); }
    public fc2GigEStreamChannel(int size) { allocateArray(size); }
    public fc2GigEStreamChannel(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2GigEStreamChannel position(int position) {
        return (fc2GigEStreamChannel)super.position(position);
    }

    public native @Cast("unsigned int") int networkInterfaceIndex(); public native fc2GigEStreamChannel networkInterfaceIndex(int networkInterfaceIndex);
    public native @Cast("unsigned int") int hostPost(); public native fc2GigEStreamChannel hostPost(int hostPost);
    public native @Cast("BOOL") int doNotFragment(); public native fc2GigEStreamChannel doNotFragment(int doNotFragment);
    public native @Cast("unsigned int") int packetSize(); public native fc2GigEStreamChannel packetSize(int packetSize);
    public native @Cast("unsigned int") int interPacketDelay(); public native fc2GigEStreamChannel interPacketDelay(int interPacketDelay);      
    public native @ByRef fc2IPAddress destinationIpAddress(); public native fc2GigEStreamChannel destinationIpAddress(fc2IPAddress destinationIpAddress);
    public native @Cast("unsigned int") int sourcePort(); public native fc2GigEStreamChannel sourcePort(int sourcePort);

    public native @Cast("unsigned int") int reserved(int i); public native fc2GigEStreamChannel reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();
}

public static class fc2GigEConfig extends Pointer {
    static { Loader.load(); }
    public fc2GigEConfig() { allocate(); }
    public fc2GigEConfig(int size) { allocateArray(size); }
    public fc2GigEConfig(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2GigEConfig position(int position) {
        return (fc2GigEConfig)super.position(position);
    }

    /** Turn on/off packet resend functionality */
	public native @Cast("BOOL") int enablePacketResend(); public native fc2GigEConfig enablePacketResend(int enablePacketResend);
	/** The number of miliseconds to wait for each requested packet */
	public native @Cast("unsigned int") int timeoutForPacketResend(); public native fc2GigEConfig timeoutForPacketResend(int timeoutForPacketResend);
	/** The max number of packets that can be requested to be resend */
	public native @Cast("unsigned int") int maxPacketsToResend(); public native fc2GigEConfig maxPacketsToResend(int maxPacketsToResend);

    public native @Cast("unsigned int") int reserved(int i); public native fc2GigEConfig reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();
}

public static class fc2GigEImageSettingsInfo extends Pointer {
    static { Loader.load(); }
    public fc2GigEImageSettingsInfo() { allocate(); }
    public fc2GigEImageSettingsInfo(int size) { allocateArray(size); }
    public fc2GigEImageSettingsInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2GigEImageSettingsInfo position(int position) {
        return (fc2GigEImageSettingsInfo)super.position(position);
    }

    public native @Cast("unsigned int") int maxWidth(); public native fc2GigEImageSettingsInfo maxWidth(int maxWidth);
    public native @Cast("unsigned int") int maxHeight(); public native fc2GigEImageSettingsInfo maxHeight(int maxHeight);
    public native @Cast("unsigned int") int offsetHStepSize(); public native fc2GigEImageSettingsInfo offsetHStepSize(int offsetHStepSize);
    public native @Cast("unsigned int") int offsetVStepSize(); public native fc2GigEImageSettingsInfo offsetVStepSize(int offsetVStepSize);
    public native @Cast("unsigned int") int imageHStepSize(); public native fc2GigEImageSettingsInfo imageHStepSize(int imageHStepSize);
    public native @Cast("unsigned int") int imageVStepSize(); public native fc2GigEImageSettingsInfo imageVStepSize(int imageVStepSize);
    public native @Cast("unsigned int") int pixelFormatBitField(); public native fc2GigEImageSettingsInfo pixelFormatBitField(int pixelFormatBitField);
    public native @Cast("unsigned int") int vendorPixelFormatBitField(); public native fc2GigEImageSettingsInfo vendorPixelFormatBitField(int vendorPixelFormatBitField);

    public native @Cast("unsigned int") int reserved(int i); public native fc2GigEImageSettingsInfo reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();
}

public static class fc2GigEImageSettings extends Pointer {
    static { Loader.load(); }
    public fc2GigEImageSettings() { allocate(); }
    public fc2GigEImageSettings(int size) { allocateArray(size); }
    public fc2GigEImageSettings(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2GigEImageSettings position(int position) {
        return (fc2GigEImageSettings)super.position(position);
    }

    public native @Cast("unsigned int") int offsetX(); public native fc2GigEImageSettings offsetX(int offsetX);
    public native @Cast("unsigned int") int offsetY(); public native fc2GigEImageSettings offsetY(int offsetY);
    public native @Cast("unsigned int") int width(); public native fc2GigEImageSettings width(int width);
    public native @Cast("unsigned int") int height(); public native fc2GigEImageSettings height(int height);
    public native @Cast("fc2PixelFormat") int pixelFormat(); public native fc2GigEImageSettings pixelFormat(int pixelFormat);

    public native @Cast("unsigned int") int reserved(int i); public native fc2GigEImageSettings reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();
}

public static class fc2TimeStamp extends Pointer {
    static { Loader.load(); }
    public fc2TimeStamp() { allocate(); }
    public fc2TimeStamp(int size) { allocateArray(size); }
    public fc2TimeStamp(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2TimeStamp position(int position) {
        return (fc2TimeStamp)super.position(position);
    }

    public native long seconds(); public native fc2TimeStamp seconds(long seconds);
    public native @Cast("unsigned int") int microSeconds(); public native fc2TimeStamp microSeconds(int microSeconds);
    public native @Cast("unsigned int") int cycleSeconds(); public native fc2TimeStamp cycleSeconds(int cycleSeconds);
    public native @Cast("unsigned int") int cycleCount(); public native fc2TimeStamp cycleCount(int cycleCount);
    public native @Cast("unsigned int") int cycleOffset(); public native fc2TimeStamp cycleOffset(int cycleOffset);
    public native @Cast("unsigned int") int reserved(int i); public native fc2TimeStamp reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2ConfigROM extends Pointer {
    static { Loader.load(); }
    public fc2ConfigROM() { allocate(); }
    public fc2ConfigROM(int size) { allocateArray(size); }
    public fc2ConfigROM(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2ConfigROM position(int position) {
        return (fc2ConfigROM)super.position(position);
    }

    public native @Cast("unsigned int") int nodeVendorId(); public native fc2ConfigROM nodeVendorId(int nodeVendorId);
    public native @Cast("unsigned int") int chipIdHi(); public native fc2ConfigROM chipIdHi(int chipIdHi);
    public native @Cast("unsigned int") int chipIdLo(); public native fc2ConfigROM chipIdLo(int chipIdLo);
    public native @Cast("unsigned int") int unitSpecId(); public native fc2ConfigROM unitSpecId(int unitSpecId);
    public native @Cast("unsigned int") int unitSWVer(); public native fc2ConfigROM unitSWVer(int unitSWVer);
    public native @Cast("unsigned int") int unitSubSWVer(); public native fc2ConfigROM unitSubSWVer(int unitSubSWVer);
    public native @Cast("unsigned int") int vendorUniqueInfo_0(); public native fc2ConfigROM vendorUniqueInfo_0(int vendorUniqueInfo_0);
    public native @Cast("unsigned int") int vendorUniqueInfo_1(); public native fc2ConfigROM vendorUniqueInfo_1(int vendorUniqueInfo_1);
    public native @Cast("unsigned int") int vendorUniqueInfo_2(); public native fc2ConfigROM vendorUniqueInfo_2(int vendorUniqueInfo_2);
    public native @Cast("unsigned int") int vendorUniqueInfo_3(); public native fc2ConfigROM vendorUniqueInfo_3(int vendorUniqueInfo_3);
    public native @Cast("char") byte pszKeyword(int i); public native fc2ConfigROM pszKeyword(int i, byte pszKeyword);
    @MemberGetter public native @Cast("char*") BytePointer pszKeyword();
    public native @Cast("unsigned int") int reserved(int i); public native fc2ConfigROM reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2CameraInfo extends Pointer {
    static { Loader.load(); }
    public fc2CameraInfo() { allocate(); }
    public fc2CameraInfo(int size) { allocateArray(size); }
    public fc2CameraInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2CameraInfo position(int position) {
        return (fc2CameraInfo)super.position(position);
    }
      
    public native @Cast("unsigned int") int serialNumber(); public native fc2CameraInfo serialNumber(int serialNumber);
    public native @Cast("fc2InterfaceType") int interfaceType(); public native fc2CameraInfo interfaceType(int interfaceType);
    public native @Cast("fc2DriverType") int driverType(); public native fc2CameraInfo driverType(int driverType);
    public native @Cast("BOOL") int isColorCamera(); public native fc2CameraInfo isColorCamera(int isColorCamera);
    public native @Cast("char") byte modelName(int i); public native fc2CameraInfo modelName(int i, byte modelName);
    @MemberGetter public native @Cast("char*") BytePointer modelName();
    public native @Cast("char") byte vendorName(int i); public native fc2CameraInfo vendorName(int i, byte vendorName);
    @MemberGetter public native @Cast("char*") BytePointer vendorName();
    public native @Cast("char") byte sensorInfo(int i); public native fc2CameraInfo sensorInfo(int i, byte sensorInfo);
    @MemberGetter public native @Cast("char*") BytePointer sensorInfo();
    public native @Cast("char") byte sensorResolution(int i); public native fc2CameraInfo sensorResolution(int i, byte sensorResolution);
    @MemberGetter public native @Cast("char*") BytePointer sensorResolution();
    public native @Cast("char") byte driverName(int i); public native fc2CameraInfo driverName(int i, byte driverName);
    @MemberGetter public native @Cast("char*") BytePointer driverName();
    public native @Cast("char") byte firmwareVersion(int i); public native fc2CameraInfo firmwareVersion(int i, byte firmwareVersion);
    @MemberGetter public native @Cast("char*") BytePointer firmwareVersion();
    public native @Cast("char") byte firmwareBuildTime(int i); public native fc2CameraInfo firmwareBuildTime(int i, byte firmwareBuildTime);
    @MemberGetter public native @Cast("char*") BytePointer firmwareBuildTime();
    public native @Cast("fc2BusSpeed") int maximumBusSpeed(); public native fc2CameraInfo maximumBusSpeed(int maximumBusSpeed);
	public native @Cast("fc2PCIeBusSpeed") int pcieBusSpeed(); public native fc2CameraInfo pcieBusSpeed(int pcieBusSpeed);
    public native @Cast("fc2BayerTileFormat") int bayerTileFormat(); public native fc2CameraInfo bayerTileFormat(int bayerTileFormat);
    public native @Cast("unsigned short") short busNumber(); public native fc2CameraInfo busNumber(short busNumber);
    public native @Cast("unsigned short") short nodeNumber(); public native fc2CameraInfo nodeNumber(short nodeNumber);

    // IIDC specific information
    public native @Cast("unsigned int") int iidcVer(); public native fc2CameraInfo iidcVer(int iidcVer);
    public native @ByRef fc2ConfigROM configROM(); public native fc2CameraInfo configROM(fc2ConfigROM configROM);

    // GigE specific information
    public native @Cast("unsigned int") int gigEMajorVersion(); public native fc2CameraInfo gigEMajorVersion(int gigEMajorVersion);
    public native @Cast("unsigned int") int gigEMinorVersion(); public native fc2CameraInfo gigEMinorVersion(int gigEMinorVersion);
    public native @Cast("char") byte userDefinedName(int i); public native fc2CameraInfo userDefinedName(int i, byte userDefinedName);
    @MemberGetter public native @Cast("char*") BytePointer userDefinedName();
    public native @Cast("char") byte xmlURL1(int i); public native fc2CameraInfo xmlURL1(int i, byte xmlURL1);
    @MemberGetter public native @Cast("char*") BytePointer xmlURL1();
    public native @Cast("char") byte xmlURL2(int i); public native fc2CameraInfo xmlURL2(int i, byte xmlURL2);
    @MemberGetter public native @Cast("char*") BytePointer xmlURL2();
    public native @ByRef fc2MACAddress macAddress(); public native fc2CameraInfo macAddress(fc2MACAddress macAddress);
    public native @ByRef fc2IPAddress ipAddress(); public native fc2CameraInfo ipAddress(fc2IPAddress ipAddress);
    public native @ByRef fc2IPAddress subnetMask(); public native fc2CameraInfo subnetMask(fc2IPAddress subnetMask);
    public native @ByRef fc2IPAddress defaultGateway(); public native fc2CameraInfo defaultGateway(fc2IPAddress defaultGateway);

	/** Status/Content of CCP register */
	public native @Cast("unsigned int") int ccpStatus(); public native fc2CameraInfo ccpStatus(int ccpStatus);
	/** Local Application IP Address. */
    public native @Cast("unsigned int") int applicationIPAddress(); public native fc2CameraInfo applicationIPAddress(int applicationIPAddress);
    /** Local Application port. */
    public native @Cast("unsigned int") int applicationPort(); public native fc2CameraInfo applicationPort(int applicationPort);

    public native @Cast("unsigned int") int reserved(int i); public native fc2CameraInfo reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2EmbeddedImageInfoProperty extends Pointer {
    static { Loader.load(); }
    public fc2EmbeddedImageInfoProperty() { allocate(); }
    public fc2EmbeddedImageInfoProperty(int size) { allocateArray(size); }
    public fc2EmbeddedImageInfoProperty(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2EmbeddedImageInfoProperty position(int position) {
        return (fc2EmbeddedImageInfoProperty)super.position(position);
    }

    public native @Cast("BOOL") int available(); public native fc2EmbeddedImageInfoProperty available(int available);
    public native @Cast("BOOL") int onOff(); public native fc2EmbeddedImageInfoProperty onOff(int onOff);

}

public static class fc2EmbeddedImageInfo extends Pointer {
    static { Loader.load(); }
    public fc2EmbeddedImageInfo() { allocate(); }
    public fc2EmbeddedImageInfo(int size) { allocateArray(size); }
    public fc2EmbeddedImageInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2EmbeddedImageInfo position(int position) {
        return (fc2EmbeddedImageInfo)super.position(position);
    }

    public native @ByRef fc2EmbeddedImageInfoProperty timestamp(); public native fc2EmbeddedImageInfo timestamp(fc2EmbeddedImageInfoProperty timestamp);
    public native @ByRef fc2EmbeddedImageInfoProperty gain(); public native fc2EmbeddedImageInfo gain(fc2EmbeddedImageInfoProperty gain);
    public native @ByRef fc2EmbeddedImageInfoProperty shutter(); public native fc2EmbeddedImageInfo shutter(fc2EmbeddedImageInfoProperty shutter);
    public native @ByRef fc2EmbeddedImageInfoProperty brightness(); public native fc2EmbeddedImageInfo brightness(fc2EmbeddedImageInfoProperty brightness);
    public native @ByRef fc2EmbeddedImageInfoProperty exposure(); public native fc2EmbeddedImageInfo exposure(fc2EmbeddedImageInfoProperty exposure);
    public native @ByRef fc2EmbeddedImageInfoProperty whiteBalance(); public native fc2EmbeddedImageInfo whiteBalance(fc2EmbeddedImageInfoProperty whiteBalance);
    public native @ByRef fc2EmbeddedImageInfoProperty frameCounter(); public native fc2EmbeddedImageInfo frameCounter(fc2EmbeddedImageInfoProperty frameCounter);
    public native @ByRef fc2EmbeddedImageInfoProperty strobePattern(); public native fc2EmbeddedImageInfo strobePattern(fc2EmbeddedImageInfoProperty strobePattern);
    public native @ByRef fc2EmbeddedImageInfoProperty GPIOPinState(); public native fc2EmbeddedImageInfo GPIOPinState(fc2EmbeddedImageInfoProperty GPIOPinState);
    public native @ByRef fc2EmbeddedImageInfoProperty ROIPosition(); public native fc2EmbeddedImageInfo ROIPosition(fc2EmbeddedImageInfoProperty ROIPosition);

}

public static class fc2ImageMetadata extends Pointer {
    static { Loader.load(); }
    public fc2ImageMetadata() { allocate(); }
    public fc2ImageMetadata(int size) { allocateArray(size); }
    public fc2ImageMetadata(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2ImageMetadata position(int position) {
        return (fc2ImageMetadata)super.position(position);
    }

    public native @Cast("unsigned int") int embeddedTimeStamp(); public native fc2ImageMetadata embeddedTimeStamp(int embeddedTimeStamp);
    public native @Cast("unsigned int") int embeddedGain(); public native fc2ImageMetadata embeddedGain(int embeddedGain);
    public native @Cast("unsigned int") int embeddedShutter(); public native fc2ImageMetadata embeddedShutter(int embeddedShutter);
    public native @Cast("unsigned int") int embeddedBrightness(); public native fc2ImageMetadata embeddedBrightness(int embeddedBrightness);
    public native @Cast("unsigned int") int embeddedExposure(); public native fc2ImageMetadata embeddedExposure(int embeddedExposure);
    public native @Cast("unsigned int") int embeddedWhiteBalance(); public native fc2ImageMetadata embeddedWhiteBalance(int embeddedWhiteBalance);
    public native @Cast("unsigned int") int embeddedFrameCounter(); public native fc2ImageMetadata embeddedFrameCounter(int embeddedFrameCounter);
    public native @Cast("unsigned int") int embeddedStrobePattern(); public native fc2ImageMetadata embeddedStrobePattern(int embeddedStrobePattern);
    public native @Cast("unsigned int") int embeddedGPIOPinState(); public native fc2ImageMetadata embeddedGPIOPinState(int embeddedGPIOPinState);
    public native @Cast("unsigned int") int embeddedROIPosition(); public native fc2ImageMetadata embeddedROIPosition(int embeddedROIPosition);        
    public native @Cast("unsigned int") int reserved(int i); public native fc2ImageMetadata reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2LUTData extends Pointer {
    static { Loader.load(); }
    public fc2LUTData() { allocate(); }
    public fc2LUTData(int size) { allocateArray(size); }
    public fc2LUTData(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2LUTData position(int position) {
        return (fc2LUTData)super.position(position);
    }

    public native @Cast("BOOL") int supported(); public native fc2LUTData supported(int supported);
    public native @Cast("BOOL") int enabled(); public native fc2LUTData enabled(int enabled);
    public native @Cast("unsigned int") int numBanks(); public native fc2LUTData numBanks(int numBanks);
    public native @Cast("unsigned int") int numChannels(); public native fc2LUTData numChannels(int numChannels);
    public native @Cast("unsigned int") int inputBitDepth(); public native fc2LUTData inputBitDepth(int inputBitDepth);
    public native @Cast("unsigned int") int outputBitDepth(); public native fc2LUTData outputBitDepth(int outputBitDepth);
    public native @Cast("unsigned int") int numEntries(); public native fc2LUTData numEntries(int numEntries);
    public native @Cast("unsigned int") int reserved(int i); public native fc2LUTData reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2PNGOption extends Pointer {
    static { Loader.load(); }
    public fc2PNGOption() { allocate(); }
    public fc2PNGOption(int size) { allocateArray(size); }
    public fc2PNGOption(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2PNGOption position(int position) {
        return (fc2PNGOption)super.position(position);
    }

    public native @Cast("BOOL") int interlaced(); public native fc2PNGOption interlaced(int interlaced); 
    public native @Cast("unsigned int") int compressionLevel(); public native fc2PNGOption compressionLevel(int compressionLevel);
    public native @Cast("unsigned int") int reserved(int i); public native fc2PNGOption reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

 public static class fc2PPMOption extends Pointer {
    static { Loader.load(); }
    public fc2PPMOption() { allocate(); }
    public fc2PPMOption(int size) { allocateArray(size); }
    public fc2PPMOption(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2PPMOption position(int position) {
        return (fc2PPMOption)super.position(position);
    }

    public native @Cast("BOOL") int binaryFile(); public native fc2PPMOption binaryFile(int binaryFile);
    public native @Cast("unsigned int") int reserved(int i); public native fc2PPMOption reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

} 

public static class fc2PGMOption extends Pointer {
    static { Loader.load(); }
    public fc2PGMOption() { allocate(); }
    public fc2PGMOption(int size) { allocateArray(size); }
    public fc2PGMOption(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2PGMOption position(int position) {
        return (fc2PGMOption)super.position(position);
    }

    public native @Cast("BOOL") int binaryFile(); public native fc2PGMOption binaryFile(int binaryFile);
    public native @Cast("unsigned int") int reserved(int i); public native fc2PGMOption reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

/** enum fc2TIFFCompressionMethod */
public static final int
    FC2_TIFF_NONE = 1,
    FC2_TIFF_PACKBITS = 2,
    FC2_TIFF_DEFLATE = 3,
    FC2_TIFF_ADOBE_DEFLATE = 4,
    FC2_TIFF_CCITTFAX3 = 5,
    FC2_TIFF_CCITTFAX4 = 6,
    FC2_TIFF_LZW = 7,
    FC2_TIFF_JPEG = 8;

public static class fc2TIFFOption extends Pointer {
    static { Loader.load(); }
    public fc2TIFFOption() { allocate(); }
    public fc2TIFFOption(int size) { allocateArray(size); }
    public fc2TIFFOption(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2TIFFOption position(int position) {
        return (fc2TIFFOption)super.position(position);
    }

    public native @Cast("fc2TIFFCompressionMethod") int compression(); public native fc2TIFFOption compression(int compression); 
    public native @Cast("unsigned int") int reserved(int i); public native fc2TIFFOption reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2JPEGOption extends Pointer {
    static { Loader.load(); }
    public fc2JPEGOption() { allocate(); }
    public fc2JPEGOption(int size) { allocateArray(size); }
    public fc2JPEGOption(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2JPEGOption position(int position) {
        return (fc2JPEGOption)super.position(position);
    }

    public native @Cast("BOOL") int progressive(); public native fc2JPEGOption progressive(int progressive); 
    public native @Cast("unsigned int") int quality(); public native fc2JPEGOption quality(int quality);
    public native @Cast("unsigned int") int reserved(int i); public native fc2JPEGOption reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2JPG2Option extends Pointer {
    static { Loader.load(); }
    public fc2JPG2Option() { allocate(); }
    public fc2JPG2Option(int size) { allocateArray(size); }
    public fc2JPG2Option(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2JPG2Option position(int position) {
        return (fc2JPG2Option)super.position(position);
    }

    public native @Cast("unsigned int") int quality(); public native fc2JPG2Option quality(int quality);
    public native @Cast("unsigned int") int reserved(int i); public native fc2JPG2Option reserved(int i, int reserved);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();
}

public static class fc2AVIOption extends Pointer {
    static { Loader.load(); }
    public fc2AVIOption() { allocate(); }
    public fc2AVIOption(int size) { allocateArray(size); }
    public fc2AVIOption(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2AVIOption position(int position) {
        return (fc2AVIOption)super.position(position);
    }

   public native float frameRate(); public native fc2AVIOption frameRate(float frameRate);
   public native @Cast("unsigned int") int reserved(int i); public native fc2AVIOption reserved(int i, int reserved);
   @MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}

public static class fc2MJPGOption extends Pointer {
    static { Loader.load(); }
    public fc2MJPGOption() { allocate(); }
    public fc2MJPGOption(int size) { allocateArray(size); }
    public fc2MJPGOption(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2MJPGOption position(int position) {
        return (fc2MJPGOption)super.position(position);
    }

	public native float frameRate(); public native fc2MJPGOption frameRate(float frameRate);
	public native @Cast("unsigned int") int quality(); public native fc2MJPGOption quality(int quality);
	public native @Cast("unsigned int") int reserved(int i); public native fc2MJPGOption reserved(int i, int reserved);
	@MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}


public static class fc2H264Option extends Pointer {
    static { Loader.load(); }
    public fc2H264Option() { allocate(); }
    public fc2H264Option(int size) { allocateArray(size); }
    public fc2H264Option(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public fc2H264Option position(int position) {
        return (fc2H264Option)super.position(position);
    }

	public native float frameRate(); public native fc2H264Option frameRate(float frameRate);
	public native @Cast("unsigned int") int width(); public native fc2H264Option width(int width);
	public native @Cast("unsigned int") int height(); public native fc2H264Option height(int height);
	public native @Cast("unsigned int") int bitrate(); public native fc2H264Option bitrate(int bitrate);
	public native @Cast("unsigned int") int reserved(int i); public native fc2H264Option reserved(int i, int reserved);
	@MemberGetter public native @Cast("unsigned int*") IntPointer reserved();

}


//=============================================================================
// Callbacks
//=============================================================================

@Opaque public static class fc2CallbackHandle extends Pointer {
    public fc2CallbackHandle() { }
    public fc2CallbackHandle(Pointer p) { super(p); }
}
public static class fc2BusEventCallback extends FunctionPointer {
    static { Loader.load(); }
    public    fc2BusEventCallback(Pointer p) { super(p); }
    protected fc2BusEventCallback() { allocate(); }
    private native void allocate();
    public native void call( Pointer pParameter, @Cast("unsigned int") int serialNumber );
}
public static class fc2ImageEventCallback extends FunctionPointer {
    static { Loader.load(); }
    public    fc2ImageEventCallback(Pointer p) { super(p); }
    protected fc2ImageEventCallback() { allocate(); }
    private native void allocate();
    public native void call( fc2Image image, Pointer pCallbackData );
}
public static class fc2AsyncCommandCallback extends FunctionPointer {
    static { Loader.load(); }
    public    fc2AsyncCommandCallback(Pointer p) { super(p); }
    protected fc2AsyncCommandCallback() { allocate(); }
    private native void allocate();
    public native void call( @Cast("fc2Error") int retError, Pointer pUserData );
}

// #ifdef __cplusplus
// #endif

// #endif // PGR_FC2_FLYCAPTURE2DEFS_C_H



// Parsed from <FlyCapture2_C.h>

//=============================================================================
// Copyright � 2008 Point Grey Research, Inc. All Rights Reserved.
//
// This software is the confidential and proprietary information of Point
// Grey Research, Inc. ("Confidential Information").  You shall not
// disclose such Confidential Information and shall use it only in
// accordance with the terms of the license agreement you entered into
// with PGR.
//
// PGR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
// SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, OR NON-INFRINGEMENT. PGR SHALL NOT BE LIABLE FOR ANY DAMAGES
// SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
// THIS SOFTWARE OR ITS DERIVATIVES.
//=============================================================================
//=============================================================================
// $Id: FlyCapture2_C.h,v 1.69 2010-12-13 23:58:00 mgara Exp $
//=============================================================================

// #ifndef PGR_FC2_FLYCAPTURE2_C_H
// #define PGR_FC2_FLYCAPTURE2_C_H

//=============================================================================
// Global C header file for FlyCapture2. 
//
// This file defines the C API for FlyCapture2
//=============================================================================

// #include "FlyCapture2Platform_C.h"
// #include "FlyCapture2Defs_C.h"

// #ifdef __cplusplus
// #endif

/**
 * Create a FC2 context for IIDC camaera. 
 * This call must be made before any other calls that use a context
 * will succeed.
 *
 * @param pContext A pointer to the fc2Context to be created.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2CreateContext(
    @Cast("fc2Context*") @ByPtrPtr fc2Context pContext );

/**
 * Create a FC2 context for a GigE Vision camera. 
 * This call must be made before any other calls that use a context
 * will succeed.
 *
 * @param pContext A pointer to the fc2Context to be created.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2CreateGigEContext(
    @Cast("fc2Context*") @ByPtrPtr fc2Context pContext );

/**
 * Destroy the FC2 context. This must be called when the user is finished
 * with the context in order to prevent memory leaks.
 *
 * @param context The context to be destroyed.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2DestroyContext(
    fc2Context context );

/** 
 * Fire a bus reset. The actual bus reset is only fired for the 
 * specified 1394 bus, but it will effectively cause a global bus
 * reset for the library.
 *
 * @param context The fc2Context to be used.
 * @param pGuid PGRGuid of the camera or the device to cause bus reset.
 *
 * @return An Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2FireBusReset( 
    fc2Context context,
    fc2PGRGuid pGuid);

/**
 * Gets the number of cameras attached to the PC.
 *
 * @param context The fc2Context to be used.
 * @param pNumCameras Number of cameras detected.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetNumOfCameras(
    fc2Context context,
    @Cast("unsigned int*") IntPointer pNumCameras );
public static native @Cast("fc2Error") int fc2GetNumOfCameras(
    fc2Context context,
    @Cast("unsigned int*") IntBuffer pNumCameras );
public static native @Cast("fc2Error") int fc2GetNumOfCameras(
    fc2Context context,
    @Cast("unsigned int*") int[] pNumCameras );

/**
 * Query whether a GigE camera is controlable.
 *
 * @param context The fc2Context to be used.
 * @param pGuid Unique PGRGuid for the camera.
 * @param pControlable True indicates camera is controllable
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2IsCameraControlable(
    fc2Context context,
	fc2PGRGuid pGuid,
    @Cast("BOOL*") IntPointer pControlable );
public static native @Cast("fc2Error") int fc2IsCameraControlable(
    fc2Context context,
	fc2PGRGuid pGuid,
    @Cast("BOOL*") IntBuffer pControlable );
public static native @Cast("fc2Error") int fc2IsCameraControlable(
    fc2Context context,
	fc2PGRGuid pGuid,
    @Cast("BOOL*") int[] pControlable );

/**
 * Gets the PGRGuid for a camera on the PC. It uniquely identifies
 * the camera specified by the index and is used to identify the camera
 * during a fc2Connect() call.
 *
 * @param context The fc2Context to be used.
 * @param index Zero based index of camera.
 * @param pGuid Unique PGRGuid for the camera.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetCameraFromIndex(
    fc2Context context,
    @Cast("unsigned int") int index,
    fc2PGRGuid pGuid );

/**
 * Gets the PGRGuid for a camera on the PC. It uniquely identifies
 * the camera specified by the serial number and is used to identify the camera
 * during a fc2Connect() call.
 *
 * @param context The fc2Context to be used.
 * @param serialNumber Serial number of camera.
 * @param pGuid Unique PGRGuid for the camera.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetCameraFromSerialNumber(
    fc2Context context,
    @Cast("unsigned int") int serialNumber,
    fc2PGRGuid pGuid );

/**
 * Gets the serial number of the camera with the specified index.
 *
 * @param context The fc2Context to be used.
 * @param index Zero based index of desired camera.
 * @param pSerialNumber Serial number of camera.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetCameraSerialNumberFromIndex(
    fc2Context context,
    @Cast("unsigned int") int index,
    @Cast("unsigned int*") IntPointer pSerialNumber );
public static native @Cast("fc2Error") int fc2GetCameraSerialNumberFromIndex(
    fc2Context context,
    @Cast("unsigned int") int index,
    @Cast("unsigned int*") IntBuffer pSerialNumber );
public static native @Cast("fc2Error") int fc2GetCameraSerialNumberFromIndex(
    fc2Context context,
    @Cast("unsigned int") int index,
    @Cast("unsigned int*") int[] pSerialNumber );

/**
 * Gets the interface type associated with a PGRGuid. This is useful
 * in situations where there is a need to enumerate all cameras
 * for a particular interface.
 *
 * @param context The fc2Context to be used.
 * @param pGuid The PGRGuid to get the interface for.
 * @param pInterfaceType The interface type of the PGRGuid.
 *
 * @return
 */
public static native @Cast("fc2Error") int fc2GetInterfaceTypeFromGuid(
    fc2Context context,
    fc2PGRGuid pGuid,
    @Cast("fc2InterfaceType*") IntPointer pInterfaceType );
public static native @Cast("fc2Error") int fc2GetInterfaceTypeFromGuid(
    fc2Context context,
    fc2PGRGuid pGuid,
    @Cast("fc2InterfaceType*") IntBuffer pInterfaceType );
public static native @Cast("fc2Error") int fc2GetInterfaceTypeFromGuid(
    fc2Context context,
    fc2PGRGuid pGuid,
    @Cast("fc2InterfaceType*") int[] pInterfaceType );

/**
 * Gets the number of devices. This may include hubs, host controllers
 * and other hardware devices (including cameras).
 *
 * @param context The fc2Context to be used.
 * @param pNumDevices The number of devices found.
 *
 * @return An Error indicating the success or failure of the function.
 */ 
public static native @Cast("fc2Error") int fc2GetNumOfDevices( 
    fc2Context context,
    @Cast("unsigned int*") IntPointer pNumDevices );
public static native @Cast("fc2Error") int fc2GetNumOfDevices( 
    fc2Context context,
    @Cast("unsigned int*") IntBuffer pNumDevices );
public static native @Cast("fc2Error") int fc2GetNumOfDevices( 
    fc2Context context,
    @Cast("unsigned int*") int[] pNumDevices );

/**
 * Gets the PGRGuid for a device. It uniquely identifies the device
 * specified by the index.
 *
 * @param context The fc2Context to be used.
 * @param index Zero based index of device.
 * @param pGuid Unique PGRGuid for the device.
 *
 * @see fc2GetNumOfDevices()
 *
 * @return An Error indicating the success or failure of the function.
 */ 
public static native @Cast("fc2Error") int fc2GetDeviceFromIndex( 
    fc2Context context,
    @Cast("unsigned int") int index, 
    fc2PGRGuid pGuid );

/**
 * Register a callback function that will be called when the
 * specified callback event occurs.
 *
 * @param context The fc2Context to be used.
 * @param enumCallback Pointer to function that will receive the callback.
 * @param callbackType Type of callback to register for.
 * @param pParameter Callback parameter to be passed to callback.
 * @param pCallbackHandle Unique callback handle used for unregistering 
 *                        callback.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2RegisterCallback(
    fc2Context context,
    fc2BusEventCallback enumCallback,
    @Cast("fc2BusCallbackType") int callbackType,
    Pointer pParameter,
    fc2CallbackHandle pCallbackHandle );

/**
 * Unregister a callback function.
 *
 * @param context The fc2Context to be used.
 * @param callbackHandle Unique callback handle.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2UnregisterCallback(
    fc2Context context,
    @ByVal fc2CallbackHandle callbackHandle );

/**
 * Force a rescan of the buses. This does not trigger a bus reset.
 * However, any current connections to a Camera object will be
 * invalidated.
 *
 * @return An Error indicating the success or failure of the function.
 */ 
public static native @Cast("fc2Error") int fc2RescanBus( fc2Context context);

/**
 * Force the camera with the specific MAC address to the specified
 * IP address, subnet mask and default gateway. This is useful in
 * situations where a GigE Vision camera is using Persistent IP and the
 * application's subnet is different from the device subnet.
 *
 * @param context The fc2Context to be used.
 * @param macAddress MAC address of the camera.
 * @param ipAddress IP address to set on the camera.
 * @param subnetMask Subnet mask to set on the camera.
 * @param defaultGateway Default gateway to set on the camera.
 *
 * @return An Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2ForceIPAddressToCamera( 
    fc2Context context,
    @ByVal fc2MACAddress macAddress,
    @ByVal fc2IPAddress ipAddress,
    @ByVal fc2IPAddress subnetMask,
    @ByVal fc2IPAddress defaultGateway );

/**
* Force all cameras on the network to be assigned sequential IP addresses
* on the same subnet as the netowrk adapters that they are connected to. 
* This is useful in situations where a GigE Vision cameras are using 
* Persistent IP addresses and the application's subnet is different from 
* the devices.
*
* @return An Error indicating the success or failure of the function.
*/
public static native @Cast("fc2Error") int fc2ForceAllIPAddressesAutomatically();

/**
* Force a cameras on the network to be assigned sequential IP addresses
* on the same subnet as the netowrk adapters that it is connected to. 
* This is useful in situations where a GigE Vision cameras is using 
* Persistent IP addresses and the application's subnet is different from 
* the device.
*
* @return An Error indicating the success or failure of the function.
*/
public static native @Cast("fc2Error") int fc2ForceIPAddressAutomatically(@Cast("unsigned int") int serialNumber);

/**
 * Discover all cameras connected to the network even if they reside
 * on a different subnet. This is useful in situations where a GigE
 * camera is using Persistent IP and the application's subnet is
 * different from the device subnet. After discovering the camera,  
 * it is easy to use ForceIPAddressToCamera() to set a different IP 
 * configuration.
 *
 * @param context The fc2Context to be used.
 * @param gigECameras Pointer to an array of CameraInfo structures.
 * @param arraySize Size of the array. Number of discovered cameras
 *                  is returned in the same value.
 *
 * @return An Error indicating the success or failure of the function.
 *         If the error is PGRERROR_BUFFER_TOO_SMALL then arraySize will
 *         contain the minimum size needed for gigECameras array.
 */
public static native @Cast("fc2Error") int fc2DiscoverGigECameras( 
    fc2Context context,
    fc2CameraInfo gigECameras,
    @Cast("unsigned int*") IntPointer arraySize  );
public static native @Cast("fc2Error") int fc2DiscoverGigECameras( 
    fc2Context context,
    fc2CameraInfo gigECameras,
    @Cast("unsigned int*") IntBuffer arraySize  );
public static native @Cast("fc2Error") int fc2DiscoverGigECameras( 
    fc2Context context,
    fc2CameraInfo gigECameras,
    @Cast("unsigned int*") int[] arraySize  );

/**
 * Write to the specified register on the camera.
 *
 * @param context The fc2Context to be used.
 * @param address DCAM address to be written to.
 * @param value The value to be written.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2WriteRegister(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("unsigned int") int value);

/**
 * Write to the specified register on the camera with broadcast.
 *
 * @param context The fc2Context to be used.
 * @param address DCAM address to be written to.
 * @param value The value to be written.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2WriteRegisterBroadcast(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("unsigned int") int value);

/**
 * Read the specified register from the camera.
 *
 * @param context The fc2Context to be used.
 * @param address DCAM address to be read from.
 * @param pValue The value that is read.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2ReadRegister(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("unsigned int*") IntPointer pValue );
public static native @Cast("fc2Error") int fc2ReadRegister(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("unsigned int*") IntBuffer pValue );
public static native @Cast("fc2Error") int fc2ReadRegister(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("unsigned int*") int[] pValue );

/**
 * Write to the specified register block on the camera.
 *
 * @param context The fc2Context to be used.
 * @param addressHigh Top 16 bits of the 48 bit absolute address to write to.
 * @param addressLow Bottom 32 bits of the 48 bits absolute address to write to.              
 * @param pBuffer Array containing data to be written.
 * @param length Size of array, in quadlets.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2WriteRegisterBlock(
    fc2Context context,
    @Cast("unsigned short") short addressHigh,
    @Cast("unsigned int") int addressLow,
    @Cast("const unsigned int*") IntPointer pBuffer,
    @Cast("unsigned int") int length );
public static native @Cast("fc2Error") int fc2WriteRegisterBlock(
    fc2Context context,
    @Cast("unsigned short") short addressHigh,
    @Cast("unsigned int") int addressLow,
    @Cast("const unsigned int*") IntBuffer pBuffer,
    @Cast("unsigned int") int length );
public static native @Cast("fc2Error") int fc2WriteRegisterBlock(
    fc2Context context,
    @Cast("unsigned short") short addressHigh,
    @Cast("unsigned int") int addressLow,
    @Cast("const unsigned int*") int[] pBuffer,
    @Cast("unsigned int") int length );

/**
 * Write to the specified register block on the camera.
 *
 * @param context The fc2Context to be used.
 * @param addressHigh Top 16 bits of the 48 bit absolute address to read from.
 * @param addressLow Bottom 32 bits of the 48 bits absolute address to read from.                
 * @param pBuffer Array to store read data.
 * @param length Size of array, in quadlets.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2ReadRegisterBlock(
    fc2Context context,
    @Cast("unsigned short") short addressHigh,
    @Cast("unsigned int") int addressLow,
    @Cast("unsigned int*") IntPointer pBuffer,
    @Cast("unsigned int") int length );
public static native @Cast("fc2Error") int fc2ReadRegisterBlock(
    fc2Context context,
    @Cast("unsigned short") short addressHigh,
    @Cast("unsigned int") int addressLow,
    @Cast("unsigned int*") IntBuffer pBuffer,
    @Cast("unsigned int") int length );
public static native @Cast("fc2Error") int fc2ReadRegisterBlock(
    fc2Context context,
    @Cast("unsigned short") short addressHigh,
    @Cast("unsigned int") int addressLow,
    @Cast("unsigned int*") int[] pBuffer,
    @Cast("unsigned int") int length );     

/**
 * Connects the camera object to the camera specified by the GUID.
 *
 * @param context The fc2Context to be used.
 * @param guid The unique identifier for a specific camera on the PC.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2Connect( 
    fc2Context context,
    fc2PGRGuid guid );

/**
 * Disconnects the fc2Context from the camera.
 *
 * @param context The fc2Context to be used.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2Disconnect( 
    fc2Context context );

/**
 * Sets the callback data to be used on completion of image transfer.
 * To clear the current stored callback data, pass in NULL for both
 * callback arguments.
 *
 * @param context The fc2Context to be used.
 * @param pCallbackFn A function to be called when a new image is received.
 * @param pCallbackData A pointer to data that can be passed to the
 *                      callback function.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetCallback(
    fc2Context context,
    fc2ImageEventCallback pCallbackFn,
    Pointer pCallbackData);

/**
 * Starts isochronous image capture. It will use either the current
 * video mode or the most recently set video mode of the camera.
 *
 * @param context The fc2Context to be used.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2StartCapture( 
    fc2Context context );

/**
 * Starts isochronous image capture. It will use either the current
 * video mode or the most recently set video mode of the camera. The callback
 * function is called when a new image is received from the camera.
 *
 * @param context The fc2Context to be used.
 * @param pCallbackFn A function to be called when a new image is received.        
 * @param pCallbackData A pointer to data that can be passed to the
 *                      callback function. A NULL pointer is acceptable.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2StartCaptureCallback(
    fc2Context context,
    fc2ImageEventCallback pCallbackFn,
    Pointer pCallbackData);

/**
 * Starts synchronized isochronous image capture on multiple cameras.
 *
 * @param numCameras Number of fc2Contexts in the ppCameras array.
 * @param pContexts Array of fc2Contexts.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2StartSyncCapture( 
    @Cast("unsigned int") int numCameras, 
    @Cast("fc2Context*") @ByPtrPtr fc2Context pContexts );

/**
 * Starts synchronized isochronous image capture on multiple cameras.
 *
 * @param numCameras Number of fc2Contexts in the ppCameras array.
 * @param pContexts Array of fc2Contexts.
 * @param pCallbackFns Array of callback functions for each camera.
 * @param pCallbackDataArray Array of callback data pointers. 
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2StartSyncCaptureCallback( 
    @Cast("unsigned int") int numCameras, 
    @Cast("fc2Context*") @ByPtrPtr fc2Context pContexts,
    @Cast("fc2ImageEventCallback*") @ByPtrPtr fc2ImageEventCallback pCallbackFns,
    @Cast("void**") PointerPointer pCallbackDataArray);
public static native @Cast("fc2Error") int fc2StartSyncCaptureCallback( 
    @Cast("unsigned int") int numCameras, 
    @Cast("fc2Context*") @ByPtrPtr fc2Context pContexts,
    @Cast("fc2ImageEventCallback*") @ByPtrPtr fc2ImageEventCallback pCallbackFns,
    @Cast("void**") @ByPtrPtr Pointer pCallbackDataArray);

/**
 * Retrieves the the next image object containing the next image.
 *
 * @param context The fc2Context to be used.
 * @param pImage Pointer to fc2Image to store image data.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2RetrieveBuffer( 
    fc2Context context,
    fc2Image pImage );

/**
 * Stops isochronous image transfer and cleans up all associated
 * resources.
 *
 * @param context The fc2Context to be used.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2StopCapture( 
    fc2Context context );

/**
 * Specify user allocated buffers to use as image data buffers.
 *
 * @param context The fc2Context to be used.
 * @param ppMemBuffers Pointer to memory buffers to be written to. The
 *                     size of the data should be equal to 
 *                     (size * numBuffers) or larger.
 * @param size The size of each buffer (in bytes).
 * @param nNumBuffers Number of buffers in the array.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetUserBuffers(
    fc2Context context,
    @Cast("unsigned char*const") BytePointer ppMemBuffers,
    int size,
    int nNumBuffers );
public static native @Cast("fc2Error") int fc2SetUserBuffers(
    fc2Context context,
    @Cast("unsigned char*const") ByteBuffer ppMemBuffers,
    int size,
    int nNumBuffers );
public static native @Cast("fc2Error") int fc2SetUserBuffers(
    fc2Context context,
    @Cast("unsigned char*const") byte[] ppMemBuffers,
    int size,
    int nNumBuffers );

/**
 * Get the configuration associated with the camera.
 *
 * @param context The fc2Context to be used.
 * @param config Pointer to the configuration structure to be filled.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetConfiguration( 
    fc2Context context,
    fc2Config config );

/**
 * Set the configuration associated with the camera.
 *
 * @param context The fc2Context to be used.
 * @param config Pointer to the configuration structure to be used.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetConfiguration( 
    fc2Context context,
    fc2Config config );

/**
 * Retrieves information from the camera such as serial number, model
 * name and other camera information.
 *
 * @param context The fc2Context to be used.
 * @param pCameraInfo Pointer to the camera information structure
 *                    to be filled.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetCameraInfo( 
    fc2Context context,
    fc2CameraInfo pCameraInfo );

/**
 * Retrieves information about the specified camera property. The 
 * property type must be specified in the fc2PropertyInfo structure 
 * passed into the function in order for the function to succeed.
 *
 * @param context The fc2Context to be used.
 * @param propInfo Pointer to the PropertyInfo structure to be filled.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetPropertyInfo( 
    fc2Context context,
    fc2PropertyInfo propInfo );

/**
 * Reads the settings for the specified property from the camera. The 
 * property type must be specified in the fc2Property structure passed
 * into the function in order for the function to succeed. If auto
 * is on, the integer and abs values returned may not be consistent
 * with each other.
 *
 * @param context The fc2Context to be used.
 * @param prop Pointer to the Property structure to be filled.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetProperty( 
    fc2Context context,
    fc2Property prop );

/**
 * Writes the settings for the specified property to the camera. The 
 * property type must be specified in the Property structure passed
 * into the function in order for the function to succeed.
 * The absControl flag controls whether the absolute or integer value
 * is written to the camera.
 *
 * @param context The fc2Context to be used.
 * @param prop Pointer to the Property structure to be used.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetProperty( 
    fc2Context context,
    fc2Property prop );

/**
 * Writes the settings for the specified property to the camera. The 
 * property type must be specified in the Property structure passed
 * into the function in order for the function to succeed.
 * The absControl flag controls whether the absolute or integer value
 * is written to the camera.
 *
 * @param context The fc2Context to be used.
 * @param prop Pointer to the Property structure to be used.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetPropertyBroadcast( 
    fc2Context context,
    fc2Property prop );

/**
 * Get the GPIO pin direction for the specified pin. This is not a
 * required call when using the trigger or strobe functions as
 * the pin direction is set automatically internally.
 *
 * @param context The fc2Context to be used.
 * @param pin Pin to get the direction for.
 * @param pDirection Direction of the pin. 0 for input, 1 for output.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetGPIOPinDirection( 
    fc2Context context,
    @Cast("unsigned int") int pin, 
    @Cast("unsigned int*") IntPointer pDirection );
public static native @Cast("fc2Error") int fc2GetGPIOPinDirection( 
    fc2Context context,
    @Cast("unsigned int") int pin, 
    @Cast("unsigned int*") IntBuffer pDirection );
public static native @Cast("fc2Error") int fc2GetGPIOPinDirection( 
    fc2Context context,
    @Cast("unsigned int") int pin, 
    @Cast("unsigned int*") int[] pDirection );

/**
 * Set the GPIO pin direction for the specified pin. This is useful if
 * there is a need to set the pin into an input pin (i.e. to read the
 * voltage) off the pin without setting it as a trigger source. This 
 * is not a required call when using the trigger or strobe functions as
 * the pin direction is set automatically internally.
 *
 * @param context The fc2Context to be used.
 * @param pin Pin to get the direction for.
 * @param direction Direction of the pin. 0 for input, 1 for output.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetGPIOPinDirection( 
    fc2Context context,
    @Cast("unsigned int") int pin, 
    @Cast("unsigned int") int direction);

/**
 * Set the GPIO pin direction for the specified pin. This is useful if
 * there is a need to set the pin into an input pin (i.e. to read the
 * voltage) off the pin without setting it as a trigger source. This 
 * is not a required call when using the trigger or strobe functions as
 * the pin direction is set automatically internally.
 *
 * @param context The fc2Context to be used.
 * @param pin Pin to get the direction for.
 * @param direction Direction of the pin. 0 for input, 1 for output.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetGPIOPinDirectionBroadcast( 
    fc2Context context,
    @Cast("unsigned int") int pin, 
    @Cast("unsigned int") int direction);

/**
 * Retrieve trigger information from the camera.
 *
 * @param context The fc2Context to be used.
 * @param triggerModeInfo Structure to receive trigger information.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetTriggerModeInfo( 
    fc2Context context,
    fc2TriggerModeInfo triggerModeInfo );

/**
 * Retrieve current trigger settings from the camera.
 *
 * @param context The fc2Context to be used.
 * @param triggerMode Structure to receive trigger mode settings.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetTriggerMode( 
    fc2Context context,
    fc2TriggerMode triggerMode );

/**
 * Set the specified trigger settings to the camera.
 *
 * @param context The fc2Context to be used.
 * @param triggerMode Structure providing trigger mode settings.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetTriggerMode( 
    fc2Context context,
    fc2TriggerMode triggerMode );

/**
 * Set the specified trigger settings to the camera.
 *
 * @param context The fc2Context to be used.
 * @param triggerMode Structure providing trigger mode settings.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetTriggerModeBroadcast( 
    fc2Context context,
    fc2TriggerMode triggerMode );

/**
 * 
 *
 * @param context The fc2Context to be used.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2FireSoftwareTrigger(
    fc2Context context );

/**
 * Fire the software trigger according to the DCAM specifications.
 *
 * @param context The fc2Context to be used.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2FireSoftwareTriggerBroadcast(
    fc2Context context );

/**
 * Retrieve trigger delay information from the camera.
 *
 * @param context The fc2Context to be used.
 * @param triggerDelayInfo Structure to receive trigger delay information.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetTriggerDelayInfo( 
    fc2Context context,
    @Cast("fc2TriggerDelayInfo*") fc2PropertyInfo triggerDelayInfo );

/**
 * Retrieve current trigger delay settings from the camera.
 *
 * @param context The fc2Context to be used.
 * @param triggerDelay Structure to receive trigger delay settings.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetTriggerDelay( 
    fc2Context context,
    @Cast("fc2TriggerDelay*") fc2Property triggerDelay );

/**
 * Set the specified trigger delay settings to the camera.
 *
 * @param context The fc2Context to be used.
 * @param triggerDelay Structure providing trigger delay settings.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetTriggerDelay( 
    fc2Context context,
    @Cast("fc2TriggerDelay*") fc2Property triggerDelay );

/**
 * Set the specified trigger delay settings to the camera.
 *
 * @param context The fc2Context to be used.
 * @param triggerDelay Structure providing trigger delay settings.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetTriggerDelayBroadcast( 
    fc2Context context,
    @Cast("fc2TriggerDelay*") fc2Property triggerDelay );

/**
 * Retrieve strobe information from the camera.
 *
 * @param context The fc2Context to be used.
 * @param strobeInfo Structure to receive strobe information.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetStrobeInfo( 
    fc2Context context,
    fc2StrobeInfo strobeInfo );

/**
 * Retrieve current strobe settings from the camera. The strobe pin
 * must be specified in the structure before being passed in to
 * the function.
 *
 * @param context The fc2Context to be used.
 * @param strobeControl Structure to receive strobe settings.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetStrobe( 
    fc2Context context,
    fc2StrobeControl strobeControl );

/**
 * Set current strobe settings to the camera. The strobe pin
 * must be specified in the structure before being passed in to
 * the function.
 *
 * @param context The fc2Context to be used.
 * @param strobeControl Structure providing strobe settings.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetStrobe( 
    fc2Context context,
    fc2StrobeControl strobeControl );

/**
 * Set current strobe settings to the camera. The strobe pin
 * must be specified in the structure before being passed in to
 * the function.
 *
 * @param context The fc2Context to be used.
 * @param strobeControl Structure providing strobe settings.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetStrobeBroadcast( 
    fc2Context context,
    fc2StrobeControl strobeControl );

/**
 * Query the camera to determine if the specified video mode and 
 * frame rate is supported.
 *
 * @param context The fc2Context to be used.
 * @param videoMode Video mode to check.
 * @param frameRate Frame rate to check.
 * @param pSupported Whether the video mode and frame rate is supported.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetVideoModeAndFrameRateInfo(
    fc2Context context,
    @Cast("fc2VideoMode") int videoMode,
    @Cast("fc2FrameRate") int frameRate,
    @Cast("BOOL*") IntPointer pSupported);
public static native @Cast("fc2Error") int fc2GetVideoModeAndFrameRateInfo(
    fc2Context context,
    @Cast("fc2VideoMode") int videoMode,
    @Cast("fc2FrameRate") int frameRate,
    @Cast("BOOL*") IntBuffer pSupported);
public static native @Cast("fc2Error") int fc2GetVideoModeAndFrameRateInfo(
    fc2Context context,
    @Cast("fc2VideoMode") int videoMode,
    @Cast("fc2FrameRate") int frameRate,
    @Cast("BOOL*") int[] pSupported);

/**
 * Get the current video mode and frame rate from the camera. If
 * the camera is in Format7, the video mode will be VIDEOMODE_FORMAT7
 * and the frame rate will be FRAMERATE_FORMAT7.
 *
 * @param context The fc2Context to be used.
 * @param videoMode Current video mode.
 * @param frameRate Current frame rate.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetVideoModeAndFrameRate(
    fc2Context context,
    @Cast("fc2VideoMode*") IntPointer videoMode,
    @Cast("fc2FrameRate*") IntPointer frameRate );
public static native @Cast("fc2Error") int fc2GetVideoModeAndFrameRate(
    fc2Context context,
    @Cast("fc2VideoMode*") IntBuffer videoMode,
    @Cast("fc2FrameRate*") IntBuffer frameRate );
public static native @Cast("fc2Error") int fc2GetVideoModeAndFrameRate(
    fc2Context context,
    @Cast("fc2VideoMode*") int[] videoMode,
    @Cast("fc2FrameRate*") int[] frameRate );

/**
 * Set the specified video mode and frame rate to the camera. It is
 * not possible to set the camera to VIDEOMODE_FORMAT7 or 
 * FRAMERATE_FORMAT7. Use the Format7 functions to set the camera
 * into Format7.
 *
 * @param context The fc2Context to be used.
 * @param videoMode Video mode to set to camera.
 * @param frameRate Frame rate to set to camera.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetVideoModeAndFrameRate(
    fc2Context context,
    @Cast("fc2VideoMode") int videoMode,
    @Cast("fc2FrameRate") int frameRate );

/**
 * Retrieve the availability of Format7 custom image mode and the
 * camera capabilities for the specified Format7 mode. The mode must
 * be specified in the Format7Info structure in order for the
 * function to succeed.
 *
 * @param context The fc2Context to be used.
 * @param info Structure to be filled with the capabilities of the specified
 *             mode and the current state in the specified mode.
 * @param pSupported Whether the specified mode is supported.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetFormat7Info( 
    fc2Context context,
    fc2Format7Info info,
    @Cast("BOOL*") IntPointer pSupported );
public static native @Cast("fc2Error") int fc2GetFormat7Info( 
    fc2Context context,
    fc2Format7Info info,
    @Cast("BOOL*") IntBuffer pSupported );
public static native @Cast("fc2Error") int fc2GetFormat7Info( 
    fc2Context context,
    fc2Format7Info info,
    @Cast("BOOL*") int[] pSupported );

/**
 * Validates Format7ImageSettings structure and returns valid packet
 * size information if the image settings are valid. The current
 * image settings are cached while validation is taking place. The
 * cached settings are restored when validation is complete.
 *
 * @param context The fc2Context to be used.
 * @param imageSettings Structure containing the image settings.
 * @param settingsAreValid Whether the settings are valid.
 * @param packetInfo Packet size information that can be used to
 *                   determine a valid packet size.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2ValidateFormat7Settings( 
    fc2Context context,
    fc2Format7ImageSettings imageSettings,
    @Cast("BOOL*") IntPointer settingsAreValid,
    fc2Format7PacketInfo packetInfo );
public static native @Cast("fc2Error") int fc2ValidateFormat7Settings( 
    fc2Context context,
    fc2Format7ImageSettings imageSettings,
    @Cast("BOOL*") IntBuffer settingsAreValid,
    fc2Format7PacketInfo packetInfo );
public static native @Cast("fc2Error") int fc2ValidateFormat7Settings( 
    fc2Context context,
    fc2Format7ImageSettings imageSettings,
    @Cast("BOOL*") int[] settingsAreValid,
    fc2Format7PacketInfo packetInfo );

/**
 * Get the current Format7 configuration from the camera. This call
 * will only succeed if the camera is already in Format7.
 *
 * @param context The fc2Context to be used.
 * @param imageSettings Current image settings.
 * @param packetSize Current packet size.
 * @param percentage Current packet size as a percentage.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetFormat7Configuration( 
    fc2Context context,
    fc2Format7ImageSettings imageSettings,
    @Cast("unsigned int*") IntPointer packetSize,
    FloatPointer percentage );
public static native @Cast("fc2Error") int fc2GetFormat7Configuration( 
    fc2Context context,
    fc2Format7ImageSettings imageSettings,
    @Cast("unsigned int*") IntBuffer packetSize,
    FloatBuffer percentage );
public static native @Cast("fc2Error") int fc2GetFormat7Configuration( 
    fc2Context context,
    fc2Format7ImageSettings imageSettings,
    @Cast("unsigned int*") int[] packetSize,
    float[] percentage );

/**
 * Set the current Format7 configuration to the camera.
 *
 * @param context The fc2Context to be used.
 * @param imageSettings Image settings to be written to the camera.
 * @param packetSize Packet size to be written to the camera.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetFormat7ConfigurationPacket( 
    fc2Context context,
    fc2Format7ImageSettings imageSettings,
    @Cast("unsigned int") int packetSize );

/**
* Set the current Format7 configuration to the camera.
*
* @param context The fc2Context to be used.
* @param imageSettings Image settings to be written to the camera.
* @param percentSpeed Packet size as a percentage to be written to the camera.
*
* @return A fc2Error indicating the success or failure of the function.
*/
public static native @Cast("fc2Error") int fc2SetFormat7Configuration(
    fc2Context context,
    fc2Format7ImageSettings imageSettings,
    float percentSpeed );

/**
 * Write a GVCP register.
 *
 * @param context The fc2Context to be used.
 * @param address GVCP address to be written to.
 * @param value The value to be written.
 *
 * @return An Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2WriteGVCPRegister( 
    fc2Context context,
    @Cast("unsigned int") int address, 
    @Cast("unsigned int") int value);     

/**
 * Write a GVCP register with broadcast
 *
 * @param context The fc2Context to be used.
 * @param address GVCP address to be written to.
 * @param value The value to be written.
 *
 * @return An Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2WriteGVCPRegisterBroadcast( 
    fc2Context context,
    @Cast("unsigned int") int address, 
    @Cast("unsigned int") int value);     

/**
 * Read a GVCP register.
 *
 * @param context The fc2Context to be used.
 * @param address GVCP address to be read from.
 * @param pValue The value that is read.
 *
 * @return An Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2ReadGVCPRegister( 
    fc2Context context,
    @Cast("unsigned int") int address, 
    @Cast("unsigned int*") IntPointer pValue );
public static native @Cast("fc2Error") int fc2ReadGVCPRegister( 
    fc2Context context,
    @Cast("unsigned int") int address, 
    @Cast("unsigned int*") IntBuffer pValue );
public static native @Cast("fc2Error") int fc2ReadGVCPRegister( 
    fc2Context context,
    @Cast("unsigned int") int address, 
    @Cast("unsigned int*") int[] pValue );       

/**
 * Write a GVCP register block.
 *
 * @param context The fc2Context to be used.
 * @param address GVCP address to be write to.
 * @param pBuffer Array containing data to be written.
 * @param length Size of array, in quadlets.
 *
 * @return An Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2WriteGVCPRegisterBlock(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("const unsigned int*") IntPointer pBuffer,
    @Cast("unsigned int") int length );
public static native @Cast("fc2Error") int fc2WriteGVCPRegisterBlock(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("const unsigned int*") IntBuffer pBuffer,
    @Cast("unsigned int") int length );
public static native @Cast("fc2Error") int fc2WriteGVCPRegisterBlock(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("const unsigned int*") int[] pBuffer,
    @Cast("unsigned int") int length );    

/**
 * Read a GVCP register block.
 *
 * @param context The fc2Context to be used.
 * @param address GVCP address to be read from.
 * @param pBuffer Array containing data to be written.
 * @param length Size of array, in quadlets.
 *
 * @return An Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2ReadGVCPRegisterBlock(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("unsigned int*") IntPointer pBuffer,
    @Cast("unsigned int") int length );
public static native @Cast("fc2Error") int fc2ReadGVCPRegisterBlock(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("unsigned int*") IntBuffer pBuffer,
    @Cast("unsigned int") int length );
public static native @Cast("fc2Error") int fc2ReadGVCPRegisterBlock(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("unsigned int*") int[] pBuffer,
    @Cast("unsigned int") int length ); 

/**
 * Write a GVCP memory block.
 *
 * @param context The fc2Context to be used.
 * @param address GVCP address to be write to.
 * @param pBuffer Array containing data to be written.
 * @param length Size of array, in quadlets.
 *
 * @return An Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2WriteGVCPMemory(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("const unsigned char*") BytePointer pBuffer,
    @Cast("unsigned int") int length );
public static native @Cast("fc2Error") int fc2WriteGVCPMemory(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("const unsigned char*") ByteBuffer pBuffer,
    @Cast("unsigned int") int length );
public static native @Cast("fc2Error") int fc2WriteGVCPMemory(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("const unsigned char*") byte[] pBuffer,
    @Cast("unsigned int") int length );    

/**
 * Read a GVCP memory block.
 *
 * @param context The fc2Context to be used.
 * @param address GVCP address to be read from.
 * @param pBuffer Array containing data to be written.
 * @param length Size of array, in quadlets.
 *
 * @return An Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2ReadGVCPMemory(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("unsigned char*") BytePointer pBuffer,
    @Cast("unsigned int") int length );
public static native @Cast("fc2Error") int fc2ReadGVCPMemory(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("unsigned char*") ByteBuffer pBuffer,
    @Cast("unsigned int") int length );
public static native @Cast("fc2Error") int fc2ReadGVCPMemory(
    fc2Context context,
    @Cast("unsigned int") int address,
    @Cast("unsigned char*") byte[] pBuffer,
    @Cast("unsigned int") int length ); 

/**
 * Get the specified GigEProperty. The GigEPropertyType field must
 * be set in order for this function to succeed.
 *
 * @param context The fc2Context to be used.
 * @param pGigEProp The GigE property to get.
 *
 * @return An Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetGigEProperty( 
        fc2Context context,
        fc2GigEProperty pGigEProp );

/**
 * Set the specified GigEProperty. The GigEPropertyType field must
 * be set in order for this function to succeed.
 *
 * @param context The fc2Context to be used.
 * @param pGigEProp The GigE property to set.
 *
 * @return An Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetGigEProperty( 
    fc2Context context,
    @Const fc2GigEProperty pGigEProp );

public static native @Cast("fc2Error") int fc2QueryGigEImagingMode( 
    fc2Context context,
    @Cast("fc2Mode") int mode, 
    @Cast("BOOL*") IntPointer isSupported );
public static native @Cast("fc2Error") int fc2QueryGigEImagingMode( 
    fc2Context context,
    @Cast("fc2Mode") int mode, 
    @Cast("BOOL*") IntBuffer isSupported );
public static native @Cast("fc2Error") int fc2QueryGigEImagingMode( 
    fc2Context context,
    @Cast("fc2Mode") int mode, 
    @Cast("BOOL*") int[] isSupported );

public static native @Cast("fc2Error") int fc2GetGigEImagingMode(
    fc2Context context,
    @Cast("fc2Mode*") IntPointer mode );
public static native @Cast("fc2Error") int fc2GetGigEImagingMode(
    fc2Context context,
    @Cast("fc2Mode*") IntBuffer mode );
public static native @Cast("fc2Error") int fc2GetGigEImagingMode(
    fc2Context context,
    @Cast("fc2Mode*") int[] mode );

public static native @Cast("fc2Error") int fc2SetGigEImagingMode( 
    fc2Context context,
    @Cast("fc2Mode") int mode );        

public static native @Cast("fc2Error") int fc2GetGigEImageSettingsInfo( 
    fc2Context context,
    fc2GigEImageSettingsInfo pInfo );

public static native @Cast("fc2Error") int fc2GetGigEImageSettings(
    fc2Context context,
    fc2GigEImageSettings pImageSettings );

public static native @Cast("fc2Error") int fc2SetGigEImageSettings( 
    fc2Context context,
    @Const fc2GigEImageSettings pImageSettings );

public static native @Cast("fc2Error") int fc2GetGigEConfig(
    fc2Context context,
    fc2GigEConfig pConfig );

public static native @Cast("fc2Error") int fc2SetGigEConfig( 
    fc2Context context,
    @Const fc2GigEConfig pConfig );

public static native @Cast("fc2Error") int fc2GetGigEImageBinningSettings( 
    fc2Context context, 
    @Cast("unsigned int*") IntPointer horzBinnningValue, 
    @Cast("unsigned int*") IntPointer vertBinnningValue );
public static native @Cast("fc2Error") int fc2GetGigEImageBinningSettings( 
    fc2Context context, 
    @Cast("unsigned int*") IntBuffer horzBinnningValue, 
    @Cast("unsigned int*") IntBuffer vertBinnningValue );
public static native @Cast("fc2Error") int fc2GetGigEImageBinningSettings( 
    fc2Context context, 
    @Cast("unsigned int*") int[] horzBinnningValue, 
    @Cast("unsigned int*") int[] vertBinnningValue );

public static native @Cast("fc2Error") int fc2SetGigEImageBinningSettings( 
    fc2Context context, 
    @Cast("unsigned int") int horzBinnningValue, 
    @Cast("unsigned int") int vertBinnningValue );

public static native @Cast("fc2Error") int fc2GetNumStreamChannels(
    fc2Context context,
    @Cast("unsigned int*") IntPointer numChannels );
public static native @Cast("fc2Error") int fc2GetNumStreamChannels(
    fc2Context context,
    @Cast("unsigned int*") IntBuffer numChannels );
public static native @Cast("fc2Error") int fc2GetNumStreamChannels(
    fc2Context context,
    @Cast("unsigned int*") int[] numChannels );

public static native @Cast("fc2Error") int fc2GetGigEStreamChannelInfo(
    fc2Context context,
    @Cast("unsigned int") int channel, 
    fc2GigEStreamChannel pChannel );

public static native @Cast("fc2Error") int fc2SetGigEStreamChannelInfo( 
    fc2Context context,
    @Cast("unsigned int") int channel, 
    fc2GigEStreamChannel pChannel );

/**
 * Query if LUT support is available on the camera. Note that some cameras 
 * may report support for the LUT and return an inputBitDepth of 0. In these 
 * cases use log2(numEntries) for the inputBitDepth.
 *
 * @param context The fc2Context to be used.
 * @param pData The LUT structure to be filled.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetLUTInfo(
    fc2Context context,
    fc2LUTData pData );

/**
 * Query the read/write status of a single LUT bank.
 *
 * @param context The fc2Context to be used.
 * @param bank The bank to query.
 * @param pReadSupported Whether reading from the bank is supported.
 * @param pWriteSupported Whether writing to the bank is supported.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetLUTBankInfo(
    fc2Context context,
    @Cast("unsigned int") int bank,
    @Cast("BOOL*") IntPointer pReadSupported,
    @Cast("BOOL*") IntPointer pWriteSupported );
public static native @Cast("fc2Error") int fc2GetLUTBankInfo(
    fc2Context context,
    @Cast("unsigned int") int bank,
    @Cast("BOOL*") IntBuffer pReadSupported,
    @Cast("BOOL*") IntBuffer pWriteSupported );
public static native @Cast("fc2Error") int fc2GetLUTBankInfo(
    fc2Context context,
    @Cast("unsigned int") int bank,
    @Cast("BOOL*") int[] pReadSupported,
    @Cast("BOOL*") int[] pWriteSupported );

/**
 * Get the LUT bank that is currently being used. For cameras with
 * PGR LUT, the active bank is always 0.
 *
 * @param context The fc2Context to be used.
 * @param pActiveBank The currently active bank.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetActiveLUTBank(
    fc2Context context,
    @Cast("unsigned int*") IntPointer pActiveBank );
public static native @Cast("fc2Error") int fc2GetActiveLUTBank(
    fc2Context context,
    @Cast("unsigned int*") IntBuffer pActiveBank );
public static native @Cast("fc2Error") int fc2GetActiveLUTBank(
    fc2Context context,
    @Cast("unsigned int*") int[] pActiveBank );

/**
 * Set the LUT bank that will be used.
 *
 * @param context The fc2Context to be used.
 * @param activeBank The bank to be set as active.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetActiveLUTBank(
    fc2Context context,
    @Cast("unsigned int") int activeBank );

/**
 * Enable or disable LUT functionality on the camera.
 *
 * @param context The fc2Context to be used.
 * @param on Whether to enable or disable LUT.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2EnableLUT(
    fc2Context context,
    @Cast("BOOL") int on );

/**
 * Get the LUT channel settings from the camera.
 *
 * @param context The fc2Context to be used.
 * @param bank Bank to retrieve.
 * @param channel Channel to retrieve.
 * @param sizeEntries Number of entries in LUT table to read.
 * @param pEntries Array to store LUT entries.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetLUTChannel(
    fc2Context context,
    @Cast("unsigned int") int bank, 
    @Cast("unsigned int") int channel,
    @Cast("unsigned int") int sizeEntries,
    @Cast("unsigned int*") IntPointer pEntries );
public static native @Cast("fc2Error") int fc2GetLUTChannel(
    fc2Context context,
    @Cast("unsigned int") int bank, 
    @Cast("unsigned int") int channel,
    @Cast("unsigned int") int sizeEntries,
    @Cast("unsigned int*") IntBuffer pEntries );
public static native @Cast("fc2Error") int fc2GetLUTChannel(
    fc2Context context,
    @Cast("unsigned int") int bank, 
    @Cast("unsigned int") int channel,
    @Cast("unsigned int") int sizeEntries,
    @Cast("unsigned int*") int[] pEntries );

/**
 * Set the LUT channel settings to the camera.
 *
 * @param context The fc2Context to be used.
 * @param bank Bank to set.
 * @param channel Channel to set.
 * @param sizeEntries Number of entries in LUT table to write. This must be the 
 *					  same size as numEntries returned by GetLutInfo().
 * @param pEntries Array containing LUT entries to write.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetLUTChannel(
    fc2Context context,
    @Cast("unsigned int") int bank, 
    @Cast("unsigned int") int channel,
    @Cast("unsigned int") int sizeEntries,
    @Cast("unsigned int*") IntPointer pEntries );
public static native @Cast("fc2Error") int fc2SetLUTChannel(
    fc2Context context,
    @Cast("unsigned int") int bank, 
    @Cast("unsigned int") int channel,
    @Cast("unsigned int") int sizeEntries,
    @Cast("unsigned int*") IntBuffer pEntries );
public static native @Cast("fc2Error") int fc2SetLUTChannel(
    fc2Context context,
    @Cast("unsigned int") int bank, 
    @Cast("unsigned int") int channel,
    @Cast("unsigned int") int sizeEntries,
    @Cast("unsigned int*") int[] pEntries );

/**
 * Retrieve the current memory channel from the camera.
 *
 * @param context The fc2Context to be used.
 * @param pCurrentChannel Current memory channel.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetMemoryChannel( 
    fc2Context context,
    @Cast("unsigned int*") IntPointer pCurrentChannel );
public static native @Cast("fc2Error") int fc2GetMemoryChannel( 
    fc2Context context,
    @Cast("unsigned int*") IntBuffer pCurrentChannel );
public static native @Cast("fc2Error") int fc2GetMemoryChannel( 
    fc2Context context,
    @Cast("unsigned int*") int[] pCurrentChannel );

/**
 * Save the current settings to the specfied current memory channel.
 *
 * @param context The fc2Context to be used.
 * @param channel Memory channel to save to.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SaveToMemoryChannel( 
    fc2Context context,
    @Cast("unsigned int") int channel );

/**
 * Restore the specfied current memory channel.
 *
 * @param context The fc2Context to be used.
 * @param channel Memory channel to restore from.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2RestoreFromMemoryChannel( 
    fc2Context context,
    @Cast("unsigned int") int channel );

/**
 * Query the camera for memory channel support. If the number of 
 * channels is 0, then memory channel support is not available.
 *
 * @param context The fc2Context to be used.
 * @param pNumChannels Number of memory channels supported.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetMemoryChannelInfo( 
    fc2Context context,
    @Cast("unsigned int*") IntPointer pNumChannels );
public static native @Cast("fc2Error") int fc2GetMemoryChannelInfo( 
    fc2Context context,
    @Cast("unsigned int*") IntBuffer pNumChannels );
public static native @Cast("fc2Error") int fc2GetMemoryChannelInfo( 
    fc2Context context,
    @Cast("unsigned int*") int[] pNumChannels );

/**
 * Get the current status of the embedded image information register,
 * as well as the availability of each embedded property.
 *
 * @param context The fc2Context to be used.
 * @param pInfo Structure to be filled.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetEmbeddedImageInfo( 
    fc2Context context,
    fc2EmbeddedImageInfo pInfo );

/**
 * Sets the on/off values of the embedded image information structure
 * to the camera.
 *
 * @param context The fc2Context to be used.
 * @param pInfo Structure to be used.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetEmbeddedImageInfo( 
    fc2Context context,
    fc2EmbeddedImageInfo pInfo );

/**
 * Returns a text representation of the register value.
 *
 * @param registerVal The register value to query.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("const char*") BytePointer fc2GetRegisterString( 
    @Cast("unsigned int") int registerVal);

/**
 * Create a fc2Image. If externally allocated memory is to be used for the 
 * converted image, simply assigning the pData member of the fc2Image structure 
 * is insufficient. fc2SetImageData() should be called in order to populate
 * the fc2Image structure correctly.
 *
 * @param pImage Pointer to image to be created.
 *
 * @see fc2SetImageData()
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2CreateImage( 
    fc2Image pImage );

/**
 * Destroy the fc2Image.
 *
 * @param image Pointer to image to be destroyed.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2DestroyImage( 
    fc2Image image );

/**
 * Set the default color processing algorithm.  This method will be 
 * used for any image with the DEFAULT algorithm set. The method used 
 * is determined at the time of the Convert() call, therefore the most 
 * recent execution of this function will take precedence. The default 
 * setting is shared within the current process.
 *
 * @param defaultMethod The color processing algorithm to set.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetDefaultColorProcessing( 
    @Cast("fc2ColorProcessingAlgorithm") int defaultMethod );

/**
 * Get the default color processing algorithm.
 *
 * @param pDefaultMethod The default color processing algorithm.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetDefaultColorProcessing( 
    @Cast("fc2ColorProcessingAlgorithm*") IntPointer pDefaultMethod );
public static native @Cast("fc2Error") int fc2GetDefaultColorProcessing( 
    @Cast("fc2ColorProcessingAlgorithm*") IntBuffer pDefaultMethod );
public static native @Cast("fc2Error") int fc2GetDefaultColorProcessing( 
    @Cast("fc2ColorProcessingAlgorithm*") int[] pDefaultMethod );

/**
 * Set the default output pixel format. This format will be used for any 
 * call to Convert() that does not specify an output format. The format 
 * used will be determined at the time of the Convert() call, therefore 
 * the most recent execution of this function will take precedence. 
 * The default is shared within the current process.
 *
 * @param format The output pixel format to set.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetDefaultOutputFormat( 
    @Cast("fc2PixelFormat") int format );

/**
 * Get the default output pixel format.
 *
 * @param pFormat The default pixel format.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetDefaultOutputFormat( 
    @Cast("fc2PixelFormat*") IntPointer pFormat );
public static native @Cast("fc2Error") int fc2GetDefaultOutputFormat( 
    @Cast("fc2PixelFormat*") IntBuffer pFormat );
public static native @Cast("fc2Error") int fc2GetDefaultOutputFormat( 
    @Cast("fc2PixelFormat*") int[] pFormat );

/**
 * Calculate the bits per pixel for the specified pixel format.
 *
 * @param format The pixel format.
 * @param pBitsPerPixel The bits per pixel.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2DetermineBitsPerPixel( 
    @Cast("fc2PixelFormat") int format,
    @Cast("unsigned int*") IntPointer pBitsPerPixel );
public static native @Cast("fc2Error") int fc2DetermineBitsPerPixel( 
    @Cast("fc2PixelFormat") int format,
    @Cast("unsigned int*") IntBuffer pBitsPerPixel );
public static native @Cast("fc2Error") int fc2DetermineBitsPerPixel( 
    @Cast("fc2PixelFormat") int format,
    @Cast("unsigned int*") int[] pBitsPerPixel );

/**
 * Save the image to the specified file name with the file format
 * specified.
 *
 * @param pImage The fc2Image to be used.
 * @param pFilename Filename to save image with.
 * @param format File format to save in.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SaveImage(
    fc2Image pImage,
    @Cast("const char*") BytePointer pFilename,
    @Cast("fc2ImageFileFormat") int format );
public static native @Cast("fc2Error") int fc2SaveImage(
    fc2Image pImage,
    String pFilename,
    @Cast("fc2ImageFileFormat") int format );

/**
 * Save the image to the specified file name with the file format
 * specified.
 *
 * @param pImage The fc2Image to be used.
 * @param pFilename Filename to save image with.
 * @param format File format to save in.
 * @param pOption Options for saving image.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SaveImageWithOption( 
    fc2Image pImage, 
    @Cast("const char*") BytePointer pFilename, 
    @Cast("fc2ImageFileFormat") int format, 
    Pointer pOption );
public static native @Cast("fc2Error") int fc2SaveImageWithOption( 
    fc2Image pImage, 
    String pFilename, 
    @Cast("fc2ImageFileFormat") int format, 
    Pointer pOption );

/**
 * 
 *
 * @param pImageIn
 * @param pImageOut
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2ConvertImage(
    fc2Image pImageIn,
    fc2Image pImageOut );

/**
 * Converts the current image buffer to the specified output format and
 * stores the result in the specified image. The destination image 
 * does not need to be configured in any way before the call is made.
 *
 * @param format Output format of the converted image.
 * @param pImageIn Input image.
 * @param pImageOut Output image.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2ConvertImageTo(
    @Cast("fc2PixelFormat") int format,
    fc2Image pImageIn,
    fc2Image pImageOut );

/**
 * Get a pointer to the data associated with the image. This function
 * is considered unsafe. The pointer returned could be invalidated if
 * the buffer is resized or released. The pointer may also be
 * invalidated if the Image object is passed to fc2RetrieveBuffer().
 *
 * @param pImage The fc2Image to be used.
 * @param ppData A pointer to the image data.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetImageData(
    fc2Image pImage,
    @Cast("unsigned char**") PointerPointer ppData);
public static native @Cast("fc2Error") int fc2GetImageData(
    fc2Image pImage,
    @Cast("unsigned char**") @ByPtrPtr BytePointer ppData);
public static native @Cast("fc2Error") int fc2GetImageData(
    fc2Image pImage,
    @Cast("unsigned char**") @ByPtrPtr ByteBuffer ppData);
public static native @Cast("fc2Error") int fc2GetImageData(
    fc2Image pImage,
    @Cast("unsigned char**") @ByPtrPtr byte[] ppData);

/**
 * Set the data of the Image object.
 * Ownership of the image buffer is not transferred to the Image object.
 * It is the user's responsibility to delete the buffer when it is
 * no longer in use.
 *
 * @param pImage The fc2Image to be used.
 * @param pData Pointer to the image buffer.
 * @param dataSize Size of the image buffer.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetImageData( 
    fc2Image pImage,
    @Cast("const unsigned char*") BytePointer pData,
    @Cast("unsigned int") int dataSize);
public static native @Cast("fc2Error") int fc2SetImageData( 
    fc2Image pImage,
    @Cast("const unsigned char*") ByteBuffer pData,
    @Cast("unsigned int") int dataSize);
public static native @Cast("fc2Error") int fc2SetImageData( 
    fc2Image pImage,
    @Cast("const unsigned char*") byte[] pData,
    @Cast("unsigned int") int dataSize);

/**
 * Sets the dimensions of the image object.
 *
 * @param pImage The fc2Image to be used.
 * @param rows Number of rows to set.
 * @param cols Number of cols to set.
 * @param stride Stride to set.
 * @param pixelFormat Pixel format to set.
 * @param bayerFormat Bayer tile format to set.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetImageDimensions( 
    fc2Image pImage,
    @Cast("unsigned int") int rows,
    @Cast("unsigned int") int cols,
    @Cast("unsigned int") int stride,
    @Cast("fc2PixelFormat") int pixelFormat,
    @Cast("fc2BayerTileFormat") int bayerFormat);

/**
 * Get the timestamp data associated with the image. 
 *
 * @param pImage The fc2Image to be used.
 *
 * @return Timestamp data associated with the image.
 */
public static native @ByVal fc2TimeStamp fc2GetImageTimeStamp( 
    fc2Image pImage);

/**
 * Calculate statistics associated with the image. In order to collect
 * statistics for a particular channel, the enabled flag for the
 * channel must be set to true. Statistics can only be collected for
 * images in Mono8, Mono16, RGB, RGBU, BGR and BGRU.
 *
 * @param pImage The fc2Image to be used.
 * @param pImageStatisticsContext The fc2ImageStatisticsContext to hold the 
 *                                statistics.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2CalculateImageStatistics(
    fc2Image pImage,
    fc2ImageStatisticsContext pImageStatisticsContext );

/**
 * Create a statistics context.
 *
 * @param pImageStatisticsContext A statistics context.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2CreateImageStatistics(
    fc2ImageStatisticsContext pImageStatisticsContext );

/**
 * Destroy a statistics context.
 *
 * @param imageStatisticsContext A statistics context.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2DestroyImageStatistics(
    @ByVal fc2ImageStatisticsContext imageStatisticsContext );

/**
 * Get the status of a statistics channel.
 *
 * @param imageStatisticsContext A statistics context.
 * @param channel The statistics channel.
 * @param pEnabled Whether the channel is enabled.
 *
 * @see SetChannelStatus()
 *
 * @return An Error indicating the success or failure of the function.
 */
public static native @Cast("const fc2Error") int fc2GetChannelStatus(
	@ByVal fc2ImageStatisticsContext imageStatisticsContext,
	@Cast("fc2StatisticsChannel") int channel,
	@Cast("BOOL*") IntPointer pEnabled );
public static native @Cast("const fc2Error") int fc2GetChannelStatus(
	@ByVal fc2ImageStatisticsContext imageStatisticsContext,
	@Cast("fc2StatisticsChannel") int channel,
	@Cast("BOOL*") IntBuffer pEnabled );
public static native @Cast("const fc2Error") int fc2GetChannelStatus(
	@ByVal fc2ImageStatisticsContext imageStatisticsContext,
	@Cast("fc2StatisticsChannel") int channel,
	@Cast("BOOL*") int[] pEnabled );

/**
 * Set the status of a statistics channel.
 *
 * @param imageStatisticsContext A statistics context.
 * @param channel The statistics channel.
 * @param enabled Whether the channel should be enabled.
 *
 * @see GetChannelStatus()
 *
 * @return An Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2SetChannelStatus( 
	@ByVal fc2ImageStatisticsContext imageStatisticsContext,
	@Cast("fc2StatisticsChannel") int channel,
	@Cast("BOOL") int enabled );

/**
 * Get all statistics for the image.
 *
 * @param imageStatisticsContext The statistics context.
 * @param channel The statistics channel.
 * @param pRangeMin The minimum possible value.
 * @param pRangeMax The maximum possible value.
 * @param pPixelValueMin The minimum pixel value.
 * @param pPixelValueMax The maximum pixel value.
 * @param pNumPixelValues The number of unique pixel values.
 * @param pPixelValueMean The mean of the image.
 * @param ppHistogram Pointer to an array containing the histogram.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetImageStatistics(
    @ByVal fc2ImageStatisticsContext imageStatisticsContext,
    @Cast("fc2StatisticsChannel") int channel,
    @Cast("unsigned int*") IntPointer pRangeMin,
    @Cast("unsigned int*") IntPointer pRangeMax,
    @Cast("unsigned int*") IntPointer pPixelValueMin,
    @Cast("unsigned int*") IntPointer pPixelValueMax,
    @Cast("unsigned int*") IntPointer pNumPixelValues,
    FloatPointer pPixelValueMean,
    @Cast("int**") PointerPointer ppHistogram );
public static native @Cast("fc2Error") int fc2GetImageStatistics(
    @ByVal fc2ImageStatisticsContext imageStatisticsContext,
    @Cast("fc2StatisticsChannel") int channel,
    @Cast("unsigned int*") IntPointer pRangeMin,
    @Cast("unsigned int*") IntPointer pRangeMax,
    @Cast("unsigned int*") IntPointer pPixelValueMin,
    @Cast("unsigned int*") IntPointer pPixelValueMax,
    @Cast("unsigned int*") IntPointer pNumPixelValues,
    FloatPointer pPixelValueMean,
    @ByPtrPtr IntPointer ppHistogram );
public static native @Cast("fc2Error") int fc2GetImageStatistics(
    @ByVal fc2ImageStatisticsContext imageStatisticsContext,
    @Cast("fc2StatisticsChannel") int channel,
    @Cast("unsigned int*") IntBuffer pRangeMin,
    @Cast("unsigned int*") IntBuffer pRangeMax,
    @Cast("unsigned int*") IntBuffer pPixelValueMin,
    @Cast("unsigned int*") IntBuffer pPixelValueMax,
    @Cast("unsigned int*") IntBuffer pNumPixelValues,
    FloatBuffer pPixelValueMean,
    @ByPtrPtr IntBuffer ppHistogram );
public static native @Cast("fc2Error") int fc2GetImageStatistics(
    @ByVal fc2ImageStatisticsContext imageStatisticsContext,
    @Cast("fc2StatisticsChannel") int channel,
    @Cast("unsigned int*") int[] pRangeMin,
    @Cast("unsigned int*") int[] pRangeMax,
    @Cast("unsigned int*") int[] pPixelValueMin,
    @Cast("unsigned int*") int[] pPixelValueMax,
    @Cast("unsigned int*") int[] pNumPixelValues,
    float[] pPixelValueMean,
    @ByPtrPtr int[] ppHistogram );

/**
 * Create a AVI context.
 *
 * @param pAVIContext A AVI context.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2CreateAVI(
    fc2AVIContext pAVIContext );

/**
 * Open an AVI file in preparation for writing Images to disk.
 * The size of AVI files is limited to 2GB. The filenames are
 * automatically generated using the filename specified.
 *
 * @param AVIContext The AVI context to use.
 * @param pFileName The filename of the AVI file.
 * @param pOption Options to apply to the AVI file.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2AVIOpen( 
    @ByVal fc2AVIContext AVIContext,
    @Cast("const char*") BytePointer pFileName, 
    fc2AVIOption pOption );
public static native @Cast("fc2Error") int fc2AVIOpen( 
    @ByVal fc2AVIContext AVIContext,
    String pFileName, 
    fc2AVIOption pOption );

/**
 * Open an MJPEG file in preparation for writing Images to disk.
 * The size of AVI files is limited to 2GB. The filenames are
 * automatically generated using the filename specified.
 *
 * @param AVIContext The AVI context to use.
 * @param pFileName The filename of the AVI file.
 * @param pOption Options to apply to the AVI file.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2MJPGOpen( 
    @ByVal fc2AVIContext AVIContext,
    @Cast("const char*") BytePointer pFileName, 
    fc2MJPGOption pOption );
public static native @Cast("fc2Error") int fc2MJPGOpen( 
    @ByVal fc2AVIContext AVIContext,
    String pFileName, 
    fc2MJPGOption pOption );

/**
 * Open an H.264 file in preparation for writing Images to disk.
 * The size of AVI files is limited to 2GB. The filenames are
 * automatically generated using the filename specified.
 *
 * @param AVIContext The AVI context to use.
 * @param pFileName The filename of the AVI file.
 * @param pOption Options to apply to the AVI file.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2H264Open( 
    @ByVal fc2AVIContext AVIContext,
    @Cast("const char*") BytePointer pFileName, 
    fc2H264Option pOption );
public static native @Cast("fc2Error") int fc2H264Open( 
    @ByVal fc2AVIContext AVIContext,
    String pFileName, 
    fc2H264Option pOption );

/**
 * Append an image to the AVI file.
 *
 * @param AVIContext The AVI context to use.
 * @param pImage The image to append.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2AVIAppend( 
    @ByVal fc2AVIContext AVIContext,
    fc2Image pImage );

/**
 * Close the AVI file.
 *
 * @param AVIContext The AVI context to use.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2AVIClose(
    @ByVal fc2AVIContext AVIContext );

/**
 * Destroy a AVI context.
 *
 * @param AVIContext A AVI context.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2DestroyAVI(
    @ByVal fc2AVIContext AVIContext );

/**
 * Get system information.
 *
 * @param pSystemInfo Structure to receive system information.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetSystemInfo( fc2SystemInfo pSystemInfo);

/**
 * Get library version.
 *
 * @param pVersion Structure to receive the library version.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetLibraryVersion( fc2Version pVersion);

/**
 * Launch a URL in the system default browser.
 *
 * @param pAddress URL to open in browser.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2LaunchBrowser( @Cast("const char*") BytePointer pAddress);
public static native @Cast("fc2Error") int fc2LaunchBrowser( String pAddress);

/**
 * Open a CHM file in the system default CHM viewer.
 *
 * @param pFileName Filename of CHM file to open.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2LaunchHelp( @Cast("const char*") BytePointer pFileName);
public static native @Cast("fc2Error") int fc2LaunchHelp( String pFileName);

/**
 * Execute a command in the terminal. This is a blocking call that 
 * will return when the command completes.
 *
 * @param pCommand Command to execute.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2LaunchCommand( @Cast("const char*") BytePointer pCommand);
public static native @Cast("fc2Error") int fc2LaunchCommand( String pCommand);

/**
 * Execute a command in the terminal. This is a non-blocking call that 
 * will return immediately. The return value of the command can be
 * retrieved in the callback.
 *
 * @param pCommand Command to execute.
 * @param pCallback Callback to fire when command is complete.
 * @param pUserData Data pointer to pass to callback.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2LaunchCommandAsync( 
    @Cast("const char*") BytePointer pCommand,
    fc2AsyncCommandCallback pCallback,
    Pointer pUserData );
public static native @Cast("fc2Error") int fc2LaunchCommandAsync( 
    String pCommand,
    fc2AsyncCommandCallback pCallback,
    Pointer pUserData );

/**
 * Get a string representation of an error.
 *
 * @param error Error to be parsed.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("const char*") BytePointer fc2ErrorToDescription( 
    @Cast("fc2Error") int error);

/**
 * Get cycle time from camera
 *
 * @param Timestamp struct.
 *
 * @return A fc2Error indicating the success or failure of the function.
 */
public static native @Cast("fc2Error") int fc2GetCycleTime( fc2Context context, fc2TimeStamp pTimeStamp );

// #ifdef __cplusplus
// #endif

// #endif // PGR_FC2_FLYCAPTURE2_C_H



}
