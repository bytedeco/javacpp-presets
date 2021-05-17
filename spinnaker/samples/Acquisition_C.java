//=============================================================================
// Copyright © 2018 FLIR Integrated Imaging Solutions, Inc. All Rights Reserved.
//
// This software is the confidential and proprietary information of FLIR
// Integrated Imaging Solutions, Inc. ("Confidential Information"). You
// shall not disclose such Confidential Information and shall use it only in
// accordance with the terms of the license agreement you entered into
// with FLIR Integrated Imaging Solutions, Inc. (FLIR).
//
// FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
// SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
// SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
// THIS SOFTWARE OR ITS DERIVATIVES.
//=============================================================================*/

import java.io.File;

import org.bytedeco.javacpp.*;
import org.bytedeco.spinnaker.Spinnaker_C.*;
import static org.bytedeco.spinnaker.global.Spinnaker_C.*;

/**
 * Example how to enumerate cameras, start acquisition, and grab images.
 * <p>
 * Please see Enumeration_C example for more in-depth comments on preparing and cleaning up the system.
 */
public class Acquisition_C {
    private final static int MAX_BUFF_LEN = 256;

    private static String findErrorNameByValue(int value) {
        for (spinError v : spinError.values()) {
            if (v.value == value) {
                return v.name();
            }
        }
        return "???";
    }

    private static String findImageStatusNameByValue(int value) {
        for (spinImageStatus v : spinImageStatus.values()) {
            if (v.value == value) {
                return v.name();
            }
        }
        return "???";
    }


    /**
     * Check if 'err' is 'SPINNAKER_ERR_SUCCESS'.
     * If it is do nothing otherwise print error description and exit.
     *
     * @param err     error value.
     * @param message additional message to print.
     */

    private static void exitOnError(spinError err, String message) {
        if (printOnError(err, message)) {
            System.out.println("Aborting.");
            System.exit(err.value);
        }
    }

    /**
     * Check if 'err' is 'SPINNAKER_ERR_SUCCESS'.
     * If it is do nothing otherwise print error information.
     *
     * @param err     error value.
     * @param message additional message to print.
     * @return 'false' if err is not SPINNAKER_ERR_SUCCESS, or 'true' for any other 'err' value.
     */
    private static boolean printOnError(spinError err, String message) {
        if (err.value != spinError.SPINNAKER_ERR_SUCCESS.value) {
            System.out.println(message);
            System.out.println("Error " + err.value + " " + findErrorNameByValue(err.value) + "\n");
            return true;
        } else {
            return false;
        }
    }

    /**
     * This function helps to check if a node is available and readable
     */
    private static boolean isAvailableAndReadable(spinNodeHandle hNode, String nodeName) {
        BytePointer pbAvailable = new BytePointer(1);
        spinError err;
        err = spinNodeIsAvailable(hNode, pbAvailable);
        printOnError(err, "Unable to retrieve node availability (" + nodeName + " node)");

        BytePointer pbReadable = new BytePointer(1);
        err = spinNodeIsReadable(hNode, pbReadable);
        printOnError(err, "Unable to retrieve node readability (" + nodeName + " node)");
        return pbReadable.getBool() && pbAvailable.getBool();
    }

    /**
     * This function helps to check if a node is available and writable
     */
    private static boolean isAvailableAndWritable(spinNodeHandle hNode, String nodeName) {
        BytePointer pbAvailable = new BytePointer(1);
        spinError err;
        err = spinNodeIsAvailable(hNode, pbAvailable);
        printOnError(err, "Unable to retrieve node availability (" + nodeName + " node).");

        BytePointer pbWritable = new BytePointer(1);
        err = spinNodeIsWritable(hNode, pbWritable);
        printOnError(err, "Unable to retrieve node writability (" + nodeName + " node).");
        return pbWritable.getBool() && pbAvailable.getBool();
    }

    /**
     * This function handles the error prints when a node or entry is unavailable or
     * not readable/writable on the connected camera
     */
    private static void printRetrieveNodeFailure(String node, String name) {
        System.out.println("Unable to get " + node + " (" + name + " " + node + " retrieval failed).\n");
    }

    /**
     * This function prints the device information of the camera from the transport
     * layer; please see NodeMapInfo_C example for more in-depth comments on
     * printing device information from the nodemap.
     */
    private static spinError printDeviceInfo(spinNodeMapHandle hNodeMap) {
        spinError err;
        System.out.println("\n*** DEVICE INFORMATION ***\n\n");
        // Retrieve device information category node
        spinNodeHandle hDeviceInformation = new spinNodeHandle();
        err = spinNodeMapGetNode(hNodeMap, new BytePointer("DeviceInformation"), hDeviceInformation);
        printOnError(err, "Unable to retrieve node.");

        // Retrieve number of nodes within device information node
        SizeTPointer numFeatures = new SizeTPointer(1);
        if (isAvailableAndReadable(hDeviceInformation, "DeviceInformation")) {
            err = spinCategoryGetNumFeatures(hDeviceInformation, numFeatures);
            printOnError(err, "Unable to retrieve number of nodes.");
        } else {
            printRetrieveNodeFailure("node", "DeviceInformation");
            return spinError.SPINNAKER_ERR_ACCESS_DENIED;
        }

        // Iterate through nodes and print information
        for (int i = 0; i < numFeatures.get(); i++) {
            spinNodeHandle hFeatureNode = new spinNodeHandle();
            err = spinCategoryGetFeatureByIndex(hDeviceInformation, i, hFeatureNode);
            printOnError(err, "Unable to retrieve node.");

            // get feature node name
            BytePointer featureName = new BytePointer(MAX_BUFF_LEN);
            SizeTPointer lenFeatureName = new SizeTPointer(1);
            lenFeatureName.put(MAX_BUFF_LEN);
            err = spinNodeGetName(hFeatureNode, featureName, lenFeatureName);
            if (printOnError(err, "Error retrieving node name.")) {
                featureName.putString("Unknown name");
            }

            int[] featureType = {spinNodeType.UnknownNode.value};
            if (isAvailableAndReadable(hFeatureNode, featureName.getString())) {
                err = spinNodeGetType(hFeatureNode, featureType);
                if (printOnError(err, "Unable to retrieve node type.")) {
                    continue;
                }
            } else {
                System.out.println(featureName + ": Node not readable");
                continue;
            }
            BytePointer featureValue = new BytePointer(MAX_BUFF_LEN);
            SizeTPointer lenFeatureValue = new SizeTPointer(1);
            lenFeatureValue.put(MAX_BUFF_LEN);
            err = spinNodeToString(hFeatureNode, featureValue, lenFeatureValue);
            if (printOnError(err, "spinNodeToString")) {
                featureValue.putString("Unknown value");
            }
            System.out.println(featureName.getString().trim() + ": " + featureValue.getString().trim() + ".");
        }
        System.out.println();
        return err;
    }

    // This function acquires and saves 10 images from a device.
    private static spinError acquireImages(spinCamera hCam, spinNodeMapHandle hNodeMap, spinNodeMapHandle hNodeMapTLDevice) {
        System.out.println("\n*** IMAGE ACQUISITION ***\n");
        spinError err;
        //
        // Set acquisition mode to continuous
        //
        // *** NOTES ***
        // Because the example acquires and saves 10 images, setting acquisition
        // mode to continuous lets the example finish. If set to single frame
        // or multiframe (at a lower number of images), the example would just
        // hang. This would happen because the example has been written to acquire
        // 10 images while the camera would have been programmed to retrieve
        // less than that.
        //
        // Setting the value of an enumeration node is slightly more complicated
        // than other node types, and especially so in C. It can roughly be broken
        // down into four steps: first, the enumeration node is retrieved from the
        // nodemap; second, the entry node is retrieved from the enumeration node;
        // third, an integer is retrieved from the entry node; and finally, the
        // integer is set as the new value of the enumeration node.
        //
        // It is important to note that there are two sets of functions that might
        // produce erroneous results if they were to be mixed up. The first two
        // functions, spinEnumerationSetIntValue() and
        // spinEnumerationEntryGetIntValue(), use the integer values stored on each
        // individual cameras. The second two, spinEnumerationSetEnumValue() and
        // spinEnumerationEntryGetEnumValue(), use enum values defined in the
        // Spinnaker library. The int and enum values will most likely be
        // different from another.
        //

        // Retrieve enumeration node from nodemap
        spinNodeHandle hAcquisitionMode = new spinNodeHandle(); //NULL
        err = spinNodeMapGetNode(hNodeMap, new BytePointer("AcquisitionMode"), hAcquisitionMode);
        if (printOnError(err, "Unable to set acquisition mode to continuous (node retrieval).")) {
            return err;
        }

        // Retrieve entry node from enumeration node
        spinNodeHandle hAcquisitionModeContinuous = new spinNodeHandle(); // NULL
        if (isAvailableAndReadable(hAcquisitionMode, "AcquisitionMode")) {
            err = spinEnumerationGetEntryByName(hAcquisitionMode, new BytePointer("Continuous"), hAcquisitionModeContinuous);
            if (printOnError(err, "Unable to set acquisition mode to continuous (entry 'continuous' retrieval).")) {
                return err;
            }
        } else {
            printRetrieveNodeFailure("entry", "AcquisitionMode");
            return spinError.SPINNAKER_ERR_ACCESS_DENIED;
        }

        // Retrieve integer from entry node
        LongPointer acquisitionModeContinuous = new LongPointer(1);
        if (isAvailableAndReadable(hAcquisitionModeContinuous, "AcquisitionModeContinuous")) {
            err = spinEnumerationEntryGetIntValue(hAcquisitionModeContinuous, acquisitionModeContinuous);

            if (printOnError(err, "Unable to set acquisition mode to continuous (entry int value retrieval).")) {
                return err;
            }
        } else {
            printRetrieveNodeFailure("entry", "AcquisitionMode 'Continuous'");
            return spinError.SPINNAKER_ERR_ACCESS_DENIED;
        }

        // Set integer as new value of enumeration node
        if (isAvailableAndWritable(hAcquisitionMode, "AcquisitionMode")) {
            err = spinEnumerationSetIntValue(hAcquisitionMode, acquisitionModeContinuous.get());
            if (printOnError(err, "Unable to set acquisition mode to continuous (entry int value setting).")) {
                return err;
            }
        } else {
            printRetrieveNodeFailure("entry", "AcquisitionMode");
            return spinError.SPINNAKER_ERR_ACCESS_DENIED;
        }

        System.out.println("Acquisition mode set to continuous...");

        //
        // Begin acquiring images
        //
        // *** NOTES ***
        // What happens when the camera begins acquiring images depends on the
        // acquisition mode. Single frame captures only a single image, multi
        // frame catures a set number of images, and continuous captures a
        // continuous stream of images. Because the example calls for the retrieval
        // of 10 images, continuous mode has been set.
        //
        // *** LATER ***
        // Image acquisition must be ended when no more images are needed.
        //
        err = spinCameraBeginAcquisition(hCam);
        if (printOnError(err, "Unable to begin image acquisition.")) {
            return err;
        }

        System.out.println("Acquiring images...");

        //
        // Retrieve device serial number for filename
        //
        // *** NOTES ***
        // The device serial number is retrieved in order to keep cameras from
        // overwriting one another. Grabbing image IDs could also accomplish this.
        //
        spinNodeHandle hDeviceSerialNumber = new spinNodeHandle(); // NULL;
        BytePointer deviceSerialNumber = new BytePointer(MAX_BUFF_LEN);
        SizeTPointer lenDeviceSerialNumber = new SizeTPointer(1);
        lenDeviceSerialNumber.put(MAX_BUFF_LEN);
        err = spinNodeMapGetNode(hNodeMapTLDevice, new BytePointer("DeviceSerialNumber"), hDeviceSerialNumber);
        if (printOnError(err, "")) {
            deviceSerialNumber.putString("");
            lenDeviceSerialNumber.put(0);
        } else {
            if (isAvailableAndReadable(hDeviceSerialNumber, "DeviceSerialNumber")) {
                err = spinStringGetValue(hDeviceSerialNumber, deviceSerialNumber, lenDeviceSerialNumber);
                if (printOnError(err, "")) {
                    deviceSerialNumber.putString("");
                    lenDeviceSerialNumber.put(0);
                }
            } else {
                deviceSerialNumber.putString("");
                lenDeviceSerialNumber.put(0);
                printRetrieveNodeFailure("node", "DeviceSerialNumber");
            }
            System.out.println("Device serial number retrieved as " + deviceSerialNumber.getString().trim() + "...");
        }
        System.out.println();

        // Retrieve, convert, and save images
        final int k_numImages = 10;
        for (int imageCnt = 0; imageCnt < k_numImages; imageCnt++) {
            //
            // Retrieve next received image
            //
            // *** NOTES ***
            // Capturing an image houses images on the camera buffer. Trying to
            // capture an image that does not exist will hang the camera.
            //
            // *** LATER ***
            // Once an image from the buffer is saved and/or no longer needed, the
            // image must be released in orer to keep the buffer from filling up.
            //
            spinImage hResultImage = new spinImage(); //NULL;
            err = spinCameraGetNextImage(hCam, hResultImage);
            if (printOnError(err, "Unable to get next image. Non-fatal error.")) {
                continue;
            }
            //
            // Ensure image completion
            //
            // *** NOTES ***
            // Images can easily be checked for completion. This should be done
            // whenever a complete image is expected or required. Further, check
            // image status for a little more insight into why an image is
            // incomplete.
            //
            BytePointer isIncomplete = new BytePointer(1);
            boolean hasFailed = false;
            err = spinImageIsIncomplete(hResultImage, isIncomplete);
            if (printOnError(err, "Unable to determine image completion. Non-fatal error.")) {
                hasFailed = true;
            }
            // Check image for completion
            if (isIncomplete.getBool()) {
                IntPointer imageStatus = new IntPointer(1); //_spinImageStatus.IMAGE_NO_ERROR;
                err = spinImageGetStatus(hResultImage, imageStatus);
                if (!printOnError(err,
                        "Unable to retrieve image status. Non-fatal error. " + findImageStatusNameByValue(imageStatus.get()))) {
                    System.out.println(
                            "Image incomplete with image status " + findImageStatusNameByValue(imageStatus.get()) +
                                    "...");
                }
                hasFailed = true;
            }
            // Release incomplete or failed image
            if (hasFailed) {
                err = spinImageRelease(hResultImage);
                printOnError(err, "Unable to release image. Non-fatal error.");
                continue;
            }
            //
            // Print image information; height and width recorded in pixels
            //
            // *** NOTES ***
            // Images have quite a bit of available metadata including things such
            // as CRC, image status, and offset values, to name a few.
            //
            System.out.println("Grabbed image " + imageCnt);

            // Retrieve image width
            SizeTPointer width = new SizeTPointer(1);
            err = spinImageGetWidth(hResultImage, width);
            if (printOnError(err, "spinImageGetWidth()")) {
                System.out.println("width  = unknown");
            } else {
                System.out.println("width  = " + width.get());
            }

            // Retrieve image height
            SizeTPointer height = new SizeTPointer(1);
            err = spinImageGetHeight(hResultImage, height);
            if (printOnError(err, "spinImageGetHeight()")) {
                System.out.println("height = unknown");
            } else {
                System.out.println("height = " + height.get());
            }


            //
            // Convert image to mono 8
            //
            // *** NOTES ***
            // Images not gotten from a camera directly must be created and
            // destroyed. This includes any image copies, conversions, or
            // otherwise. Basically, if the image was gotten, it should be
            // released, if it was created, it needs to be destroyed.
            //
            // Images can be converted between pixel formats by using the
            // appropriate enumeration value. Unlike the original image, the
            // converted one does not need to be released as it does not affect the
            // camera buffer.
            //
            // Optionally, the color processing algorithm can also be set using
            // the alternate spinImageConvertEx() function.
            //
            // *** LATER ***
            // The converted image was created, so it must be destroyed to avoid
            // memory leaks.
            //
            spinImage hConvertedImage = new spinImage(); //NULL;
            err = spinImageCreateEmpty(hConvertedImage);
            if (printOnError(err, "Unable to create image. Non-fatal error.")) {
                hasFailed = true;
            }
            err = spinImageConvert(hResultImage, spinPixelFormatEnums.PixelFormat_Mono8, hConvertedImage);
            if (printOnError(err, "\"Unable to convert image. Non-fatal error.")) {
                hasFailed = true;
            }

            if (!hasFailed) {
                // Create a unique filename
                String filename = lenDeviceSerialNumber.get() == 0
                        ? ("Acquisition-C-" + imageCnt + ".jpg")
                        : ("Acquisition-C-" + deviceSerialNumber.getString().trim() + "-" + imageCnt + ".jpg");

                //
                // Save image
                //
                // *** NOTES ***
                // The standard practice of the examples is to use device serial
                // numbers to keep images of one device from overwriting those of
                // another.
                //
                err = spinImageSave(hConvertedImage, new BytePointer(filename), spinImageFileFormat.JPEG);
                if (!printOnError(err, "Unable to save image. Non-fatal error.")) {
                    System.out.println("Image saved at " + filename + "\n");
                }
            }

            //
            // Destroy converted image
            //
            // *** NOTES ***
            // Images that are created must be destroyed in order to avoid memory
            // leaks.
            //
            err = spinImageDestroy(hConvertedImage);
            printOnError(err, "Unable to destroy image. Non-fatal error.");
            //
            // Release image from camera
            //
            // *** NOTES ***
            // Images retrieved directly from the camera (i.e. non-converted
            // images) need to be released in order to keep from filling the
            // buffer.
            //
            err = spinImageRelease(hResultImage);
            printOnError(err, "Unable to release image. Non-fatal error.");
        }
        //
        // End acquisition
        //
        // *** NOTES ***
        // Ending acquisition appropriately helps ensure that devices clean up
        // properly and do not need to be power-cycled to maintain integrity.
        //
        err = spinCameraEndAcquisition(hCam);
        printOnError(err, "Unable to end acquisition.");
        return err;
    }


    /**
     * This function acts as the body of the example; please see NodeMapInfo_C
     * example for more in-depth comments on setting up cameras.
     */
    private static spinError runSingleCamera(spinCamera hCam) {
        spinError err;
        // Retrieve TL device nodemap and print device information
        spinNodeMapHandle hNodeMapTLDevice = new spinNodeMapHandle();
        err = spinCameraGetTLDeviceNodeMap(hCam, hNodeMapTLDevice);
        if (!printOnError(err, "Unable to retrieve TL device nodemap .")) {
            err = printDeviceInfo(hNodeMapTLDevice);
        }

        // Initialize camera
        err = spinCameraInit(hCam);
        if (printOnError(err, "Unable to initialize camera.")) {
            return err;
        }

        // Retrieve GenICam nodemap
        spinNodeMapHandle hNodeMap = new spinNodeMapHandle();
        err = spinCameraGetNodeMap(hCam, hNodeMap);
        if (printOnError(err, "Unable to retrieve GenICam nodemap.")) {
            return err;
        }

        // Acquire images
        err = acquireImages(hCam, hNodeMap, hNodeMapTLDevice);
        if (printOnError(err, "acquireImages")) {
            return err;
        }

        // Deinitialize camera
        err = spinCameraDeInit(hCam);
        if (printOnError(err, "Unable to deinitialize camera.")) {
            return err;
        }
        return err;
    }


    /**
     * Example entry point; please see Enumeration_C example for more in-depth
     * comments on preparing and cleaning up the system.
     */
    public static void main(String[] args) {


        spinError err;

        // Since this application saves images in the current folder
        // we must ensure that we have permission to write to this folder.
        // If we do not have permission, fail right away.
        if (!new File(".").canWrite()) {
            System.out.println("Failed to create file in current folder.  Please check permissions.");
            return;
        }

        // Retrieve singleton reference to system object
        spinSystem hSystem = new spinSystem();
        err = spinSystemGetInstance(hSystem);
        exitOnError(err, "Unable to retrieve system instance.");

        // Retrieve list of cameras from the system
        spinCameraList hCameraList = new spinCameraList();
        err = spinCameraListCreateEmpty(hCameraList);
        exitOnError(err, "Unable to create camera list.");

        err = spinSystemGetCameras(hSystem, hCameraList);
        exitOnError(err, "Unable to retrieve camera list.");

        // Retrieve number of cameras
        SizeTPointer numCameras = new SizeTPointer(1);
        err = spinCameraListGetSize(hCameraList, numCameras);
        exitOnError(err, "Unable to retrieve number of cameras.");
        System.out.println("Number of cameras detected: " + numCameras.get() + "\n");
        // Finish if there are no cameras
        if (numCameras.get() == 0) {
            // Clear and destroy camera list before releasing system
            err = spinCameraListClear(hCameraList);
            exitOnError(err, "Unable to clear camera list.");

            err = spinCameraListDestroy(hCameraList);
            exitOnError(err, "Unable to destroy camera list.");

            // Release system
            err = spinSystemReleaseInstance(hSystem);
            exitOnError(err, "Unable to release system instance.");

            System.out.println("Not enough cameras!");
            return;
        }

        // Run example on each camera
        for (int i = 0; i < numCameras.get(); i++) {
            System.out.println("\nRunning example for camera " + i + "...");
            // Select camera
            spinCamera hCamera = new spinCamera();
            err = spinCameraListGet(hCameraList, i, hCamera);
            if (!printOnError(err, "Unable to retrieve camera from list.")) {
                // Run example
                err = runSingleCamera(hCamera);
                printOnError(err, "RunSingleCamera");
            }
            // Release camera
            err = spinCameraRelease(hCamera);
            printOnError(err, "Error releasing camera.");
            System.out.println("Camera " + i + " example complete...\n");
        }
        // Clear and destroy camera list before releasing system
        err = spinCameraListClear(hCameraList);
        exitOnError(err, "Unable to clear camera list.");

        err = spinCameraListDestroy(hCameraList);
        exitOnError(err, "Unable to destroy camera list.");

        // Release system
        err = spinSystemReleaseInstance(hSystem);
        exitOnError(err, "Unable to release system instance.");
    }
}
