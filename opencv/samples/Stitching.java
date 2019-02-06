/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
// License Agreement
// For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistribution's of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistribution's in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * The name of the copyright holders may not be used to endorse or promote products
// derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

import org.bytedeco.javacpp.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_stitching.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_stitching.*;

public class Stitching {
    static boolean try_use_gpu = false;
    static MatVector imgs = new MatVector();
    static String result_name = "result.jpg";

    public static void main(String[] args) {
        int retval = parseCmdArgs(args);
        if (retval != 0) {
            System.exit(-1);
        }

        Mat pano = new Mat();
        Stitcher stitcher = createStitcher(try_use_gpu);
        int status = stitcher.stitch(imgs, pano);

        if (status != Stitcher.OK) {
            System.out.println("Can't stitch images, error code = " + status);
            System.exit(-1);
        }

        imwrite(result_name, pano);
        System.exit(0);
    }

    static void printUsage() {
        System.out.println(
            "Rotation model images stitcher.\n\n"
          + "stitching img1 img2 [...imgN]\n\n"
          + "Flags:\n"
          + "  --try_use_gpu (yes|no)\n"
          + "      Try to use GPU. The default value is 'no'. All default values\n"
          + "      are for CPU mode.\n"
          + "  --output <result_img>\n"
          + "      The default is 'result.jpg'.");
    }

    static int parseCmdArgs(String[] args) {
        if (args.length == 0) {
            printUsage();
            return -1;
        }
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--help") || args.equals("/?")) {
                printUsage();
                return -1;
            } else if (args[i].equals("--try_use_gpu")) {
                if (args[i + 1].equals("no")) {
                    try_use_gpu = false;
                } else if (args[i + 1].equals("yes")) {
                    try_use_gpu = true;
                } else {
                    System.out.println("Bad --try_use_gpu flag value");
                    return -1;
                }
                i++;
            } else if (args[i].equals("--output")) {
                result_name = args[i + 1];
                i++;
            } else {
                Mat img = imread(args[i]);
                if (img.empty()) {
                    System.out.println("Can't read image '" + args[i] + "'");
                    return -1;
                }
                imgs.resize(imgs.size() + 1);
                imgs.put(imgs.size() - 1, img);
            }
        }
        return 0;
    }
}
