/*
 * Copyright (C) 2014-2017 Samuel Audet, Bram Biesbrouck
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * Permission was received from Point Grey Research, Inc. to disclose the
 * information released by the application of this preset under the GPL,
 * as long as it is distributed as part of a substantially larger package,
 * and not as a standalone wrapper to the FlyCapture library.
 *
 */

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * Wrapper for Allied Vision Vimba library (the C++ API).
 *
 * @author Bram Biesbrouck
 */
@Properties(target = "org.bytedeco.javacpp.Vimba", value = {
                @Platform(value = { "linux-x86_64" },
                                include = {
                                                "<VimbaC/Include/VmbCommonTypes.h>",
                                                "<VimbaC/Include/VimbaC.h>",

                                                "<VimbaCPP/Include/VimbaCPPCommon.h>",
                                                //"<VimbaCPP/Include/SharedPointer.h>",
                                                //"<VimbaCPP/Include/SharedPointer_impl.h>",
                                                //"<VimbaCPP/Include/SharedPointerDefines.h>",
                                                "<VimbaCPP/Include/UserSharedPointerDefines.h>",
                                                "<VimbaCPP/Include/Mutex.h>",
                                                "<VimbaCPP/Include/BasicLockable.h>",

                                                "<VimbaCPP/Include/AncillaryData.h>",
                                                "<VimbaCPP/Include/ICameraFactory.h>",
                                                "<VimbaCPP/Include/IFrameObserver.h>",
                                                "<VimbaCPP/Include/EnumEntry.h>",
                                                "<VimbaCPP/Include/ICameraListObserver.h>",
                                                "<VimbaCPP/Include/IInterfaceListObserver.h>",
                                                "<VimbaCPP/Include/IFeatureObserver.h>",
                                                "<VimbaCPP/Include/Interface.h>",

                                                //"<VimbaCPP/Include/FileLogger.h>",
                                                //"<VimbaCPP/Include/LoggerDefines.h>",
                                                //"<VimbaCPP/Include/UserLoggerDefines.h>",
                                                "<VimbaCPP/Include/Frame.h>",
                                                "<VimbaCPP/Include/IRegisterDevice.h>",
                                                "<VimbaCPP/Include/Camera.h>",
                                                "<VimbaCPP/Include/Feature.h>",
                                                "<VimbaCPP/Include/FeatureContainer.h>",

                                                "<VimbaCPP/Include/VimbaSystem.h>",
                                                //"<VimbaCPP/Include/VimbaCPP.h>",
                                },
                                link = { "VimbaCPP" }
                )
})
public class Vimba implements InfoMapper
{
    public void map(InfoMap infoMap)
    {
        //        infoMap.put(new Info("IMEXPORT").cppTypes().annotations());

        infoMap.put(new Info("defined (_WIN32)").define(false));
        infoMap.put(new Info("__cplusplus").define());

        infoMap.put(new Info("Logger").skip());

        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::Interface>").annotations("@SharedPtr").pointerTypes("Interface"));
        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::Camera>").annotations("@SharedPtr").pointerTypes("Camera"));
        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::Feature>").annotations("@SharedPtr").pointerTypes("Feature"));
        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::FeatureContainer>").annotations("@SharedPtr").pointerTypes("FeatureContainer"));
        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::IFeatureObserver>").annotations("@SharedPtr").pointerTypes("IFeatureObserver"));
        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::Frame>").annotations("@SharedPtr").pointerTypes("Frame"));
        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::FrameHandler>").annotations("@SharedPtr").pointerTypes("FrameHandler"));
        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::IFrameObserver>").annotations("@SharedPtr").pointerTypes("IFrameObserver"));
        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::AncillaryData>").annotations("@SharedPtr").pointerTypes("AncillaryData"));
        infoMap.put(new Info("std::shared_ptr<const AVT::VmbAPI::AncillaryData>").annotations("@SharedPtr").pointerTypes("ConstAncillaryData"));
        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::ICameraFactory>").annotations("@SharedPtr").pointerTypes("ICameraFactory"));
        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::ICameraListObserver>").annotations("@SharedPtr").pointerTypes("ICameraListObserver"));
        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::IInterfaceListObserver>").annotations("@SharedPtr").pointerTypes("IInterfaceListObserver"));
        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::Mutex>").annotations("@SharedPtr").pointerTypes("Mutex"));
        infoMap.put(new Info("std::shared_ptr<AVT::VmbAPI::BasicLockable>").annotations("@SharedPtr").pointerTypes("BasicLockable"));
    }
}
