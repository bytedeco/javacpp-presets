/*
 * Copyright (C) 2020 Samuel Audet
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
 */

package org.bytedeco.arrow.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = arrow.class,
    value = {
        @Platform(
            include = {
                "arrow/result.h",
                "arrow/flight/api.h",
                "arrow/flight/visibility.h",
                "arrow/flight/types.h",
                "arrow/flight/client_auth.h",
                "arrow/flight/client.h",
                "arrow/flight/server_auth.h",
                "arrow/flight/server.h",
            },
            link = "arrow_flight@.200",
            preload = {"libcrypto-1_1", "libssl-1_1"}
        ),
    },
    target = "org.bytedeco.arrow_flight",
    global = "org.bytedeco.arrow.global.arrow_flight"
)
public class arrow_flight implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "arrow_flight"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("ARROW_FLIGHT_EXPORT").cppTypes().annotations())
               .put(new Info("arrow::flight::FlightClientOptions::generic_options").skip())
               .put(new Info("arrow::Result<arrow::flight::FlightInfo>").pointerTypes("FlightInfoResult").define())
               .put(new Info("arrow::util::EqualityComparable<arrow::Result<arrow::flight::FlightInfo> >",
                             "arrow::flight::TimeoutDuration", "std::function<void(void*)>").cast().pointerTypes("Pointer"))
               .put(new Info("std::shared_ptr<arrow::flight::ClientMiddlewareFactory>").annotations("@SharedPtr").pointerTypes("ClientMiddlewareFactory"))
               .put(new Info("std::shared_ptr<arrow::flight::ServerMiddlewareFactory>").annotations("@SharedPtr").pointerTypes("ServerMiddlewareFactory"))
               .put(new Info("std::pair<std::string,std::shared_ptr<arrow::flight::ServerMiddlewareFactory> >").pointerTypes("ServerMiddlewareFactoryStringPair").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::flight::ClientMiddlewareFactory> >").pointerTypes("ClientMiddlewareFactoryVector").define())
               .put(new Info("std::vector<std::pair<std::string,std::shared_ptr<arrow::flight::ServerMiddlewareFactory> > >").pointerTypes("ServerMiddlewareFactoryStringPairVector").define())
               .put(new Info("arrow::flight::kSchemeGrpc").javaText("@Namespace(\"arrow::flight\") @MemberGetter public static native String kSchemeGrpc();"))
               .put(new Info("arrow::flight::kSchemeGrpcTcp").javaText("@Namespace(\"arrow::flight\") @MemberGetter public static native String kSchemeGrpcTcp();"))
               .put(new Info("arrow::flight::kSchemeGrpcUnix").javaText("@Namespace(\"arrow::flight\") @MemberGetter public static native String kSchemeGrpcUnix();"))
               .put(new Info("arrow::flight::kSchemeGrpcTls").javaText("@Namespace(\"arrow::flight\") @MemberGetter public static native String kSchemeGrpcTls();"))
        ;
    }
}
