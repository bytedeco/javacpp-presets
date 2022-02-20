/*
 * Copyright (C) 2021 Samuel Audet, Artem Martynenko
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

package org.bytedeco.modsecurity.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * @author Artem Martynenko artem7mag@gmail.com
 **/
@Properties(
        inherit = javacpp.class,
        value = @Platform(
                define = {"UNIQUE_PTR_NAMESPACE std", "SHARED_PTR_NAMESPACE std"},
                include = {
                        "modsecurity/audit_log.h",
                        "modsecurity/debug_log.h",
                        "modsecurity/intervention.h",
                        "modsecurity/rule_message.h",
                        "modsecurity/rules_set.h",
                        "modsecurity/rules_set_phases.h",
                        "modsecurity/rules_set_properties.h",
                        "modsecurity/collection/collection.h",
                        "modsecurity/modsecurity.h",
                        "modsecurity/transaction.h"},
                cinclude = "modsecurity/intervention.h",
                linkpath = {"lib","include"},
                includepath = {"lib","include"},
                link = "modsecurity@.3"),
        target = "org.bytedeco.modsecurity",
        global = "org.bytedeco.modsecurity.global.modsecurity")
public class modsecurity implements InfoMapper {
    static {
        Loader.checkVersion("org.bytedeco", "modsecurity");
    }

    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("std::set<std::string>").pointerTypes("StringSet").define());
        infoMap.put(new Info("std::list<std::string>").pointerTypes("StringList").define());
        infoMap.put(new Info("std::list<modsecurity::RuleMessage>").pointerTypes("RuleMessageList").define());
        infoMap.put(new Info("ModSecurityIntervention_t").pointerTypes("ModSecurityIntervention").define());
        infoMap.put(new Info("modsecurity").pointerTypes("ModSecurity").define());
        infoMap.put(new Info("std::basic_string<char>").annotations("@StdString").valueTypes("BytePointer").pointerTypes("@Cast({\"char*\", \"std::string\"}) BytePointer"));
        infoMap.put(new Info("string", "std::string").annotations("@StdString").valueTypes("BytePointer", "String").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"));
        infoMap.put(new Info("clock_t").skip());
        infoMap.put(new Info("std::list<int>").skip());
        infoMap.put(new Info("std::list<std::pair<int,int> >").skip());
        infoMap.put(new Info("std::list<std::pair<std::string,std::string> >").skip());
        infoMap.put(new Info("std::list<std::pair<int,std::string> >").skip());
        infoMap.put(new Info("std::ostringstream").skip());
        infoMap.put(new Info("Collections").skip());
        infoMap.put(new Info("collection::Collections").skip());
        infoMap.put(new Info("Collection").skip());
        infoMap.put(new Info("std::map<std::string,std::string>").skip());
        infoMap.put(new Info("std::shared_ptr<modsecurity::RequestBodyProcessor::MultipartPartTmpFile>").skip());
        infoMap.put(new Info("AnchoredSetVariable").skip());
        infoMap.put(new Info("AnchoredSetVariableTranslationProxy").skip());
        infoMap.put(new Info("TransactionSecMarkerManagement").skip());
        infoMap.put(new Info("TransactionAnchoredVariables").skip());
        infoMap.put(new Info("AnchoredVariable").skip());
        infoMap.put(new Info("VariableOrigin").skip());
        infoMap.put(new Info("VariableValue").skip());
        infoMap.put(new Info("Origins").skip());
        infoMap.put(new Info("Rules").skip());
        infoMap.put(new Info("Rule").skip());
        infoMap.put(new Info("RulesSetPhases").skip());
        infoMap.put(new Info("RuleWithActions").skip());
        infoMap.put(new Info("std::shared_ptr<std::string>").skip());
        infoMap.put(new Info("std::shared_ptr<RuleMessage>").skip());
        infoMap.put(new Info("std::unique_ptr<std::string>").skip());
        infoMap.put(new Info("std::shared_ptr<actions::Action>").skip());
    }
}


