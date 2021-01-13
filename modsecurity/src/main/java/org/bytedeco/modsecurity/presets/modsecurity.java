package org.bytedeco.modsecurity.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * @author Artem Martynenko artem7mag@gmail.com
 **/
@Properties(value = {
        @Platform(include = {
                "modsecurity/actions/action.h",
                "modsecurity/collection/collection.h",
                "modsecurity/collection/collections.h",
                "modsecurity/anchored_set_variable.h",
                "modsecurity/anchored_variable.h",
                "modsecurity/audit_log.h",
                "modsecurity/debug_log.h",
                "modsecurity/intervention.h",
                "modsecurity/modsecurity.h",
                "modsecurity/rule.h",
                "modsecurity/rule_marker.h",
                "modsecurity/rule_message.h",
                "modsecurity/rule_unconditional.h",
                "modsecurity/rule_with_actions.h",
                "modsecurity/rule_with_operator.h",
                "modsecurity/rules.h",
                "modsecurity/rules_exceptions.h",
                "modsecurity/rules_set.h",
                "modsecurity/rules_set_phases.h",
                "modsecurity/rules_set_properties.h",
                "modsecurity/transaction.h",
                "modsecurity/variable_origin.h",
                "modsecurity/variable_value.h"})
},
        target = "org.bytedeco.modsecurity",
        global = "org.bytedeco.modsecurity.global"
)
public class modsecurity implements InfoMapper {
    static {
        Loader.checkVersion( "org.bytedeco" , "modsecurity");
    }
    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("std::list<std::string>").pointerTypes("StringList").define());
        infoMap.put(new Info("std::list<int>").pointerTypes("IntList").define());
        infoMap.put(new Info("std::list<std::pair<int,int> >").pointerTypes("IntIntPairList").define());
        infoMap.put(new Info("std::list<std::pair<int,int> >").pointerTypes("IntIntPairList").define());
        infoMap.put(new Info("std::list<std::pair<std::string,std::string> >").pointerTypes("StringStringPairList").define());
        infoMap.put(new Info("std::list<std::pair<std::string,std::string> >").pointerTypes("StringStringPairList").define());
        infoMap.put(new Info("std::list<std::pair<int,std::string> >").pointerTypes("IntStringPairList").define());
        infoMap.put(new Info("std::list<modsecurity::RuleMessage>").pointerTypes("RuleMessageList").define());
        infoMap.put(new Info("std::map<std::string,std::string>").pointerTypes("StringStringMap").define());
        infoMap.put(new Info("std::shared_ptr<modsecurity::RequestBodyProcessor::MultipartPartTmpFile>").annotations("@SharedPtr").pointerTypes("MultipartPartTmpFile"));
        infoMap.put(new Info("std::shared_ptr<modsecurity::Rule>").annotations("@SharedPtr").pointerTypes("Rule"));
        infoMap.put(new Info("std::shared_ptr<modsecurity::actions::Action>").annotations("@SharedPtr").pointerTypes("Action"));
    }
}


