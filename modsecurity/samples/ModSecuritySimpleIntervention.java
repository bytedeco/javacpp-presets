/*
 * ModSecurity, http://www.modsecurity.org/
 * Copyright (c) 2015 - 2021 Trustwave Holdings, Inc. (http://www.trustwave.com/)
 *
 * You may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * If any of the files related to licensing are missing or if you have any
 * other questions related to licensing please contact Trustwave Holdings, Inc.
 * directly using the email address security@modsecurity.org.
 *
 */

import java.util.Optional;
import org.bytedeco.javacpp.*;
import org.bytedeco.modsecurity.*;

public class ModSecuritySimpleIntervention {
    private static final String BASIC_RULE =
            "SecRuleEngine On\n" +
            "SecRule REQUEST_URI \"@streq /attack\" \"id:1,phase:1,msg: \' Attack detected\' t:lowercase,deny\"";

    public static void main(String[]args){
        ModSecurity modSecurity = new ModSecurity();

        RulesSet rulesSet = new RulesSet();
        rulesSet.load(BASIC_RULE);

        Transaction transaction = new Transaction(modSecurity, rulesSet, null);
        transaction.processConnection("127.0.0.1", 4455, "", 80);
        transaction.processURI("https://modsecurity.org/attack", "GET", "1.0");
        transaction.addResponseHeader("HTTP/1.1", "200 OK");
        transaction.processResponseHeaders(200, "HTTP/1.1");
        transaction.processRequestBody();
        transaction.processRequestHeaders();

        ModSecurityIntervention modSecurityIntervention = new ModSecurityIntervention();
        boolean isIntervention = transaction.intervention(modSecurityIntervention);

        if (isIntervention){
            System.out.println("There is intervention !!!");
            logRuleMessages(transaction.m_rulesMessages());
        }
    }

    private static void logRuleMessages(RuleMessageList messageList){
        if (messageList != null && !messageList.isNull() && !messageList.empty()) {
            long size = messageList.size();
            System.out.println("MessageRuleSize " +  size);
            RuleMessageList.Iterator iterator = messageList.begin();
            for (int i = 0; i < size; i++) {
                logRuleMessage(iterator.get());
                iterator.increment();
            }
        }
    }

    private static void logRuleMessage(RuleMessage ruleMessage){
        System.out.println("RuleMessage id = "+ ruleMessage.m_ruleId()+ " message  = " + Optional.ofNullable(ruleMessage.m_message()).map(BytePointer::getString).orElse("NO_MESSAGE"));
    }
}
