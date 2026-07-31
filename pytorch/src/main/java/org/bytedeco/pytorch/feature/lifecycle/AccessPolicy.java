/*
 * In-process project role ACL (READ / WRITE / ADMIN).
 */
package org.bytedeco.pytorch.feature.lifecycle;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.concurrent.ConcurrentHashMap;

/** Simple feature-platform access control. */
public final class AccessPolicy {

    public enum Role {
        READ, WRITE, ADMIN
    }

    private final ConcurrentHashMap<String, EnumSet<Role>> grants = new ConcurrentHashMap<>();

    private static String key(String project, String principal) {
        return (project == null ? "default" : project) + "|" + (principal == null ? "anonymous" : principal);
    }

    public void grant(String project, String principal, Role... roles) {
        EnumSet<Role> set = EnumSet.noneOf(Role.class);
        if (roles != null) set.addAll(Arrays.asList(roles));
        grants.put(key(project, principal), set);
    }

    public boolean can(String project, String principal, Role needed) {
        if (needed == null) return true;
        EnumSet<Role> set = grants.get(key(project, principal));
        if (set == null) return false;
        if (set.contains(Role.ADMIN)) return true;
        if (needed == Role.READ) return set.contains(Role.READ) || set.contains(Role.WRITE);
        return set.contains(needed);
    }

    public void require(String project, String principal, Role needed) {
        if (!can(project, principal, needed)) {
            throw new SecurityException("access denied: " + principal + " lacks " + needed + " on " + project);
        }
    }
}
