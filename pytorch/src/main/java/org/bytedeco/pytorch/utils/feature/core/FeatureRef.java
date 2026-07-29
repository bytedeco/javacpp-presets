/*
 * Versioned reference to a feature: project/view:feature@version
 * Used for lineage pins and FeatureService selections (Feast FeatureRef).
 */
package org.bytedeco.pytorch.utils.feature.core;

import java.util.Objects;

/** Immutable fully-qualified feature reference. */
public final class FeatureRef {

    private final String project;
    private final String viewName;
    private final String featureName;
    private final String version;

    public FeatureRef(String project, String viewName, String featureName, String version) {
        this.project = project != null && !project.isEmpty() ? project : "default";
        this.viewName = Objects.requireNonNull(viewName, "viewName");
        this.featureName = Objects.requireNonNull(featureName, "featureName");
        this.version = version != null && !version.isEmpty() ? version : "latest";
    }

    public static FeatureRef of(String viewName, String featureName) {
        return new FeatureRef("default", viewName, featureName, "latest");
    }

    public static FeatureRef of(String project, String viewName, String featureName) {
        return new FeatureRef(project, viewName, featureName, "latest");
    }

    /**
     * Parse {@code [[project/]view:]feature[@version]} forms:
     * <ul>
     *   <li>{@code click_7d}</li>
     *   <li>{@code user_stats:click_7d}</li>
     *   <li>{@code rec/user_stats:click_7d@v3}</li>
     * </ul>
     */
    public static FeatureRef parse(String raw) {
        Objects.requireNonNull(raw, "raw");
        String s = raw.trim();
        if (s.isEmpty()) throw new IllegalArgumentException("empty FeatureRef");

        String version = "latest";
        int at = s.lastIndexOf('@');
        if (at > 0) {
            version = s.substring(at + 1);
            s = s.substring(0, at);
        }

        String project = "default";
        String view = "";
        String feature;

        int slash = s.indexOf('/');
        int colon = s.indexOf(':');
        if (slash >= 0 && (colon < 0 || slash < colon)) {
            project = s.substring(0, slash);
            s = s.substring(slash + 1);
            colon = s.indexOf(':');
        }
        if (colon >= 0) {
            view = s.substring(0, colon);
            feature = s.substring(colon + 1);
        } else {
            feature = s;
        }
        if (feature.isEmpty()) throw new IllegalArgumentException("feature name empty in: " + raw);
        return new FeatureRef(project, view, feature, version);
    }

    public String project() {
        return project;
    }

    public String viewName() {
        return viewName;
    }

    public String featureName() {
        return featureName;
    }

    public String version() {
        return version;
    }

    public FeatureRef withVersion(String version) {
        return new FeatureRef(project, viewName, featureName, version);
    }

    public FeatureRef withView(String viewName) {
        return new FeatureRef(project, viewName, featureName, version);
    }

    /** Qualified name without version: {@code project/view:feature} or {@code view:feature}. */
    public String qualifiedName() {
        StringBuilder sb = new StringBuilder();
        if (!"default".equals(project)) {
            sb.append(project).append('/');
        }
        if (viewName != null && !viewName.isEmpty()) {
            sb.append(viewName).append(':');
        }
        sb.append(featureName);
        return sb.toString();
    }

    /** Full pin including version. */
    public String fullyQualified() {
        return qualifiedName() + "@" + version;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof FeatureRef)) return false;
        FeatureRef that = (FeatureRef) o;
        return project.equals(that.project)
                && viewName.equals(that.viewName)
                && featureName.equals(that.featureName)
                && version.equals(that.version);
    }

    @Override
    public int hashCode() {
        return Objects.hash(project, viewName, featureName, version);
    }

    @Override
    public String toString() {
        return fullyQualified();
    }
}
