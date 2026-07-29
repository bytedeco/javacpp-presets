package org.bytedeco.pytorch.utils.vista;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Parameter / attribute metadata for a Module node (click-to-inspect popup).
 *
 * <p>Matches torchvista {@code module_info[node] = {type, parameters, attributes, extra_repr?}}.
 * Attribute extraction reuses the same option-field patterns as
 * {@link org.bytedeco.pytorch.nn.ModulePrinter}.
 */
public final class ModuleInfo {
    private final String type;
    private final Map<String, ParamInfo> parameters;
    private final Map<String, Object> attributes;
    private final String extraRepr;

    public ModuleInfo(String type,
                      Map<String, ParamInfo> parameters,
                      Map<String, Object> attributes,
                      String extraRepr) {
        this.type = type == null ? "" : type;
        this.parameters = parameters == null
                ? Collections.emptyMap()
                : Collections.unmodifiableMap(new LinkedHashMap<>(parameters));
        this.attributes = attributes == null
                ? Collections.emptyMap()
                : Collections.unmodifiableMap(new LinkedHashMap<>(attributes));
        this.extraRepr = extraRepr;
    }

    public String type() {
        return type;
    }

    public Map<String, ParamInfo> parameters() {
        return parameters;
    }

    public Map<String, Object> attributes() {
        return attributes;
    }

    public String extraRepr() {
        return extraRepr;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("type", type);
        Map<String, Object> params = new LinkedHashMap<>();
        for (Map.Entry<String, ParamInfo> e : parameters.entrySet()) {
            params.put(e.getKey(), e.getValue().toMap());
        }
        m.put("parameters", params);
        m.put("attributes", new LinkedHashMap<>(attributes));
        if (extraRepr != null && !extraRepr.isEmpty()) {
            m.put("extra_repr", extraRepr);
        }
        return m;
    }

    /** One parameter entry: shape + requires_grad. */
    public static final class ParamInfo {
        private final long[] shape;
        private final boolean requiresGrad;

        public ParamInfo(long[] shape, boolean requiresGrad) {
            this.shape = shape == null ? new long[0] : shape.clone();
            this.requiresGrad = requiresGrad;
        }

        public long[] shape() {
            return shape.clone();
        }

        public boolean requiresGrad() {
            return requiresGrad;
        }

        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            // torchvista uses a JSON tuple (list) for shape
            java.util.List<Long> shapeList = new java.util.ArrayList<>(shape.length);
            for (long s : shape) shapeList.add(s);
            m.put("shape", shapeList);
            m.put("requires_grad", requiresGrad);
            return m;
        }

        @Override
        public String toString() {
            return "ParamInfo{shape=" + java.util.Arrays.toString(shape)
                    + ", requires_grad=" + requiresGrad + "}";
        }
    }

    @Override
    public String toString() {
        return "ModuleInfo{type=" + type + ", params=" + parameters.size()
                + ", attrs=" + attributes.size() + "}";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ModuleInfo)) return false;
        ModuleInfo that = (ModuleInfo) o;
        return Objects.equals(type, that.type)
                && Objects.equals(parameters, that.parameters)
                && Objects.equals(attributes, that.attributes);
    }

    @Override
    public int hashCode() {
        return Objects.hash(type, parameters, attributes);
    }
}
