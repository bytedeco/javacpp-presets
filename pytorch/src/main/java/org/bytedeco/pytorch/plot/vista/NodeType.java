package org.bytedeco.pytorch.plot.vista;

/**
 * Node kinds in a torchvista-style forward-pass graph.
 *
 * <p>Values match the string labels expected by the interactive HTML renderer
 * (and the original torchvista JSON schema).
 *
 * @see <a href="https://github.com/sachinhosmani/torchvista">torchvista</a>
 */
public enum NodeType {
    MODULE("Module"),
    OPERATION("Operation"),
    INPUT("Input"),
    OUTPUT("Output"),
    CONSTANT("Constant"),
    PARAMETER("Parameter");

    private final String value;

    NodeType(String value) {
        this.value = value;
    }

    public String value() {
        return value;
    }

    public static NodeType fromValue(String v) {
        if (v == null) return MODULE;
        for (NodeType t : values()) {
            if (t.value.equalsIgnoreCase(v)) return t;
        }
        return MODULE;
    }

    @Override
    public String toString() {
        return value;
    }
}
