package org.bytedeco.pytorch.plot.vista;

/**
 * Export target for the interactive graph.
 *
 * <p>Matches torchvista {@code ExportFormat}: {@code html} (file + optional
 * browser open), {@code svg}/{@code png} reserved for future raster export
 * via the embedded renderer.
 */
public enum ExportFormat {
    HTML("html"),
    SVG("svg"),
    PNG("png");

    private final String value;

    ExportFormat(String value) {
        this.value = value;
    }

    public String value() {
        return value;
    }

    public static ExportFormat from(String s) {
        if (s == null || s.isEmpty()) return null;
        String lower = s.toLowerCase();
        for (ExportFormat f : values()) {
            if (f.value.equals(lower)) return f;
        }
        throw new IllegalArgumentException(
                "Invalid export format: " + s + ". Must be one of html, svg, png.");
    }

    @Override
    public String toString() {
        return value;
    }
}
