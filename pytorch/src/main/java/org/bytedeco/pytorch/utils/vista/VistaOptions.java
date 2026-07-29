package org.bytedeco.pytorch.utils.vista;

/**
 * Options for {@link Vista#traceModel}, mirroring torchvista {@code trace_model(...)} kwargs.
 *
 * <pre>
 *   Vista.traceModel(model, inputs, VistaOptions.defaults()
 *       .collapseModulesAfterDepth(1)
 *       .height(800)
 *       .exportFormat(ExportFormat.HTML)
 *       .exportPath("/tmp/model.html"));
 * </pre>
 *
 * @see <a href="https://github.com/sachinhosmani/torchvista">torchvista</a>
 */
public final class VistaOptions {
    private boolean showNonGradientNodes = true;
    private int collapseModulesAfterDepth = 1;
    private Integer forcedModuleTracingDepth = null;
    private int height = 800;
    private Integer width = null;
    private ExportFormat exportFormat = null;
    private boolean showModuleAttrNames = false;
    private String exportPath = null;
    private boolean showCompressedView = false;
    private boolean openBrowser = true;
    private boolean evalMode = true;

    public static VistaOptions defaults() {
        return new VistaOptions();
    }

    public boolean showNonGradientNodes() {
        return showNonGradientNodes;
    }

    public VistaOptions showNonGradientNodes(boolean v) {
        this.showNonGradientNodes = v;
        return this;
    }

    public int collapseModulesAfterDepth() {
        return collapseModulesAfterDepth;
    }

    /**
     * Depth to initially expand nested modules in the interactive view;
     * {@code 0} collapses everything (nodes can still be expanded interactively).
     */
    public VistaOptions collapseModulesAfterDepth(int v) {
        this.collapseModulesAfterDepth = Math.max(0, v);
        return this;
    }

    public Integer forcedModuleTracingDepth() {
        return forcedModuleTracingDepth;
    }

    /**
     * Maximum depth of module internals to expand during the live forward pass.
     * {@code null} (default) expands only Sequential chains; composites stay
     * black-box leaves. Set to e.g. {@code 2} to open two levels of nesting.
     */
    public VistaOptions forcedModuleTracingDepth(Integer v) {
        this.forcedModuleTracingDepth = v;
        return this;
    }

    public int height() {
        return height;
    }

    public VistaOptions height(int v) {
        this.height = Math.max(100, v);
        return this;
    }

    public Integer width() {
        return width;
    }

    /** Canvas width in pixels; {@code null} → full available width ({@code 100%}). */
    public VistaOptions width(Integer v) {
        this.width = v;
        return this;
    }

    public ExportFormat exportFormat() {
        return exportFormat;
    }

    public VistaOptions exportFormat(ExportFormat v) {
        this.exportFormat = v;
        return this;
    }

    public boolean showModuleAttrNames() {
        return showModuleAttrNames;
    }

    /** Prefer attribute names ({@code fc1}) over class names ({@code LinearImpl}). */
    public VistaOptions showModuleAttrNames(boolean v) {
        this.showModuleAttrNames = v;
        return this;
    }

    public String exportPath() {
        return exportPath;
    }

    public VistaOptions exportPath(String v) {
        this.exportPath = v;
        return this;
    }

    public boolean showCompressedView() {
        return showCompressedView;
    }

    /**
     * Experimental: compress repeating Sequential/ModuleList nodes of the same
     * type with identical in/out dims into single "repeat" blocks.
     */
    public VistaOptions showCompressedView(boolean v) {
        this.showCompressedView = v;
        return this;
    }

    public boolean openBrowser() {
        return openBrowser;
    }

    /** When exporting HTML (or default display), open the file in the system browser. */
    public VistaOptions openBrowser(boolean v) {
        this.openBrowser = v;
        return this;
    }

    public boolean evalMode() {
        return evalMode;
    }

    /**
     * If true (default), call {@code model.eval()} before the tracing forward pass
     * so Dropout/BatchNorm behave deterministically. Set false to keep training mode.
     */
    public VistaOptions evalMode(boolean v) {
        this.evalMode = v;
        return this;
    }
}
