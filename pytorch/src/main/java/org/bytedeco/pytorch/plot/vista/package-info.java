/**
 * Interactive forward-pass graph visualisation (torchvista conceptual port).
 *
 * <p>Entry point: {@link Vista#traceModel(org.bytedeco.pytorch.nn.Module, Object)}.
 *
 * <p>Builds a module-level adjacency list by walking
 * {@link org.bytedeco.pytorch.nn.Module#named_children()}, expanding
 * {@code Sequential} chains, and recording real tensor shapes on edges during a
 * live forward pass. Renders a self-contained interactive HTML viewer
 * (pan / zoom / collapse / click-to-inspect).
 *
 * <p>Reuses existing jnitorch pieces:
 * <ul>
 *   <li>{@link org.bytedeco.pytorch.nn.ModulePrinter} — type names + attribute text</li>
 *   <li>{@link org.bytedeco.pytorch.nn.ModuleAsHelper} — typed module recovery</li>
 *   <li>{@link org.bytedeco.pytorch.utils.json.Json} — graph JSON payload</li>
 * </ul>
 *
 * @see Vista
 * @see VistaOptions
 * @see TraceGraph
 * @see <a href="https://github.com/sachinhosmani/torchvista">torchvista</a>
 */
package org.bytedeco.pytorch.plot.vista;
