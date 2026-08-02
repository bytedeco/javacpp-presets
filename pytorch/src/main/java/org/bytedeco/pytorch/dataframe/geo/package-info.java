/**
 * Enterprise geospatial support for DataFrame: WKT/WKB/GeoJSON cells,
 * pure-Java H3/S2 indexing, spatial predicates, and spatial join.
 *
 * <p>Complex spatial operators may be pushed down to DuckDB; this package
 * provides client-side geometry cells and index keys without embedding PostGIS.
 *
 * @see org.bytedeco.pytorch.dataframe.geo.GeoData
 * @see org.bytedeco.pytorch.dataframe.geo.GeoJoin
 */
package org.bytedeco.pytorch.dataframe.geo;
