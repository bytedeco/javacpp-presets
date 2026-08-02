/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.lake;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.List;
import java.util.Set;

/**
 * Lake catalog: namespaces and tables.
 *
 * <p>Implementations wrap Doris information_schema, Iceberg Catalog, Gravitino REST, etc.</p>
 */
public interface LakeCatalog extends AutoCloseable {

    LakeFormat format();

    /** Declared capabilities for this catalog backend. */
    Set<LakeCapabilities> capabilities();

    List<String> listNamespaces() throws LakeException;

    List<String> listTables(String namespaceName) throws LakeException;

    boolean tableExists(String namespaceName, String table) throws LakeException;

    LakeTable loadTable(String namespaceName, String table) throws LakeException;

    default LakeTable loadTable(String table) throws LakeException {
        return loadTable(null, table);
    }

    /**
     * Create a table if the engine supports DDL through this client.
     * @return loaded table handle
     */
    default LakeTable createTable(String namespaceName, String table, LakeSchema schema,
                                  PartitionSpec partitionSpec, java.util.Map<String, String> props)
            throws LakeException {
        throw new LakeException(format(), "createTable", "not supported by this catalog");
    }

    default void dropTable(String namespaceName, String table, boolean ifExists) throws LakeException {
        throw new LakeException(format(), "dropTable", "not supported by this catalog");
    }

    /** Convenience: full table scan into one DataFrame (use {@link LakeScan} for large tables). */
    default DataFrame read(String namespaceName, String table) throws LakeException {
        return scan(namespaceName, table).collect();
    }

    LakeScan scan(String namespaceName, String table) throws LakeException;

    LakeWrite write(String namespaceName, String table) throws LakeException;

    default LakeStream stream(String namespaceName, String table) throws LakeException {
        return scan(namespaceName, table).stream();
    }

    @Override
    default void close() {}
}
