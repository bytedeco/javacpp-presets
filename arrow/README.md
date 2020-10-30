JavaCPP Presets for Arrow
=========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * Arrow 2.0.0  https://arrow.apache.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/arrow/apidocs/


Sample Usage
------------
Here is a simple example of Arrow ported to Java from this C++ source file:

 * https://github.com/apache/arrow/blob/apache-arrow-2.0.0/cpp/examples/arrow/row-wise-conversion-example.cc

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SimpleExample.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.arrow</groupId>
    <artifactId>row-wise-conversion-example</artifactId>
    <version>1.5.5-SNAPSHOT</version>
    <properties>
        <exec.mainClass>RowWiseConversionExample</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>arrow-platform</artifactId>
            <version>2.0.0-1.5.5-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `RowWiseConversionExample.java` source file
```java
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

import java.util.Arrays;
import org.bytedeco.javacpp.*;
import org.bytedeco.arrow.*;
import static org.bytedeco.arrow.global.arrow.*;

public class RowWiseConversionExample {
    // While we want to use columnar data structures to build efficient operations, we
    // often receive data in a row-wise fashion from other systems. In the following,
    // we want give a brief introduction into the classes provided by Apache Arrow by
    // showing how to transform row-wise data into a columnar table.
    //
    // The data in this example is stored in the following struct:
    static class data_row {
      long id;
      double cost;
      double[] cost_components;
      data_row(long id, double cost, double... cost_components) {
        this.id = id;
        this.cost = cost;
        this.cost_components = cost_components;
      }
      public String toString() {
        return "[id=" + id + ",cost=" + cost + ",cost_components=" + Arrays.toString(cost_components) + "]";
      }
    }

    public static void THROW_ON_FAILURE(Status status_) {
        if (!status_.ok()) {
          throw new RuntimeException(status_.message());
        }
    }

    // Transforming a vector of structs into a columnar Table.
    //
    // The final representation should be an `arrow::Table` which in turn
    // is made up of an `arrow::Schema` and a list of
    // `arrow::ChunkedArray` instances. As the first step, we will iterate
    // over the data and build up the arrays incrementally.  For this
    // task, we provide `arrow::ArrayBuilder` classes that help in the
    // construction of the final `arrow::Array` instances.
    //
    // For each type, Arrow has a specially typed builder class. For the primitive
    // values `id` and `cost` we can use the respective `arrow::Int64Builder` and
    // `arrow::DoubleBuilder`. For the `cost_components` vector, we need to have two
    // builders, a top-level `arrow::ListBuilder` that builds the array of offsets and
    // a nested `arrow::DoubleBuilder` that constructs the underlying values array that
    // is referenced by the offsets in the former array.
    static Status VectorToColumnarTable(data_row[] rows,
                                        Table[] table) {
      // The builders are more efficient using
      // arrow::jemalloc::MemoryPool::default_pool() as this can increase the size of
      // the underlying memory regions in-place. At the moment, arrow::jemalloc is only
      // supported on Unix systems, not Windows.
      MemoryPool pool = default_memory_pool();

      Int64Builder id_builder = new Int64Builder(int64(), pool);
      DoubleBuilder cost_builder = new DoubleBuilder(float64(), pool);
      ListBuilder components_builder = new ListBuilder(pool, new DoubleBuilder(float64(), pool));
      // The following builder is owned by components_builder.
      DoubleBuilder cost_components_builder =
          new DoubleBuilder(components_builder.value_builder());

      // Now we can loop over our existing data and insert it into the builders. The
      // `Append` calls here may fail (e.g. we cannot allocate enough additional memory).
      // Thus we need to check their return values. For more information on these values,
      // check the documentation about `arrow::Status`.
      for (data_row row : rows) {
        THROW_ON_FAILURE(id_builder.Append(row.id));
        THROW_ON_FAILURE(cost_builder.Append(row.cost));

        // Indicate the start of a new list row. This will memorise the current
        // offset in the values builder.
        THROW_ON_FAILURE(components_builder.Append());
        // Store the actual values. The final nullptr argument tells the underyling
        // builder that all added values are valid, i.e. non-null.
        THROW_ON_FAILURE(cost_components_builder.AppendValues(row.cost_components,
                                                              row.cost_components.length));
      }

      // At the end, we finalise the arrays, declare the (type) schema and combine them
      // into a single `arrow::Table`:
      Array id_array = new Array(null);
      THROW_ON_FAILURE(id_builder.Finish(id_array));
      Array cost_array = new Array(null);
      THROW_ON_FAILURE(cost_builder.Finish(cost_array));
      // No need to invoke cost_components_builder.Finish because it is implied by
      // the parent builder's Finish invocation.
      Array cost_components_array = new Array(null);
      THROW_ON_FAILURE(components_builder.Finish(cost_components_array));

      FieldVector schema_vector = new FieldVector(
          new Field("id", int64()), new Field("cost", float64()),
          new Field("cost_components", list(float64())));

      Schema schema = new Schema(schema_vector);

      // The final `table` variable is the one we then can pass on to other functions
      // that can consume Apache Arrow memory structures. This object has ownership of
      // all referenced data, thus we don't have to care about undefined references once
      // we leave the scope of the function building the table and its underlying arrays.
      table[0] = Table.Make(schema, new ArrayVector(id_array, cost_array, cost_components_array));

      return Status.OK();
    }

    static Status ColumnarTableToVector(Table table,
                                        data_row[][] rows) {
      // To convert an Arrow table back into the same row-wise representation as in the
      // above section, we first will check that the table conforms to our expected
      // schema and then will build up the vector of rows incrementally.
      //
      // For the check if the table is as expected, we can utilise solely its schema.
      FieldVector schema_vector = new FieldVector(
          new Field("id", int64()), new Field("cost", float64()),
          new Field("cost_components", list(float64())));
      Schema expected_schema = new Schema(schema_vector);

      if (!expected_schema.Equals(table.schema())) {
        // The table doesn't have the expected schema thus we cannot directly
        // convert it to our target representation.
        return new Status(StatusCode.Invalid, "Schemas are not matching!");
      }

      // As we have ensured that the table has the expected structure, we can unpack the
      // underlying arrays. For the primitive columns `id` and `cost` we can use the high
      // level functions to get the values whereas for the nested column
      // `cost_components` we need to access the C-pointer to the data to copy its
      // contents into the resulting `std::vector<double>`. Here we need to be care to
      // also add the offset to the pointer. This offset is needed to enable zero-copy
      // slicing operations. While this could be adjusted automatically for double
      // arrays, this cannot be done for the accompanying bitmap as often the slicing
      // border would be inside a byte.

      Int64Array ids =
          new Int64Array(table.column(0).chunk(0));
      DoubleArray costs =
          new DoubleArray(table.column(1).chunk(0));
      ListArray cost_components =
          new ListArray(table.column(2).chunk(0));
      DoubleArray cost_components_values =
          new DoubleArray(cost_components.values());
      // To enable zero-copy slices, the native values pointer might need to account
      // for this slicing offset. This is not needed for the higher level functions
      // like Value(…) that already account for this offset internally.
      DoublePointer ccv_ptr = cost_components_values.data().GetValuesDouble(1);

      rows[0] = new data_row[(int)table.num_rows()];
      for (int i = 0; i < rows[0].length; i++) {
        // Another simplification in this example is that we assume that there are
        // no null entries, e.g. each row is fill with valid values.
        long id = ids.Value(i);
        double cost = costs.Value(i);
        long first = cost_components.value_offset(i);
        long last = cost_components.value_offset(i + 1);
        double[] components_vec = new double[(int)(last - first)];
        ccv_ptr.position(first).capacity(last).get(components_vec);
        rows[0][i] = new data_row(id, cost, components_vec);
      }

      return Status.OK();
    }

    public static void main(String args[]) {
      data_row[] rows = {
          new data_row(1, 1.0, 1.0), new data_row(2, 2.0, 1.0, 2.0), new data_row(3, 3.0, 1.0, 2.0, 3.0)};

      Table[] table = new Table[1];
      THROW_ON_FAILURE(VectorToColumnarTable(rows, table));

      data_row[][] expected_rows = new data_row[1][];
      THROW_ON_FAILURE(ColumnarTableToVector(table[0], expected_rows));

      assert rows.length == expected_rows[0].length;

      System.out.println(Arrays.deepToString(rows));
      System.out.println(Arrays.deepToString(expected_rows[0]));
      System.exit(0);
    }
}
```
