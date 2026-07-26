package org.bytedeco.pytorch.data.dataframe.feature;

import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * Base class for all feature transformers (sklearn-style fit/transform on {@link DataFrame}).
 */
public abstract class BaseTransformer implements Serializable {
    private static final long serialVersionUID = 1L;

    protected boolean fitted = false;
    protected List<String> columns;

    public BaseTransformer(String... columns) {
        if (columns == null || columns.length == 0) {
            this.columns = new ArrayList<>();
        } else {
            this.columns = new ArrayList<>(new LinkedHashSet<>(Arrays.asList(columns)));
        }
    }

    public abstract BaseTransformer fit(DataFrame X);

    public abstract DataFrame transform(DataFrame X) throws Exception;

    public DataFrame fitTransform(DataFrame X) throws Exception {
        return fit(X).transform(X);
    }

    public boolean isFitted() {
        return fitted;
    }

    public List<String> getColumns() {
        return columns;
    }

    public void save(String filepath) throws IOException {
        if (!fitted) throw new IllegalStateException("Transformer not fitted: " + getClass().getSimpleName());
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filepath))) {
            oos.writeObject(this);
        }
    }

    @SuppressWarnings("unchecked")
    public static <T extends BaseTransformer> T load(String filepath, Class<T> clazz)
            throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filepath))) {
            Object obj = ois.readObject();
            if (clazz.isInstance(obj)) return (T) obj;
            throw new ClassCastException("Loaded object is not " + clazz.getName());
        }
    }

    public static BaseTransformer load(String filepath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filepath))) {
            return (BaseTransformer) ois.readObject();
        }
    }

    protected void requireFitted() {
        if (!fitted) throw new IllegalStateException("Not fitted: " + getClass().getSimpleName());
    }
}
