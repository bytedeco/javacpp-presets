package org.bytedeco.pytorch.data.dataframe.feature.text;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
 import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.*;
import java.util.stream.Collectors;

public class CountVectorizer extends BaseTransformer {
    private Set<String> vocabulary = new HashSet<>();
    private List<String> vocabList;
    private String textColumn;

    public CountVectorizer(String textColumn) {

//        super(columns);
        this.textColumn = textColumn;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        List<String> texts = X.column(textColumn).data().stream()
            .map(Object::toString)
            .collect(Collectors.toList());

        for (String text : texts) {
            String[] words = text.toLowerCase().split("\\s+");
            vocabulary.addAll(Arrays.asList(words));
        }

        vocabList = new ArrayList<>(vocabulary);
        Collections.sort(vocabList);
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("未拟合");

        List<String> texts = X.column(textColumn).data().stream()
            .map(Object::toString)
            .collect(Collectors.toList());

        DataFrame result = X.copy();

        for (String word : vocabList) {
            List<Integer> counts = texts.stream()
                .map(text -> {
                    String[] words = text.toLowerCase().split("\\s+");
                    return (int) Arrays.stream(words).filter(w -> w.equals(word)).count();
                })
                .collect(Collectors.toList());

            result = result.withColumn("count_" + word, counts);
        }

        return result;
    }
}
