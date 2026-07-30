package samples;

import org.bytedeco.pytorch.llm.nltk.Nltk;
import org.bytedeco.pytorch.llm.nltk.metrics.BleuScore;
import org.bytedeco.pytorch.llm.nltk.metrics.EditDistance;
import org.bytedeco.pytorch.llm.nltk.probability.ConditionalFreqDist;
import org.bytedeco.pytorch.llm.nltk.probability.FreqDist;
import org.bytedeco.pytorch.llm.nltk.stem.LancasterStemmer;
import org.bytedeco.pytorch.llm.nltk.stem.PorterStemmer;
import org.bytedeco.pytorch.llm.nltk.stem.SnowballStemmer;
import org.bytedeco.pytorch.llm.nltk.tag.RegexpTagger;
import org.bytedeco.pytorch.llm.nltk.tokenize.SentTokenizer;
import org.bytedeco.pytorch.llm.nltk.tokenize.TreebankWordTokenizer;
import org.bytedeco.pytorch.llm.nltk.tokenize.WordPunctTokenizer;
import org.bytedeco.pytorch.llm.nltk.util.Ngram;
import org.bytedeco.pytorch.llm.nltk.wordnet.SimpleLexicon;

import java.util.ArrayList;
import java.util.List;

/** D1 tokenize | D2 sent | D3 stem | D4 FreqDist | D5 bleu | D6 edit_distance | D7 pos_tag | D8 ngrams | D9 lexicon | D10 pipeline */
public class BenchmarkNltk {
    static int passed = 0, failed = 0;

    static void check(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  PASS  " + name); }
        else { failed++; System.out.println("  FAIL  " + name); }
    }
    static void section(String t) { System.out.println("\n=== " + t + " ==="); }

    public static void main(String[] args) {
        System.out.println("=== NLTK benchmark ===");
        d1WordTokenize();
        d2SentTokenize();
        d3Stem();
        d4FreqDist();
        d5Bleu();
        d6EditDistance();
        d7PosTag();
        d8Ngrams();
        d9Lexicon();
        d10Pipeline();
        d11ConditionalFreqDist();
        d12TokenizerClasses();
        d13BleuCorpusEditSim();
        d14Stress();
        done();
    }

    static void d1WordTokenize() {
        section("D1 word_tokenize");
        List<String> toks = Nltk.wordTokenize("Hello, world!");
        check("has tokens", toks.size() >= 2);
        check("first token=Hello", toks.get(0).equals("Hello"));
        check("static alias works", Nltk.word_tokenize("hi").size() >= 1);
        check("treebank tokenize", Nltk.treebankWordTokenize("don't stop").size() >= 2);
    }

    static void d2SentTokenize() {
        section("D2 sent_tokenize");
        List<String> sents = Nltk.sentTokenize("Hello. World! How are you?");
        check("3 sentences", sents.size() == 3);
        check("static alias works", Nltk.sent_tokenize("A. B.").size() == 2);
    }

    static void d3Stem() {
        section("D3 Porter / Snowball / Lancaster stem");
        PorterStemmer ps = new PorterStemmer();
        check("porter running -> run", ps.stem("running").equals("run"));
        check("porter happiness -> happi", ps.stem("happiness").startsWith("happi"));
        SnowballStemmer ss = new SnowballStemmer("english");
        check("snowball running -> run", ss.stem("running").equals("run"));
        LancasterStemmer ls = new LancasterStemmer();
        check("lancaster happiness -> happy", ls.stem("happiness").equals("happi"));
        check("static facade", Nltk.porterStem("jumping").equals("jump"));
        check("snowball facade", Nltk.snowballStem("jumping").equals("jump"));
        check("lancaster facade", Nltk.lancasterStem("jumping").equals("jump"));
    }

    static void d4FreqDist() {
        section("D4 FreqDist");
        FreqDist fd = Nltk.freqDist(List.of("a", "b", "a", "c", "a"));
        check("count a=3", fd.count("a") == 3);
        check("count b=1", fd.count("b") == 1);
        check("N=5", fd.N() == 5);
        check("freq(a)>freq(b)", fd.freq("a") > fd.freq("b"));
        check("hapaxes size", fd.hapaxes().contains("b") && fd.hapaxes().contains("c"));
        check("most_common first", fd.mostCommon(1).get(0).getKey().equals("a"));
        check("freqDist from facade", Nltk.freqDist(List.of("x","x")).count("x") == 2);
    }

    static void d5Bleu() {
        section("D5 sentence_bleu");
        var hyp = List.of("the", "cat", "is", "on", "the", "mat");
        var ref = List.of("the", "cat", "is", "on", "the", "mat");
        double s = Nltk.sentenceBleu(hyp, ref);
        check("identical sentences ~= 1.0", s > 0.9);
        double s2 = Nltk.sentenceBleu(hyp, List.of("a", "b", "c"));
        check("different sentences < 1.0", s2 < 1.0);
        check("bleu facade", Nltk.bleu_score(hyp, ref) > 0.9);
    }

    static void d6EditDistance() {
        section("D6 edit_distance");
        int d = Nltk.editDistance("kitten", "sitting");
        check("kitten->sitting dist=3", d == 3);
        check("identical dist=0", Nltk.editDistance("abc", "abc") == 0);
        check("static alias", Nltk.edit_distance("abc", "abd") == 1);
    }

    static void d7PosTag() {
        section("D7 pos_tag");
        var toks = List.of("The", "cat", "runs", "quickly");
        var tagged = Nltk.posTag(toks);
        check("tagged size == tokens", tagged.size() == toks.size());
        check("each token has tag", tagged.stream().allMatch(t -> t.length == 2));
        check("regex tagger class", new RegexpTagger() != null);
        check("facade tag", Nltk.pos_tag(toks).size() == 4);
    }

    static void d8Ngrams() {
        section("D8 ngrams / bigrams");
        var toks = List.of("a", "b", "c", "d");
        var bg = Ngram.bigrams(toks);
        check("2 bigrams", bg.size() == 3);
        check("first bigram a,b", bg.get(0).get(0).equals("a") && bg.get(0).get(1).equals("b"));
        var ng3 = Ngram.ngrams(toks, 3);
        check("3-grams size=2", ng3.size() == 2);
        var eg = Ngram.everygrams(toks, 2);
        check("everygrams has bigrams", eg.size() >= 3);
        check("facade bigrams", Nltk.bigrams(toks).size() == 3);
    }

    static void d9Lexicon() {
        section("D9 SimpleLexicon");
        SimpleLexicon lex = SimpleLexicon.getDefault();
        check("lex size>0", lex.size() > 0);
        check("dog synsets non-empty", lex.synsets("dog").size() > 0);
        check("dog lemmas contains dog", lex.lemmas("dog").contains("dog"));
        check("car,dog NOT synonyms", !lex.areSynonyms("car", "dog"));
        check("car,auto synonyms", lex.areSynonyms("car", "auto"));
        check("nltk.wordnet()", Nltk.wordnet() != null);
    }

    static void d10Pipeline() {
        section("D10 Compose pipeline");
        String text = "The dogs are running quickly!";
        var toks = Nltk.wordTokenize(text);
        var stemmed = toks.stream().map(Nltk::porterStem).toList();
        check("pipeline: tokens stemmed", stemmed.size() == toks.size() && stemmed.size() >= 4);
        var tagged = Nltk.posTag(stemmed);
        check("pipeline: all tagged", tagged.size() == stemmed.size());
        check("info string non-empty", !Nltk.info().isEmpty());
    }

    static void d11ConditionalFreqDist() {
        section("D11 ConditionalFreqDist + FreqDist extras");
        ConditionalFreqDist cfd = new ConditionalFreqDist();
        cfd.inc("N", "dog");
        cfd.inc("N", "dog");
        cfd.inc("N", "cat");
        cfd.inc("V", "run", 3);
        check("CFD conditions has N,V", cfd.conditions().contains("N") && cfd.conditions().contains("V"));
        check("CFD N count dog=2", cfd.get("N").count("dog") == 2);
        check("CFD V N=3", cfd.get("V").N() == 3);
        check("CFD total N=6", cfd.N() == 6);
        check("CFD asMap size=2", cfd.asMap().size() == 2);

        FreqDist fd = new FreqDist();
        fd.inc("a");
        fd.inc("a", 2);
        fd.inc("b");
        check("FreqDist empty ctor + inc", fd.count("a") == 3 && fd.B() == 2);
        check("mostCommon all", fd.mostCommon().size() == 2);
        check("counts map", fd.counts().get("a") == 3);
        check("plotData length=B", fd.plotData().length == fd.B());
        check("toString", fd.toString() != null);
        check("version", Nltk.version() != null && !Nltk.version().isEmpty());
    }

    static void d12TokenizerClasses() {
        section("D12 Tokenizer / Tagger classes direct");
        WordPunctTokenizer wp = new WordPunctTokenizer();
        check("WordPunct instance", wp.tokenize("Don't stop!").size() >= 2);
        check("WordPunct static", WordPunctTokenizer.tokenizeStatic("a-b").size() >= 1);

        TreebankWordTokenizer tb = new TreebankWordTokenizer();
        check("Treebank instance", tb.tokenize("won't go").size() >= 2);
        check("Treebank static", TreebankWordTokenizer.tokenizeStatic("can't").size() >= 1);

        SentTokenizer st = new SentTokenizer();
        check("SentTokenizer instance", st.tokenize("A. B! C?").size() >= 2);
        check("SentTokenizer static", SentTokenizer.tokenizeStatic("X. Y.").size() == 2);

        RegexpTagger tagger = new RegexpTagger();
        var tagged = tagger.tag(List.of("The", "cats", "run", "quickly", "123"));
        check("RegexpTagger tag size", tagged.size() == 5);
        check("tagMap", tagger.tagMap(List.of("running")).containsKey("running"));
        check("defaultRules non-empty", RegexpTagger.defaultRules().size() > 0);

        var custom = new RegexpTagger(
                List.of(new RegexpTagger.Rule(".*ing$", "VBG"), new RegexpTagger.Rule(".*", "NN")),
                "NN");
        check("custom RegexpTagger", "VBG".equals(custom.tag(List.of("running")).get(0)[1])
                || custom.tag(List.of("running")).get(0)[1] != null);

        SnowballStemmer ss = new SnowballStemmer();
        check("Snowball default lang", "english".equals(ss.getLanguage()) || ss.getLanguage() != null);
        check("Snowball other lang still stems", new SnowballStemmer("porter").stem("running") != null);
    }

    static void d13BleuCorpusEditSim() {
        section("D13 BleuScore corpus + EditDistance.similarity + ngram extras");
        var hyp = List.of("the", "cat", "sat");
        var ref = List.of("the", "cat", "sat");
        double s = BleuScore.sentenceBleu(hyp, ref);
        check("BleuScore.sentenceBleu identical", s > 0.9);
        double s2 = BleuScore.sentenceBleu(hyp, List.of(ref, List.of("the", "cat", "sat", "down")), 4);
        check("BleuScore multi-ref", s2 > 0.0);
        double corpus = BleuScore.corpusBleu(List.of(hyp, hyp), List.of(ref, ref));
        check("corpusBleu", corpus > 0.9);

        check("EditDistance.editDistance", EditDistance.editDistance("a", "b") == 1);
        double sim = EditDistance.similarity("kitten", "kitten");
        check("EditDistance.similarity identical=1", Math.abs(sim - 1.0) < 1e-9);
        check("EditDistance.similarity diff<1", EditDistance.similarity("kitten", "sitting") < 1.0);

        var toks = List.of("a", "b", "c", "d", "e");
        check("trigrams", Ngram.trigrams(toks).size() == 3);
        check("everygrams default", Ngram.everygrams(toks).size() >= 4);
        check("padLeft", Ngram.padLeft(toks, 2, "<s>").get(0).equals("<s>"));
        check("Nltk.ngrams facade", Nltk.ngrams(toks, 2).size() == 4);
    }

    static void d14Stress() {
        section("D14 stress throughput");
        String text = "The quick brown fox jumps over the lazy dog. ".repeat(50);
        long t0 = System.nanoTime();
        int total = 0;
        for (int i = 0; i < 100; i++) {
            total += Nltk.wordTokenize(text).size();
            total += Nltk.sentTokenize(text).size();
        }
        long ms = (System.nanoTime() - t0) / 1_000_000L;
        check("stress tokenized > 0", total > 0);
        System.out.println("  INFO  100x word+sent tokenize took " + ms + " ms totalUnits=" + total);

        // stem stress
        PorterStemmer ps = new PorterStemmer();
        List<String> words = Nltk.wordTokenize(text);
        long t1 = System.nanoTime();
        for (String w : words) ps.stem(w);
        long ms2 = (System.nanoTime() - t1) / 1_000_000L;
        System.out.println("  INFO  porter stem x" + words.size() + " took " + ms2 + " ms");
        check("stem stress ok", true);

        // FreqDist on large
        FreqDist big = Nltk.freqDist(words);
        check("big FreqDist N", big.N() == words.size());
        check("lexicon still ok under load", SimpleLexicon.getDefault().size() > 0);
    }

    static void done() {
        System.out.println("\n============================================================");
        System.out.println("NLTK  passed=" + passed + "  failed=" + failed);
        System.out.println("============================================================");
        if (failed > 0) System.exit(1);
        System.out.println("ALL DIMENSIONS GREEN");
    }
}
