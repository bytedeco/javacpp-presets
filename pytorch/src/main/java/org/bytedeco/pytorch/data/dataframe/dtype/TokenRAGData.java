package org.bytedeco.pytorch.data.dataframe.dtype;
import java.util.*;

/**
 * 文本数据容器（支持分词结果）
 */
public class TokenRAGData extends AbstractDataValue implements StructuredData  {
    private String text;
    private List<String> tokens;
    private List<Integer> tokenIds;
    private String language;
    private Map<String, Object> metadata;

    // ===== RAG & 会话内存相关字段 =====
    /** 会话ID，用于区分不同对话 */
    private String conversationId;
    /** 当前轮次在会话中的序号，从0开始 */
    private int turnIndex;
    /** 角色: user / assistant / system 等 */
    private String role;
    /** 与本文本绑定的向量嵌入，用于检索 */
    private VectorData embedding;

    /**
     * 简单的内存向量数据库（进程内）。
     * key: conversationId
     * value: 该会话下所有消息（按时间顺序）
     */
    private static final Map<String, List<TokenRAGData>> MEMORY_STORE = new HashMap<>();

    /** 全局文本向量库，用于跨会话RAG检索 */
    private static final List<TokenRAGData> GLOBAL_CORPUS = new ArrayList<>();

    public TokenRAGData(String text) {
        this.text = text;
        this.metadata = new HashMap<>();
        this.conversationId = UUID.randomUUID().toString();
        this.turnIndex = 0;
        this.role = "user";
    }

    public TokenRAGData(String text, List<String> tokens) {
        this.text = text;
        this.tokens = tokens;
        this.metadata = new HashMap<>();
        this.conversationId = UUID.randomUUID().toString();
        this.turnIndex = 0;
        this.role = "user";
    }

    // ===== 新增 getter / setter =====
    public String getConversationId() { return conversationId; }
    public void setConversationId(String conversationId) { this.conversationId = conversationId; }

    public int getTurnIndex() { return turnIndex; }
    public void setTurnIndex(int turnIndex) { this.turnIndex = turnIndex; }

    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }

    public VectorData getEmbedding() { return embedding; }
    public void setEmbedding(VectorData embedding) { this.embedding = embedding; }

    // ===== 文本嵌入 & 降维 =====

    /**
     * 生成一个简单的文本 embedding（不依赖外部模型）。
     * 思路：
     *  - 统计字符/Token 的频率与位置特征
     *  - 返回固定维度的向量（例如 128 维）
     */
    public VectorData computeTextEmbedding() {
        if (text == null || text.isEmpty()) {
            this.embedding = new VectorData(new double[128], "empty_text_embedding");
            return this.embedding;
        }

        // 1. 基于字符的简单统计（前 96 个可见 ASCII 字符）
        double[] charHist = new double[96];
        int totalChars = 0;
        for (char c : text.toCharArray()) {
            if (c >= 32 && c < 128) {
                charHist[c - 32] += 1.0;
                totalChars++;
            }
        }
        if (totalChars > 0) {
            for (int i = 0; i < charHist.length; i++) {
                charHist[i] /= totalChars; // 归一化
            }
        }

        // 2. 简单的文本统计特征
        double length = text.length();
        double wordCount = tokens != null ? tokens.size() : Math.max(1, text.split("\\s+").length);
        double avgWordLen = length / wordCount;
        double digitRatio = text.chars().filter(Character::isDigit).count() / length;
        double upperRatio = text.chars().filter(Character::isUpperCase).count() / length;

        // 3. 组合为 128 维向量
        double[] vec = new double[128];
        System.arraycopy(charHist, 0, vec, 0, charHist.length); // 0-95
        vec[96] = length / 512.0;       // 简单缩放
        vec[97] = wordCount / 128.0;
        vec[98] = avgWordLen / 16.0;
        vec[99] = digitRatio;
        vec[100] = upperRatio;
        // 剩余维度用简单 hash 特征填充
        int hash = text.hashCode();
        Random r = new Random(hash);
        for (int i = 101; i < 128; i++) {
            vec[i] = r.nextDouble() - 0.5; // [-0.5, 0.5]
        }

        this.embedding = new VectorData(vec, "token_rag_text_embedding");
        return this.embedding;
    }

    /**
     * 使用简单 PCA-like 方法将 embedding 降维到给定维度（例如 32 维）。
     * 这里只做标准化 + 分块平均，模拟降维行为，避免引入复杂线性代数依赖。
     */
    public VectorData reduceEmbedding(int targetDim) {
        if (embedding == null) {
            computeTextEmbedding();
        }
        double[] src = embedding.toDoubleArray();
        int srcDim = src.length;
        if (targetDim >= srcDim) {
            // 不需要降维，直接返回原 embedding 的拷贝
            return new VectorData(Arrays.copyOf(src, src.length), embedding.getVectorName() + "_copy");
        }

        double[] reduced = new double[targetDim];
        int blockSize = (int) Math.ceil(srcDim * 1.0 / targetDim);
        for (int i = 0; i < targetDim; i++) {
            int start = i * blockSize;
            int end = Math.min(start + blockSize, srcDim);
            if (start >= end) break;
            double sum = 0.0;
            for (int j = start; j < end; j++) sum += src[j];
            reduced[i] = sum / (end - start);
        }

        return new VectorData(reduced, embedding.getVectorName() + "_reduced_" + targetDim);
    }

    // ===== 内存式会话管理 & RAG 检索 =====

    /**
     * 将当前消息追加到会话内存中，并可选择加入全局检索语料。
     */
    public void persistToMemory(boolean addToGlobalCorpus) {
        if (conversationId == null) {
            conversationId = UUID.randomUUID().toString();
        }
        // 计算嵌入
        if (this.embedding == null) {
            computeTextEmbedding();
        }

        List<TokenRAGData> history = MEMORY_STORE.computeIfAbsent(conversationId, k -> new ArrayList<>());
        this.turnIndex = history.size();
        history.add(this);

        if (addToGlobalCorpus) {
            GLOBAL_CORPUS.add(this);
        }
    }

    /**
     * 获取当前会话的历史消息（按时间顺序）。
     */
    public List<TokenRAGData> getConversationHistory() {
        if (conversationId == null) return Collections.emptyList();
        return MEMORY_STORE.getOrDefault(conversationId, Collections.emptyList());
    }

    /**
     * 基于当前 query 文本，从会话历史中检索最相关的 topK 条消息（简单余弦相似度）。
     */
    public List<TokenRAGData> retrieveFromConversationContext(String query, int topK) {
        if (conversationId == null) return Collections.emptyList();
        List<TokenRAGData> history = MEMORY_STORE.getOrDefault(conversationId, Collections.emptyList());
        if (history.isEmpty()) return Collections.emptyList();

        TokenRAGData queryNode = new TokenRAGData(query);
        queryNode.computeTextEmbedding();
        VectorData qEmb = queryNode.getEmbedding();

        List<ScoredNode> scored = new ArrayList<>();
        for (TokenRAGData node : history) {
            if (node.getEmbedding() == null) node.computeTextEmbedding();
            double score = qEmb.cosineSimilarity(node.getEmbedding());
            scored.add(new ScoredNode(node, score));
        }

        scored.sort((a, b) -> Double.compare(b.score, a.score));
        List<TokenRAGData> result = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, scored.size()); i++) {
            result.add(scored.get(i).node);
        }
        return result;
    }

    /**
     * 基于当前 query 文本，从全局语料中检索最相关的 topK 条消息。
     */
    public static List<TokenRAGData> globalRetrieve(String query, int topK) {
        if (GLOBAL_CORPUS.isEmpty()) return Collections.emptyList();
        TokenRAGData queryNode = new TokenRAGData(query);
        queryNode.computeTextEmbedding();
        VectorData qEmb = queryNode.getEmbedding();

        List<ScoredNode> scored = new ArrayList<>();
        for (TokenRAGData node : GLOBAL_CORPUS) {
            if (node.getEmbedding() == null) node.computeTextEmbedding();
            double score = qEmb.cosineSimilarity(node.getEmbedding());
            scored.add(new ScoredNode(node, score));
        }

        scored.sort((a, b) -> Double.compare(b.score, a.score));
        List<TokenRAGData> result = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, scored.size()); i++) {
            result.add(scored.get(i).node);
        }
        return result;
    }

    /**
     * 生成用于大模型输入的 RAG 上下文，把历史消息与检索结果拼接。
     */
    public String buildRagPrompt(String userQuery, int historyTurns, int retrievedDocs) {
        StringBuilder sb = new StringBuilder();
        sb.append("[system] You are a helpful assistant with access to conversation history and retrieved documents.\n");

        // 1. 最近 historyTurns 轮对话
        List<TokenRAGData> history = getConversationHistory();
        int startIdx = Math.max(0, history.size() - historyTurns);
        for (int i = startIdx; i < history.size(); i++) {
            TokenRAGData msg = history.get(i);
            sb.append("[" + (msg.role != null ? msg.role : "user") + "] ")
              .append(msg.text)
              .append("\n");
        }

        // 2. RAG 检索结果
        List<TokenRAGData> retrieved = retrieveFromConversationContext(userQuery, retrievedDocs);
        if (!retrieved.isEmpty()) {
            sb.append("[context] Retrieved relevant information:\n");
            int idx = 1;
            for (TokenRAGData doc : retrieved) {
                sb.append("  (" + idx++ + ") ")
                  .append(doc.text)
                  .append("\n");
            }
        }

        // 3. 当前用户问题
        sb.append("[user] ").append(userQuery).append("\n");
        sb.append("[assistant]");

        return sb.toString();
    }

    // ===== 内部辅助类 =====
    private static class ScoredNode {
        final TokenRAGData node;
        final double score;
        ScoredNode(TokenRAGData node, double score) {
            this.node = node;
            this.score = score;
        }
    }

    @Override
    public Number getNumericValue(){
        return null;
    }

    public String getText() { return text; }
    public void setText(String text) { this.text = text; }
    public List<String> getTokens() { return tokens; }
    public void setTokens(List<String> tokens) { this.tokens = tokens; }
    public List<Integer> getTokenIds() { return tokenIds; }
    public void setTokenIds(List<Integer> tokenIds) { this.tokenIds = tokenIds; }
    public String getLanguage() { return language; }
    public void setLanguage(String language) { this.language = language; }
    public Map<String, Object> getMetadata() { return metadata; }

    public int length() { return text != null ? text.length() : 0; }
    public int tokenCount() { return tokens != null ? tokens.size() : 0; }

    @Override
    public String getDataType() {
        return "TEXT";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow适配：返回原始文本（Arrow Utf8类型）
        return text;
    }

    @Override
    public String getShortDesc() {
        String preview = text != null && text.length() > 50 ? text.substring(0, 50) + "..." : text;
        return String.format("len=%d, tokens=%d, lang=%s, text='%s'",
                length(), tokenCount(), language, preview);
    }

    // ========== 重写有效性校验 ==========
    @Override
    public boolean isValid() {
        // 基础校验 + 文本专属校验：文本内容非空
        return super.isValid() && text != null && !text.trim().isEmpty();
    }

    @Override
    public int getSize() {
        // 文本大小：字符长度
        return length();
    }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("text", text);
        map.put("tokens", tokens);
        map.put("tokenIds", tokenIds);
        map.put("language", language);
        map.put("metadata", metadata);
        map.put("length", length());
        map.put("tokenCount", tokenCount());
        return map;
    }
    @Override
    public String toString() {
        String preview = text != null && text.length() > 50 ? text.substring(0, 50) + "..." : text;
        return String.format("TextData[len=%d, tokens=%d, text='%s']",
                length(), tokenCount(), preview);
    }
}

