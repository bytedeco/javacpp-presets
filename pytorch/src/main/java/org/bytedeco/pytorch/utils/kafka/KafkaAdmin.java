package org.bytedeco.pytorch.utils.kafka;

import org.apache.kafka.clients.admin.AdminClient;
import org.apache.kafka.clients.admin.AlterConfigOp;
import org.apache.kafka.clients.admin.AlterConfigsResult;
import org.apache.kafka.clients.admin.Config;
import org.apache.kafka.clients.admin.ConfigEntry;
import org.apache.kafka.clients.admin.CreatePartitionsResult;
import org.apache.kafka.clients.admin.CreateTopicsResult;
import org.apache.kafka.clients.admin.DeleteTopicsResult;
import org.apache.kafka.clients.admin.DescribeTopicsResult;
import org.apache.kafka.clients.admin.ListTopicsOptions;
import org.apache.kafka.clients.admin.NewPartitions;
import org.apache.kafka.clients.admin.NewTopic;
import org.apache.kafka.clients.admin.TopicDescription;
import org.apache.kafka.common.TopicPartitionInfo;
import org.apache.kafka.common.config.ConfigResource;
import org.apache.kafka.common.errors.TopicExistsException;

import java.io.Closeable;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Admin client wrapper: create / alter / describe topics with partitions &amp; replicas.
 *
 * <pre>{@code
 * try (KafkaAdmin admin = KafkaAdmin.connect(opts)) {
 *     admin.createTopic("rec.feature.log", 64, (short) 3);
 *     admin.alterTopicConfig("rec.feature.log", Map.of("min.insync.replicas", "2"));
 *     TopicInfo info = admin.describeTopic("rec.feature.log");
 * }
 * }</pre>
 */
public final class KafkaAdmin implements Closeable {

    private final AdminClient client;
    private final boolean ownClient;
    private final KafkaOptions options;
    private final Duration timeout;

    private KafkaAdmin(AdminClient client, boolean ownClient, KafkaOptions options, Duration timeout) {
        this.client = Objects.requireNonNull(client, "client");
        this.ownClient = ownClient;
        this.options = options == null ? KafkaOptions.defaults() : options;
        this.timeout = timeout == null ? Duration.ofSeconds(30) : timeout;
    }

    public static KafkaAdmin connect(String bootstrapServers) {
        return connect(KafkaOptions.builder().bootstrapServers(bootstrapServers).build());
    }

    public static KafkaAdmin connect(KafkaOptions options) {
        Objects.requireNonNull(options, "options");
        AdminClient c = AdminClient.create(options.adminProperties());
        return new KafkaAdmin(c, true, options, Duration.ofSeconds(30));
    }

    public static KafkaAdmin wrap(AdminClient client) {
        return new KafkaAdmin(client, false, KafkaOptions.defaults(), Duration.ofSeconds(30));
    }

    public AdminClient client() {
        return client;
    }

    public KafkaOptions options() {
        return options;
    }

    public KafkaAdmin timeout(Duration timeout) {
        return new KafkaAdmin(client, false, options, timeout);
    }

    // ── topics ───────────────────────────────────────────────────────────────

    /**
     * Create a topic. When {@code ifNotExists} (default from options) is true,
     * an existing topic is a no-op rather than an error.
     */
    public void createTopic(String name, int partitions, short replicationFactor) {
        createTopic(KafkaOptions.TopicOpts.builder()
                .name(name)
                .partitions(partitions)
                .replicationFactor(replicationFactor)
                .build(), options.ifNotExists());
    }

    public void createTopic(KafkaOptions.TopicOpts topic) {
        createTopic(topic, options.ifNotExists());
    }

    public void createTopic(KafkaOptions.TopicOpts topic, boolean ifNotExists) {
        Objects.requireNonNull(topic, "topic");
        NewTopic nt = new NewTopic(topic.name(), topic.partitions(), topic.replicationFactor());
        if (!topic.configs().isEmpty()) {
            nt.configs(new HashMap<>(topic.configs()));
        }
        createTopics(List.of(nt), ifNotExists);
    }

    public void createTopics(Collection<NewTopic> topics, boolean ifNotExists) {
        Objects.requireNonNull(topics, "topics");
        if (topics.isEmpty()) return;
        try {
            CreateTopicsResult result = client.createTopics(topics);
            result.all().get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            if (ifNotExists && isTopicExists(cause)) {
                return;
            }
            // partial success: some topics may already exist
            if (ifNotExists && hasOnlyTopicExists(e)) {
                return;
            }
            throw new KafkaException("createTopics failed: " + cause.getMessage(), cause, "createTopics", null);
        } catch (TimeoutException e) {
            throw new KafkaException("createTopics timeout after " + timeout, e, "createTopics", null);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new KafkaException("createTopics interrupted", e, "createTopics", null);
        }
    }

    public void deleteTopic(String name) {
        deleteTopics(List.of(name));
    }

    public void deleteTopics(Collection<String> names) {
        Objects.requireNonNull(names, "names");
        if (names.isEmpty()) return;
        try {
            DeleteTopicsResult result = client.deleteTopics(names);
            result.all().get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new KafkaException("deleteTopics failed: " + cause.getMessage(), cause, "deleteTopics", null);
        } catch (TimeoutException e) {
            throw new KafkaException("deleteTopics timeout after " + timeout, e, "deleteTopics", null);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new KafkaException("deleteTopics interrupted", e, "deleteTopics", null);
        }
    }

    public Set<String> listTopics() {
        return listTopics(false);
    }

    public Set<String> listTopics(boolean listInternal) {
        try {
            ListTopicsOptions opts = new ListTopicsOptions().listInternal(listInternal);
            return client.listTopics(opts).names().get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new KafkaException("listTopics failed: " + cause.getMessage(), cause, "listTopics", null);
        } catch (TimeoutException e) {
            throw new KafkaException("listTopics timeout after " + timeout, e, "listTopics", null);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new KafkaException("listTopics interrupted", e, "listTopics", null);
        }
    }

    public boolean topicExists(String name) {
        return listTopics(true).contains(name);
    }

    public TopicInfo describeTopic(String name) {
        Map<String, TopicInfo> all = describeTopics(List.of(name));
        TopicInfo info = all.get(name);
        if (info == null) {
            throw new KafkaException("topic not found: " + name, null, "describeTopic", name);
        }
        return info;
    }

    public Map<String, TopicInfo> describeTopics(Collection<String> names) {
        Objects.requireNonNull(names, "names");
        if (names.isEmpty()) return Map.of();
        try {
            DescribeTopicsResult result = client.describeTopics(names);
            Map<String, TopicDescription> desc =
                    result.allTopicNames().get(timeout.toMillis(), TimeUnit.MILLISECONDS);
            Map<String, TopicInfo> out = new LinkedHashMap<>();
            for (Map.Entry<String, TopicDescription> e : desc.entrySet()) {
                out.put(e.getKey(), TopicInfo.from(e.getValue()));
            }
            return out;
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new KafkaException("describeTopics failed: " + cause.getMessage(), cause, "describeTopics", null);
        } catch (TimeoutException e) {
            throw new KafkaException("describeTopics timeout after " + timeout, e, "describeTopics", null);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new KafkaException("describeTopics interrupted", e, "describeTopics", null);
        }
    }

    /**
     * Increase partition count (cannot decrease). No-op if already {@code >= totalPartitions}.
     */
    public void createPartitions(String topic, int totalPartitions) {
        Objects.requireNonNull(topic, "topic");
        if (totalPartitions < 1) throw new IllegalArgumentException("totalPartitions < 1");
        try {
            TopicInfo cur = describeTopic(topic);
            if (cur.partitions() >= totalPartitions) return;
            Map<String, NewPartitions> map = Map.of(topic, NewPartitions.increaseTo(totalPartitions));
            CreatePartitionsResult result = client.createPartitions(map);
            result.all().get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (KafkaException e) {
            throw e;
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new KafkaException("createPartitions failed: " + cause.getMessage(), cause, "createPartitions", topic);
        } catch (TimeoutException e) {
            throw new KafkaException("createPartitions timeout after " + timeout, e, "createPartitions", topic);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new KafkaException("createPartitions interrupted", e, "createPartitions", topic);
        }
    }

    public void alterTopicConfig(String topic, Map<String, String> configs) {
        Objects.requireNonNull(topic, "topic");
        Objects.requireNonNull(configs, "configs");
        if (configs.isEmpty()) return;
        ConfigResource resource = new ConfigResource(ConfigResource.Type.TOPIC, topic);
        List<AlterConfigOp> ops = new ArrayList<>(configs.size());
        for (Map.Entry<String, String> e : configs.entrySet()) {
            ops.add(new AlterConfigOp(new ConfigEntry(e.getKey(), e.getValue()), AlterConfigOp.OpType.SET));
        }
        try {
            Map<ConfigResource, Collection<AlterConfigOp>> map = Map.of(resource, ops);
            AlterConfigsResult result = client.incrementalAlterConfigs(map);
            result.all().get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new KafkaException("alterTopicConfig failed: " + cause.getMessage(), cause, "alterTopicConfig", topic);
        } catch (TimeoutException e) {
            throw new KafkaException("alterTopicConfig timeout after " + timeout, e, "alterTopicConfig", topic);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new KafkaException("alterTopicConfig interrupted", e, "alterTopicConfig", topic);
        }
    }

    public Map<String, String> describeTopicConfig(String topic) {
        Objects.requireNonNull(topic, "topic");
        ConfigResource resource = new ConfigResource(ConfigResource.Type.TOPIC, topic);
        try {
            Map<ConfigResource, Config> result = client.describeConfigs(List.of(resource))
                    .all()
                    .get(timeout.toMillis(), TimeUnit.MILLISECONDS);
            Config cfg = result.get(resource);
            if (cfg == null) return Map.of();
            Map<String, String> out = new LinkedHashMap<>();
            for (ConfigEntry e : cfg.entries()) {
                out.put(e.name(), e.value());
            }
            return out;
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new KafkaException("describeTopicConfig failed: " + cause.getMessage(), cause, "describeTopicConfig", topic);
        } catch (TimeoutException e) {
            throw new KafkaException("describeTopicConfig timeout after " + timeout, e, "describeTopicConfig", topic);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new KafkaException("describeTopicConfig interrupted", e, "describeTopicConfig", topic);
        }
    }

    private static boolean isTopicExists(Throwable t) {
        while (t != null) {
            if (t instanceof TopicExistsException) return true;
            String msg = t.getMessage();
            if (msg != null && msg.toLowerCase().contains("already exists")) return true;
            t = t.getCause();
        }
        return false;
    }

    private static boolean hasOnlyTopicExists(ExecutionException e) {
        return isTopicExists(e);
    }

    @Override
    public void close() {
        if (ownClient) {
            try {
                client.close(timeout);
            } catch (Exception ignored) {
            }
        }
    }

    /** Immutable topic description summary. */
    public record TopicInfo(
            String name,
            int partitions,
            short replicationFactor,
            boolean internal,
            List<PartitionInfo> partitionInfos
    ) {
        static TopicInfo from(TopicDescription d) {
            List<PartitionInfo> parts = new ArrayList<>();
            short rf = 0;
            for (TopicPartitionInfo p : d.partitions()) {
                int replicas = p.replicas() == null ? 0 : p.replicas().size();
                if (replicas > rf) rf = (short) replicas;
                int leader = p.leader() == null ? -1 : p.leader().id();
                List<Integer> replicaIds = new ArrayList<>();
                if (p.replicas() != null) {
                    p.replicas().forEach(n -> replicaIds.add(n.id()));
                }
                List<Integer> isrIds = new ArrayList<>();
                if (p.isr() != null) {
                    p.isr().forEach(n -> isrIds.add(n.id()));
                }
                parts.add(new PartitionInfo(p.partition(), leader,
                        Collections.unmodifiableList(replicaIds),
                        Collections.unmodifiableList(isrIds)));
            }
            return new TopicInfo(d.name(), d.partitions().size(), rf, d.isInternal(),
                    Collections.unmodifiableList(parts));
        }
    }

    public record PartitionInfo(int partition, int leader, List<Integer> replicas, List<Integer> isr) {}
}
