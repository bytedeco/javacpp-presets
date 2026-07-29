package org.bytedeco.pytorch.utils.kafka;

/**
 * Unchecked failure from Kafka adapter operations
 * (connection, produce, consume, admin, serde, or offline file I/O).
 */
public class KafkaException extends RuntimeException {

    private final String operation;
    private final String topic;

    public KafkaException(String message) {
        this(message, null, null, null);
    }

    public KafkaException(String message, Throwable cause) {
        this(message, cause, null, null);
    }

    public KafkaException(String message, Throwable cause, String operation) {
        this(message, cause, operation, null);
    }

    public KafkaException(String message, Throwable cause, String operation, String topic) {
        super(message, cause);
        this.operation = operation;
        this.topic = topic;
    }

    public String operation() {
        return operation;
    }

    public String topic() {
        return topic;
    }
}
