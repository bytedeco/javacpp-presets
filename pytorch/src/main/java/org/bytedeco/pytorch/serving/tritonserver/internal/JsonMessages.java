package org.bytedeco.pytorch.serving.tritonserver.internal;

import com.google.gson.*;

import java.lang.reflect.Type;

/**
 * JSON message helpers for Triton in-process API.
 *
 * <p>Corresponds to Python {@code tritonserver} message serialization.
 * Uses {@code com.google.gson} for both request/response JSON roundtrips.
 */
public final class JsonMessages {
    private static final Gson gson = new GsonBuilder()
            .registerTypeAdapter(long[].class, new LongArraySerializer())
            .setPrettyPrinting()
            .create();

    private JsonMessages() {}

    /** Serialize object to JSON. */
    public static String toJson(Object obj) {
        return gson.toJson(obj);
    }

    /** Deserialize JSON to object. */
    public static <T> T fromJson(String json, Class<T> clazz) {
        return gson.fromJson(json, clazz);
    }

    /** Parse JSON string to JsonElement. */
    public static JsonElement parse(String json) {
        return JsonParser.parseString(json);
    }

    /** True if the JSON is an error object. */
    public static boolean isError(JsonElement json) {
        return json.isJsonObject() && json.getAsJsonObject().has("error");
    }

    /** Convenience: deserialize error object. */
    public static ErrorMessage fromError(String json) {
        return fromJson(json, ErrorMessage.class);
    }

    /** Error message structure (mirrors Python TritonError JSON). */
    public static final class ErrorMessage {
        public int code;
        public String message;
        public String name;
    }

    /** Long[] serializer for JSON compatibility (Java has no native long[]). */
    private static class LongArraySerializer implements JsonSerializer<long[]> {
        @Override
        public JsonElement serialize(long[] src, Type typeOfSrc, JsonSerializationContext context) {
            JsonArray array = new JsonArray();
            for (long l : src) array.add(l);
            return array;
        }
    }
}
