package org.bytedeco.pytorch.utils.visdom;

/**
 * HTTP response from a Visdom server endpoint.
 *
 * <p>For successful plot calls Visdom typically returns the window id as a
 * plain string (sometimes quoted). {@link #windowId()} strips surrounding quotes.
 */
public final class VisdomResponse {
    private final int statusCode;
    private final String body;
    private final String endpoint;

    public VisdomResponse(int statusCode, String body) {
        this(statusCode, body, "events");
    }

    public VisdomResponse(int statusCode, String body, String endpoint) {
        this.statusCode = statusCode;
        this.body = body == null ? "" : body;
        this.endpoint = endpoint == null ? "events" : endpoint;
    }

    public int statusCode() { return statusCode; }
    public String body() { return body; }
    public String endpoint() { return endpoint; }

    public boolean ok() { return statusCode >= 200 && statusCode < 300; }

    /** Window id returned by the server (quotes stripped). */
    public String windowId() {
        String b = body.trim();
        if (b.length() >= 2 && b.charAt(0) == '"' && b.charAt(b.length() - 1) == '"') {
            return b.substring(1, b.length() - 1);
        }
        return b;
    }

    @Override
    public String toString() {
        return "VisdomResponse{status=" + statusCode + ", endpoint=" + endpoint
                + ", body=" + (body.length() > 80 ? body.substring(0, 80) + "…" : body) + '}';
    }
}
