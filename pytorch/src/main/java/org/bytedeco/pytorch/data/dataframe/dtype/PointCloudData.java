package org.bytedeco.pytorch.data.dataframe.dtype;

import org.bytedeco.pytorch.data.dataframe.enums.ColumnType;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

/**
 * 点云数据类型 - 继承AbstractLanceData
 * 支持多种点云格式：PCD, PLY, XYZ, LAS, OBJ, OFF
 * 可作为StructData的一部分或单独使用
 */
public class PointCloudData extends AbstractDataValue implements Serializable {
    private static final long serialVersionUID = 1L;

    private final List<float[]> points = new ArrayList<>();
    private float[] colors;
    private float[] normals;
    private float[] intensities;
    private int numPoints;
    private final Map<String, Object> metadata = new HashMap<>();
    private PointCloudFormat format;

    public enum PointCloudFormat {
        PCD,  // Point Cloud Data
        PLY,  // Polygon File Format
        XYZ,  // Simple XYZ format
        LAS,  // LAS/LAZ format
        OBJ,  // Wavefront OBJ
        OFF   // Object File Format
    }

    public PointCloudData() {
        this.format = PointCloudFormat.PCD;
    }

    public PointCloudData(PointCloudFormat format) {
        this.format = format;
    }

    public PointCloudData(float[] flatPoints, int pointCount) {
        this(PointCloudFormat.PCD, flatPoints, pointCount);
    }

    public PointCloudData(PointCloudFormat format, float[] flatPoints, int pointCount) {
        this.format = format;
        appendFlatPoints(flatPoints, pointCount);
        syncPointCount();
    }

    /**
     * 从文件加载点云数据
     */
    public static PointCloudData fromFile(String filePath) throws IOException {
        String ext = getFileExtension(filePath).toLowerCase();

        switch (ext) {
            case "pcd":
                return readPCD(filePath);
            case "ply":
                return readPLY(filePath);
            case "xyz":
                return readXYZ(filePath);
            case "las":
            case "laz":
                return readLAS(filePath);
            case "obj":
                return readOBJ(filePath);
            case "off":
                return readOFF(filePath);
            default:
                throw new IOException("Unsupported point cloud format: " + ext);
        }
    }

    /**
     * 转换为StructData（用于Arrow/Lance集成）
     */
    public StructData toStructData(String structName) {
        StructData struct = new StructData(structName, new HashMap<>());
        struct.addField("numPoints", ColumnType.INT32, getNumPoints());
        struct.addField("format", ColumnType.STRING, format != null ? format.name() : "UNKNOWN");
        struct.addField("points", ColumnType.VECTOR, new VectorData(flattenPoints(), "points"));
        float[] bbox = getBoundingBox();
        struct.addField("boundingBox", ColumnType.VECTOR, new VectorData(bbox, "boundingBox"));
        float[] centroid = getCentroid();
        struct.addField("centroid", ColumnType.VECTOR, new VectorData(centroid, "centroid"));
        return struct;
    }

    /**
     * 从StructData恢复PointCloudData
     */
    public static PointCloudData fromStructData(StructData struct) {
        PointCloudData pc = new PointCloudData();
        pc.format = PointCloudFormat.PCD;
        VectorData pointsVec = struct.getFieldValue("points");
        if (pointsVec != null) {
            float[] coords = pointsVec.getFloatVector();
            if (coords != null) {
                for (int i = 0; i + 2 < coords.length; i += 3) {
                    pc.points.add(new float[]{coords[i], coords[i + 1], coords[i + 2]});
                }
                pc.syncPointCount();
            }
        }
        try {
            VectorData colorsVec = struct.getFieldValue("colors");
            if (colorsVec != null) {
                pc.colors = colorsVec.getFloatVector();
            }
        } catch (Exception ignored) { }
        try {
            VectorData normalsVec = struct.getFieldValue("normals");
            if (normalsVec != null) {
                pc.normals = normalsVec.getFloatVector();
            }
        } catch (Exception ignored) { }
        return pc;
    }

    /**
     * 读取PCD格式
     */
    private static PointCloudData readPCD(String filePath) throws IOException {
        PointCloudData pc = new PointCloudData();
        pc.format = PointCloudFormat.PCD;
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            String dataType = "ascii";
            List<String> fields = new ArrayList<>();
            // 读取头部
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("FIELDS")) {
                    String[] parts = line.split("\\s+");
                    Collections.addAll(fields, Arrays.copyOfRange(parts, 1, parts.length));
                } else if (line.startsWith("DATA")) {
                    dataType = line.split("\\s+")[1];
                    break;
                }
            }
            List<Float> colorsList = new ArrayList<>();
            boolean hasColor = fields.contains("rgb") || fields.contains("rgba");
            // 读取数据
            if ("ascii".equals(dataType)) {
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.trim().split("\\s+");
                    if (parts.length >= 3) {
                        pc.points.add(new float[]{Float.parseFloat(parts[0]), Float.parseFloat(parts[1]), Float.parseFloat(parts[2])});
                        if (hasColor && parts.length >= 6) {
                            colorsList.add(Float.parseFloat(parts[3]));
                            colorsList.add(Float.parseFloat(parts[4]));
                            colorsList.add(Float.parseFloat(parts[5]));
                            if (parts.length >= 7) {
                                colorsList.add(Float.parseFloat(parts[6]));
                            }
                        }
                    }
                }
            }
            pc.syncPointCount();
            if (!colorsList.isEmpty()) {
                pc.colors = toFloatArray(colorsList);
            }
        }
        return pc;
    }

    /**
     * 读取PLY格式
     */
    private static PointCloudData readPLY(String filePath) throws IOException {
        PointCloudData pc = new PointCloudData();
        pc.format = PointCloudFormat.PLY;

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean binary = false;

            // 读取头部
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("format")) {
                    binary = line.contains("binary");
                } else if (line.equals("end_header")) {
                    break;
                }
            }

            // 读取ASCII数据
            if (!binary) {
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.trim().split("\\s+");
                    if (parts.length >= 3) {
                        pc.points.add(new float[]{Float.parseFloat(parts[0]), Float.parseFloat(parts[1]), Float.parseFloat(parts[2])});
                    }
                }
            }
        }
        pc.syncPointCount();
        return pc;
    }

    /**
     * 读取XYZ格式（最简单的格式）
     */
    private static PointCloudData readXYZ(String filePath) throws IOException {
        PointCloudData pc = new PointCloudData();
        pc.format = PointCloudFormat.XYZ;

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;

                String[] parts = line.split("[\\s,]+");
                if (parts.length >= 3) {
                    pc.points.add(new float[]{Float.parseFloat(parts[0]), Float.parseFloat(parts[1]), Float.parseFloat(parts[2])});
                }
            }
        }
        pc.syncPointCount();
        return pc;
    }

    /**
     * 读取LAS格式（简化版本）
     */
    private static PointCloudData readLAS(String filePath) throws IOException {
        PointCloudData pc = new PointCloudData();
        pc.format = PointCloudFormat.LAS;

        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {

            // 读取LAS头部（227字节）
            byte[] signature = new byte[4];
            dis.readFully(signature);

            if (!"LASF".equals(new String(signature))) {
                throw new IOException("Invalid LAS file signature");
            }

            dis.skipBytes(91); // 跳到点数据记录数位置
            long numPointRecords = Integer.toUnsignedLong(readLittleEndianInt(dis));

            dis.skipBytes(51); // 跳到X/Y/Z缩放因子
            double xScale = readLittleEndianDouble(dis);
            double yScale = readLittleEndianDouble(dis);
            double zScale = readLittleEndianDouble(dis);

            double xOffset = readLittleEndianDouble(dis);
            double yOffset = readLittleEndianDouble(dis);
            double zOffset = readLittleEndianDouble(dis);

            dis.skipBytes(48); // 跳到数据起始位置

            // 简化：只读取前10000个点或实际点数
            int pointsToRead = (int) Math.min(numPointRecords, 10000);
            for (int i = 0; i < pointsToRead; i++) {
                int x = readLittleEndianInt(dis);
                int y = readLittleEndianInt(dis);
                int z = readLittleEndianInt(dis);

                pc.points.add(new float[]{(float) (x * xScale + xOffset), (float) (y * yScale + yOffset), (float) (z * zScale + zOffset)});

                dis.skipBytes(8); // 跳过其他属性
            }
        }
        pc.syncPointCount();
        return pc;
    }

    /**
     * 读取OBJ格式
     */
    private static PointCloudData readOBJ(String filePath) throws IOException {
        PointCloudData pc = new PointCloudData();
        pc.format = PointCloudFormat.OBJ;

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("v ")) {
                    String[] parts = line.substring(2).trim().split("\\s+");
                    if (parts.length >= 3) {
                        pc.points.add(new float[]{Float.parseFloat(parts[0]), Float.parseFloat(parts[1]), Float.parseFloat(parts[2])});
                    }
                }
            }
        }
        pc.syncPointCount();
        return pc;
    }

    /**
     * 读取OFF格式
     */
    private static PointCloudData readOFF(String filePath) throws IOException {
        PointCloudData pc = new PointCloudData();
        pc.format = PointCloudFormat.OFF;

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line = reader.readLine();
            if (!"OFF".equals(line != null ? line.trim() : "")) {
                throw new IOException("Invalid OFF file format");
            }
            line = reader.readLine();
            if (line == null) {
                throw new IOException("Unexpected end of OFF header");
            }
            int numVertices = Integer.parseInt(line.trim().split("\\s+")[0]);
            for (int i = 0; i < numVertices; i++) {
                line = reader.readLine();
                if (line == null) break;
                String[] parts = line.trim().split("\\s+");
                if (parts.length >= 3) {
                    pc.points.add(new float[]{Float.parseFloat(parts[0]), Float.parseFloat(parts[1]), Float.parseFloat(parts[2])});
                }
            }
        }
        pc.syncPointCount();
        return pc;
    }

    /**
     * 保存为PCD格式
     */
    public void savePCD(String filePath) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            writer.println("# .PCD v0.7 - Point Cloud Data file format");
            writer.println("VERSION 0.7");
            writer.println("FIELDS x y z" + (colors != null ? " rgb" : ""));
            writer.println("SIZE 4 4 4" + (colors != null ? " 4" : ""));
            writer.println("TYPE F F F" + (colors != null ? " F" : ""));
            writer.println("COUNT 1 1 1" + (colors != null ? " 1" : ""));
            writer.println("WIDTH " + getNumPoints());
            writer.println("HEIGHT 1");
            writer.println("VIEWPOINT 0 0 0 1 0 0 0");
            writer.println("POINTS " + getNumPoints());
            writer.println("DATA ascii");

            for (int i = 0; i < getNumPoints(); i++) {
                float[] p = points.get(i);
                writer.printf("%.6f %.6f %.6f", p[0], p[1], p[2]);
                if (colors != null && colors.length >= (i + 1) * 3) {
                    writer.printf(" %.6f %.6f %.6f", colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]);
                }
                writer.println();
            }
        }
    }

    /**
     * 保存为XYZ格式
     */
    public void saveXYZ(String filePath) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            for (float[] point : points) {
                writer.printf("%.6f %.6f %.6f%n", point[0], point[1], point[2]);
            }
        }
    }

    /**
     * 获取点云边界框
     */
    public float[] getBoundingBox() {
        if (points.isEmpty()) {
            return new float[]{0, 0, 0, 0, 0, 0};
        }

        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE, minZ = Float.MAX_VALUE;
        float maxX = Float.MIN_VALUE, maxY = Float.MIN_VALUE, maxZ = Float.MIN_VALUE;

        for (float[] point : points) {
            float x = point[0];
            float y = point[1];
            float z = point[2];

            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            minZ = Math.min(minZ, z);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
            maxZ = Math.max(maxZ, z);
        }

        return new float[]{minX, minY, minZ, maxX, maxY, maxZ};
    }

    /**
     * 获取点云中心
     */
    public float[] getCentroid() {
        float[] centroid = new float[3];
        if (points.isEmpty()) {
            return centroid;
        }
        for (float[] point : points) {
            centroid[0] += point[0];
            centroid[1] += point[1];
            centroid[2] += point[2];
        }
        centroid[0] /= points.size();
        centroid[1] /= points.size();
        centroid[2] /= points.size();
        return centroid;
    }

    /**
     * 降采样（体素网格）
     */
    public PointCloudData voxelDownsample(float voxelSize) {
        Map<String, List<float[]>> voxelMap = new HashMap<>();

        for (float[] point : points) {
            int vx = (int) Math.floor(point[0] / voxelSize);
            int vy = (int) Math.floor(point[1] / voxelSize);
            int vz = (int) Math.floor(point[2] / voxelSize);
            String key = vx + "," + vy + "," + vz;
            voxelMap.computeIfAbsent(key, k -> new ArrayList<>()).add(point);
        }

        // 计算每个体素的中心
        PointCloudData result = new PointCloudData(this.format);
        for (List<float[]> cluster : voxelMap.values()) {
            float sumX = 0, sumY = 0, sumZ = 0;
            for (float[] p : cluster) {
                sumX += p[0];
                sumY += p[1];
                sumZ += p[2];
            }
            int count = cluster.size();
            result.points.add(new float[]{sumX / count, sumY / count, sumZ / count});
        }
        result.syncPointCount();
        return result;
    }

    // =========== AbstractDataValue 实现 ===========

    @Override
    public Number getNumericValue() {
        return getNumPoints();
    }

    @Override
    public String getDataType() {
        return "PointCloudData";
    }

    @Override
    public Object toArrowCompatible() {
        // 转换为StructData以支持Arrow
        return toStructData("point_cloud");
    }

    @Override
    public String getShortDesc() {
        return String.format("PointCloud[format=%s, points=%d]", format, getNumPoints());
    }

    // =========== Getters and Setters ===========

    public List<float[]> getPoints() {
        return Collections.unmodifiableList(points);
    }

    /**
     * 返回展平后的坐标数组（xyzxyz...），用于 Arrow 写入等场景。
     */
    public float[] getFlatPoints() {
        return flattenPoints();
    }

    public void setPoints(List<float[]> source) {
        points.clear();
        if (source != null) {
            for (float[] p : source) {
                points.add(Arrays.copyOf(p, p.length));
            }
        }
        syncPointCount();
    }

    public float[] getColors() { return colors; }
    public void setColors(float[] colors) { this.colors = colors; }
    public float[] getNormals() { return normals; }
    public void setNormals(float[] normals) { this.normals = normals; }
    public float[] getIntensities() { return intensities; }
    public void setIntensities(float[] intensities) { this.intensities = intensities; }
    public int getNumPoints() { return numPoints; }
    public void setNumPoints(int numPoints) { this.numPoints = numPoints; }
    public PointCloudFormat getFormat() { return format; }
    public void setFormat(PointCloudFormat format) { this.format = format; }
    public Map<String, Object> getMetadata() { return metadata; }

    @Override
    public String toString() {
        return String.format("PointCloudData[format=%s, points=%d, hasColors=%b, hasNormals=%b]", format, getNumPoints(), colors != null, normals != null);
    }

    private void syncPointCount() {
        this.numPoints = points.size();
    }

    private void appendFlatPoints(float[] flatPoints, int pointCount) {
        if (flatPoints == null) {
            throw new IllegalArgumentException("Point array cannot be null");
        }
        if (pointCount < 0) {
            throw new IllegalArgumentException("Point count must be non-negative");
        }
        if (flatPoints.length < pointCount * 3) {
            throw new IllegalArgumentException("Flat array length " + flatPoints.length + " is insufficient for " + pointCount + " points");
        }
        for (int i = 0; i < pointCount; i++) {
            int base = i * 3;
            points.add(new float[]{flatPoints[base], flatPoints[base + 1], flatPoints[base + 2]});
        }
    }

    public float[] flattenPoints() {
        float[] flat = new float[points.size() * 3];
        for (int i = 0; i < points.size(); i++) {
            float[] p = points.get(i);
            flat[i * 3] = p[0];
            flat[i * 3 + 1] = p[1];
            flat[i * 3 + 2] = p[2];
        }
        return flat;
    }

    private static String getFileExtension(String filePath) {
        int lastDot = filePath.lastIndexOf('.');
        return lastDot > 0 ? filePath.substring(lastDot + 1) : "";
    }

    private static float[] toFloatArray(List<Float> list) {
        float[] array = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }
        return array;
    }

    private static int readLittleEndianInt(DataInputStream dis) throws IOException {
        int ch1 = dis.read();
        int ch2 = dis.read();
        int ch3 = dis.read();
        int ch4 = dis.read();
        return ((ch4 << 24) + (ch3 << 16) + (ch2 << 8) + ch1);
    }

    private static double readLittleEndianDouble(DataInputStream dis) throws IOException {
        return ByteBuffer.wrap(new byte[]{
            dis.readByte(), dis.readByte(), dis.readByte(), dis.readByte(),
            dis.readByte(), dis.readByte(), dis.readByte(), dis.readByte()
        }).order(ByteOrder.LITTLE_ENDIAN).getDouble();
    }
}
