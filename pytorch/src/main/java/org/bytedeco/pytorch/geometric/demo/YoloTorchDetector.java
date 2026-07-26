package org.bytedeco.pytorch.geometric.demo;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.jit.JitModule;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.javacpp.FloatPointer;

import static org.bytedeco.pytorch.global.torch.*;

//import torch
//from ultralytics import YOLO
//
//# 1. 加载 YOLO 模型 (以 v8/v9/v10 为例)
//model = YOLO("yolov8n.pt") 
//
//# 2. 导出为 TorchScript 格式
//# 注意：optimize=True 会进行算子融合，这对 C++ 加载非常友好
//model.export(format="torchscript", optimize=True)
//
//# 这会生成一个 yolov8n.torchscript 文件，这个文件不依赖 Python

public class YoloTorchDetector {

//    void testImag(){
//
//        // 使用 OpenCV 进行 Letterbox 缩放，保持长宽比
//        Mat image = imread("test.jpg");
//        resize(image, image, new Size(640, 640));
//// 将 Mat 转换为 Tensor
//        Tensor input = torch.from_blob(image.data(), ...);
//    }
    public static void main(String[] args) {
        // 1. 加载模型 (注意：在最新版 JavaCPP 中，load 返回的是 JitModule)
        String modelPath = "models/yolov8n.torchscript";
        JitModule module = torch.load(modelPath);

        // 2. 设置设备 (自动检测 GPU/CPU)
        Device device = new Device(torch.hasCUDA() ? kCUDA() :  kCPU());
        module.to(device);
        module.eval();

        // 3. 模拟图像数据预处理 (以 YOLO 默认的 1x3x640x640 为例)
        int batchSize = 1;
        int channels = 3;
        int height = 640;
        int width = 640;

        // 假设你已经从 OpenCV 获取了 float[] 类型的像素数据 (经过了 /255.0 归一化)
        float[] imageData = new float[batchSize * channels * height * width];

        try (Pointer scope = new Pointer()) { // 使用 try-with-resources 管理内存
            // 将数组转为 FloatPointer
            FloatPointer imgPtr = new FloatPointer(imageData);

            // 创建 Tensor (NCHW 格式)
            long[] shape = {batchSize, channels, height, width};
            TensorOptions tensorOpt = new TensorOptions().device(new DeviceOptional(device)).dtype(new ScalarTypeOptional(ScalarType.Float));
            Tensor inputTensor = torch.from_blob(imgPtr, shape, tensorOpt).to(device,ScalarType.Float);

            // 4. 执行推理
            // 根据你提供的源码：public native @ByVal IValue forward(@ByVal IValueVector inputs);
            IValueVector inputs = new IValueVector(new IValue(inputTensor));
            IValue output = module.forward(inputs);

            // 5. 获取结果 Tensor
            Tensor resultTensor = output.toTensor();

            // 注意：resultTensor 现在包含了所有候选框
            // YOLOv8 的输出格式通常是 [1, 84, 8400] (84 = 4个坐标 + 80个类别置信度)
            System.out.println("推理完成，输出维度: " + resultTensor.sizes().get(0)
                    + "x" + resultTensor.sizes().get(1));
        }
    }
}