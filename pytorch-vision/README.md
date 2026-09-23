JavaCPP Presets for PyTorch-Vision
===========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/pytorch-vision/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/pytorch-vision) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/pytorch-vision.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![pytorch-vision](https://github.com/bytedeco/javacpp-presets/workflows/pytorch-vision/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Apytorch-vision)  


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * torchvision 0.21.0  https://github.com/pytorch/vision

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
This module cam be use to load the torchvision operations. 

This way we can run models like fasterrcnn_resnet50_fpn using that have operactions specific from torchvision.

Sample Usage
------------
Here is a simple example of runnig a pytorch vision model on java.
The inspiraction for this example came from here:
 * https://github.com/pytorch/vision/tree/main/examples/cpp


We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `RunModel.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.pytorch</groupId>
    <artifactId>run-model</artifactId>
    <version>1.5.12-SNAPSHOT</version>
    <properties>
        <exec.mainClass>RunModel</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>pytorch-vision-platform</artifactId>
            <version>0.21.0-1.5.12-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```



### The `RunModel.java` source file
```java

import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorchvision.torch_vision;
import static org.bytedeco.pytorch.global.torch.*;

public class RunModel {
    {
        //Load torch_vision
        Loader.load(torch_vision.class);

    }
 

    public static void main(String[] args) throws Exception {
        //Load model
        JitModule jitModule = torch.load("fasterrcnn_resnet50_fpn_v2.pt");

        //Prepare input
        IValueVector inputs = new IValueVector();
        TensorList list = new TensorList();
        //Two random images
        list.push_back(torch.rand(3, 500, 500));
        list.push_back(torch.rand(3, 300, 300));

        inputs.push_back(new IValue(new GenericList(list)));

        //forward the images 
        IValue result = jitModule.forward(inputs);
        Tuple outputTuple = result.toTuple();
        IValue losses = outputTuple.elements().get(0); //ignore first field is the losses

        GenericList resultsOfImages = outputTuple.elements().get(1).toList(); // second is a list with the results
        for (int imageResultIndex = 0; imageResultIndex < resultsOfImages.size(); ++imageResultIndex) {
            //Result for each image is a dict with 3 tensors
            GenericDict dict = resultsOfImages.get(imageResultIndex).toGenericDict();
            Tensor tensorBoxes = dict.at(new IValue("boxes")).toTensor();
            Tensor tensorLabels = dict.at(new IValue("labels")).toTensor();
            Tensor tensorScores = dict.at(new IValue("scores")).toTensor();
        }

    }
}
```

### Export and save the model 

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
# Load the model
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
# Evaul mode
model.eval()
# Save the TorchScript model

script_module = torch.jit.script(model)
script_module.save("fasterrcnn_resnet50_fpn_v2.pt")


```