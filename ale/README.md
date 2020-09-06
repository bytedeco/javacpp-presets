JavaCPP Presets for ALE
=======================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * The Arcade Learning Environment 0.6.1  https://github.com/mgbellemare/Arcade-Learning-Environment

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/ale/apidocs/


Sample Usage
------------
Here is a simple example of ALE ported to Java from this C++ source file:

 * https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/doc/examples/sharedLibraryInterfaceExample.cpp

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SharedLibraryInterfaceExample.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java -Dexec.args="rom_file"
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.ale</groupId>
    <artifactId>sharedlibraryinterfaceexample</artifactId>
    <version>1.5.4</version>
    <properties>
        <exec.mainClass>SharedLibraryInterfaceExample</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>ale-platform</artifactId>
            <version>0.6.1-1.5.4</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `SharedLibraryInterfaceExample.java` source file
```java
/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare,
 *  Matthew Hausknecht, and the Reinforcement Learning and Artificial Intelligence 
 *  Laboratory
 * Released under the GNU General Public License; see License.txt for details. 
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  sharedLibraryInterfaceExample.cpp 
 *
 *  Sample code for running an agent with the shared library interface. 
 **************************************************************************** */

import java.lang.System;
import java.util.Random;
import org.bytedeco.javacpp.*;
import org.bytedeco.ale.*;
import static org.bytedeco.ale.global.ale.*;

public class SharedLibraryInterfaceExample {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java SharedLibraryInterfaceExample rom_file");
            System.exit(1);
        }

        ALEInterface ale = new ALEInterface();

        // Get & Set the desired settings
        ale.setInt("random_seed", 123);
        //The default is already 0.25, this is just an example
        ale.setFloat("repeat_action_probability", 0.25f);

        ale.setBool("display_screen", true);
        ale.setBool("sound", true);

        // Load the ROM file. (Also resets the system for new settings to
        // take effect.)
        ale.loadROM(args[0]);

        // Get the vector of legal actions
        IntPointer legal_actions = ale.getLegalActionSet();

        // Play 10 episodes
        Random random = new Random();
        for (int episode = 0; episode < 10; episode++) {
            float totalReward = 0;
            while (!ale.game_over()) {
                int a = legal_actions.get(random.nextInt((int)legal_actions.limit()));
                // Apply the action and get the resulting reward
                float reward = ale.act(a);
                totalReward += reward;
            }
            System.out.println("Episode " + episode + " ended with score: " + totalReward);
            ale.reset_game();
        }

        System.exit(0);
    }
}
```
