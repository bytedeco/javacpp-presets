JavaCPP Presets for libfreenect
===============================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * libfreenect 0.5.3  http://openkinect.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/libfreenect/apidocs/


Sample Usage
------------
Here is a simple example of libfreenect ported to Java from this C source file:

 * https://github.com/OpenKinect/libfreenect/blob/v0.5.3/examples/glpclview.c

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/GLPCLView.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.libfreenect</groupId>
    <artifactId>glpclview</artifactId>
    <version>1.2</version>
    <properties>
        <exec.mainClass>GLPCLView</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>libfreenect</artifactId>
            <version>0.5.3-1.2</version>
        </dependency>
        <dependency>
          <groupId>org.jogamp.gluegen</groupId>
          <artifactId>gluegen-rt-main</artifactId>
          <version>2.3.1</version>
          <optional>true</optional>
        </dependency>
        <dependency>
          <groupId>org.jogamp.jogl</groupId>
          <artifactId>jogl-all-main</artifactId>
          <version>2.3.1</version>
          <optional>true</optional>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/GLPCLView.java` source file
```java
/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2010 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * Andrew Miller <amiller@dappervision.com>
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */

import com.jogamp.opengl.*;
import com.jogamp.opengl.awt.*;
import com.jogamp.opengl.glu.*;
import com.jogamp.opengl.util.*;
import java.awt.*;
import java.awt.event.*;
import java.nio.*;
import javax.swing.*;
import org.bytedeco.javacpp.*;
import static java.lang.Math.*;
import static org.bytedeco.javacpp.freenect.*;

public class GLPCLView {
    static GLU glu = new GLU();
    static GLCanvas canvas;
    static JFrame frame;
    static FPSAnimator animator;
    static int[] gl_rgb_tex = new int[1];
    static int mx = -1, my = -1;     // Previous mouse coordinates
    static int[] rotangles = {0, 0}; // Panning angles
    static float zoom = 1;           // Zoom factor
    static boolean color = true;     // Use the RGB texture or just draw it as color
    static IntBuffer indices = ByteBuffer.allocateDirect(4 * 480 * 640).order(ByteOrder.nativeOrder()).asIntBuffer();
    static ShortBuffer xyz = ByteBuffer.allocateDirect(2 * 480 * 640 * 3).order(ByteOrder.nativeOrder()).asShortBuffer();

    // Do the projection from u,v,depth to X,Y,Z directly in an opengl matrix
    // These numbers come from a combination of the ros kinect_node wiki, and
    // nicolas burrus' posts.
    static void LoadVertexMatrix(GL2 gl2) {
        float fx = 594.21f;
        float fy = 591.04f;
        float a = -0.0030711f;
        float b = 3.3309495f;
        float cx = 339.5f;
        float cy = 242.7f;
        float mat[] = {
            1/fx,     0,  0, 0,
            0,    -1/fy,  0, 0,
            0,       0,  0, a,
            -cx/fx, cy/fy, -1, b
        };
        gl2.glMultMatrixf(mat, 0);
    }


    // This matrix comes from a combination of nicolas burrus's calibration post
    // and some python code I haven't documented yet.
    static void LoadRGBMatrix(GL2 gl2) {
        float mat[] = {
             5.34866271e+02f,   3.89654806e+00f,   0.00000000e+00f,   1.74704200e-02f,
            -4.70724694e+00f,  -5.28843603e+02f,   0.00000000e+00f,  -1.22753400e-02f,
            -3.19670762e+02f,  -2.60999685e+02f,   0.00000000e+00f,  -9.99772000e-01f,
            -6.98445586e+00f,   3.31139785e+00f,   0.00000000e+00f,   1.09167360e-02f
        };
        gl2.glMultMatrixf(mat, 0);
    }

    static void mouseMoved(int x, int y) {
        if (mx >= 0 && my >= 0) {
            rotangles[0] += y - my;
            rotangles[1] += x - mx;
        }
        mx = x;
        my = y;
    }

    static void mousePress(int button, int state, int x, int y) {
        if (button == MouseEvent.BUTTON1 && state == MouseEvent.MOUSE_PRESSED) {
            mx = x;
            my = y;
        }
        if (button == MouseEvent.BUTTON1 && state == MouseEvent.MOUSE_RELEASED) {
            mx = -1;
            my = -1;
        }
    }

    static void no_kinect_quit() {
        System.out.println("Error: Kinect not connected?");
        frame.dispose();
        System.exit(1);
    }

    static void DrawGLScene(GL2 gl2) {
        ShortPointer depthPointer = new ShortPointer((Pointer)null);
        BytePointer rgbPointer = new BytePointer((Pointer)null);
        int[] ts = new int[1];
        if (freenect_sync_get_depth(depthPointer, ts, 0, FREENECT_DEPTH_11BIT) < 0) {
            no_kinect_quit();
        }
        if (freenect_sync_get_video(rgbPointer, ts, 0, FREENECT_VIDEO_RGB) < 0) {
            no_kinect_quit();
        }

        ShortBuffer depth = depthPointer.capacity(640 * 480).asBuffer();
        ByteBuffer rgb = rgbPointer.capacity(640 * 480 * 3).asBuffer();

        for (int i = 0; i < 480; i++) {
            for (int j = 0; j < 640; j++) {
                xyz.put(i * 640 * 3 + j * 3 + 0, (short)j);
                xyz.put(i * 640 * 3 + j * 3 + 1, (short)i);
                xyz.put(i * 640 * 3 + j * 3 + 2, depth.get(i * 640 + j));
                indices.put(i * 640 + j, i * 640 + j);
            }
        }

        gl2.glClear(GL2.GL_COLOR_BUFFER_BIT | GL2.GL_DEPTH_BUFFER_BIT);
        gl2.glLoadIdentity();

        gl2.glPushMatrix();
        gl2.glScalef(zoom, zoom, 1);
        gl2.glTranslatef(0, 0, -3.5f);
        gl2.glRotatef(rotangles[0], 1, 0, 0);
        gl2.glRotatef(rotangles[1], 0, 1, 0);
        gl2.glTranslatef(0, 0, 1.5f);

        LoadVertexMatrix(gl2);

        // Set the projection from the XYZ to the texture image
        gl2.glMatrixMode(GL2.GL_TEXTURE);
        gl2.glLoadIdentity();
        gl2.glScalef(1/640.0f,1/480.0f,1);
        LoadRGBMatrix(gl2);
        LoadVertexMatrix(gl2);
        gl2.glMatrixMode(GL2.GL_MODELVIEW);

        gl2.glPointSize(1);

        gl2.glEnableClientState(GL2.GL_VERTEX_ARRAY);
        gl2.glVertexPointer(3, GL2.GL_SHORT, 0, xyz);
        gl2.glEnableClientState(GL2.GL_TEXTURE_COORD_ARRAY);
        gl2.glTexCoordPointer(3, GL2.GL_SHORT, 0, xyz);

        if (color) {
            gl2.glEnable(GL2.GL_TEXTURE_2D);
        }
        gl2.glBindTexture(GL2.GL_TEXTURE_2D, gl_rgb_tex[0]);
        gl2.glTexImage2D(GL2.GL_TEXTURE_2D, 0, 3, 640, 480, 0, GL2.GL_RGB, GL2.GL_UNSIGNED_BYTE, rgb);

        gl2.glPointSize(2.0f);
        gl2.glDrawElements(GL2.GL_POINTS, 640*480, GL2.GL_UNSIGNED_INT, indices);
        gl2.glPopMatrix();
        gl2.glDisable(GL2.GL_TEXTURE_2D);
    }

    static void keyPressed(int key) {
        if (key == KeyEvent.VK_ESCAPE) {
            freenect_sync_stop();
            frame.dispose();
            System.exit(0);
        }
        if (key == KeyEvent.VK_W) {
            zoom *= 1.1f;
        }
        if (key == KeyEvent.VK_S) {
            zoom /= 1.1f;
        }
        if (key == KeyEvent.VK_C) {
            color = !color;
        }
    }

    static void ReSizeGLScene(GL2 gl2, int Width, int Height) {
        gl2.glViewport(0,0,Width,Height);
        gl2.glMatrixMode(GL2.GL_PROJECTION);
        gl2.glLoadIdentity();
        glu.gluPerspective(60, 4/3., 0.3, 200);
        gl2.glMatrixMode(GL2.GL_MODELVIEW);
    }

    static void InitGL(GL2 gl2, int Width, int Height) {
        gl2.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        gl2.glEnable(GL2.GL_DEPTH_TEST);
        gl2.glGenTextures(1, gl_rgb_tex, 0);
        gl2.glBindTexture(GL2.GL_TEXTURE_2D, gl_rgb_tex[0]);
        gl2.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MIN_FILTER, GL2.GL_LINEAR);
        gl2.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MAG_FILTER, GL2.GL_LINEAR);
        ReSizeGLScene(gl2, Width, Height);
    }

    public static void main(String[] args) {
        Loader.load(freenect.class);

        canvas = new GLCanvas();
        canvas.addGLEventListener(new GLEventListener() {
            @Override public void init(GLAutoDrawable glautodrawable) {
                InitGL(glautodrawable.getGL().getGL2(), glautodrawable.getSurfaceWidth(), glautodrawable.getSurfaceHeight());
            }

            @Override public void display(GLAutoDrawable glautodrawable) {
                DrawGLScene(glautodrawable.getGL().getGL2());
            }

            @Override public void dispose(GLAutoDrawable glautodrawable) {
            }

            @Override public void reshape(GLAutoDrawable glautodrawable, int x, int y, int width, int height) {
                ReSizeGLScene(glautodrawable.getGL().getGL2(), width, height);
            }
        });
        canvas.addKeyListener(new KeyAdapter() {
            @Override public void keyPressed(KeyEvent e) {
                keyPressed(e.getKeyCode());
            }
        });
        canvas.addMouseMotionListener(new MouseMotionAdapter() {
            @Override public void mouseDragged(MouseEvent e) {
                GLPCLView.mouseMoved(e.getX(), e.getY());
            }
        });
        canvas.addMouseListener(new MouseAdapter() {
            @Override public void mousePressed(MouseEvent e) {
                mousePress(e.getButton(), MouseEvent.MOUSE_PRESSED, e.getX(), e.getY());
            }

            @Override public void mouseReleased(MouseEvent e) {
                mousePress(e.getButton(), MouseEvent.MOUSE_RELEASED, e.getX(), e.getY());
            }
        });

        frame = new JFrame("LibFreenect");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().setPreferredSize(new Dimension(640, 480));
        frame.getContentPane().add(canvas);
        frame.pack();
        frame.setVisible(true);

        animator = new FPSAnimator(canvas, 60, true);
        animator.start();
    }
}
```
