package com.yang.zhou.facedata;

import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLCapabilities;
import com.jogamp.opengl.GLEventListener;
import com.jogamp.opengl.GLProfile;
import com.jogamp.opengl.awt.GLCanvas;
import com.jogamp.opengl.GL2;
import com.jogamp.opengl.glu.GLU;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


import javax.swing.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Main implements GLEventListener {
    private GLU glu = new GLU();

    @Override
    public void init(GLAutoDrawable glAutoDrawable) {

    }

    @Override
    public void dispose(GLAutoDrawable glAutoDrawable) {

    }

    @Override
    public void display(GLAutoDrawable glAutoDrawable) {

        String rgbPath = "./3.jpg";
        String depthPath = "./4.jpg";
        WGY wgy = new WGY();

        List<Point> faceData = wgy.getFace(rgbPath, depthPath, 0);
        Iterator<Point> iter = faceData.iterator();

        final GL2 gl = glAutoDrawable.getGL().getGL2();
        //gl.glTranslatef(0f, 0f, -3.5f);
        while (iter.hasNext()) {
            gl.glBegin(GL2.GL_POINTS);
            Point tmpPoint = iter.next();
            gl.glVertex3f(tmpPoint.getX() / 100, tmpPoint.getY() / 100, tmpPoint.getZ() / 255f);
            gl.glColor3f(tmpPoint.getR() / 255f, tmpPoint.getR() / 255f, tmpPoint.getR() / 255f);
            //System.out.println(tmpPoint.get(2));
            //gl.glColor3f(tmpPoint.get(2) / 255.0f, tmpPoint.get(2) / 255.0f, tmpPoint.get(2) / 255.0f);
            gl.glEnd();
        }

        gl.glFlush();

    }

    @Override
    public void reshape(GLAutoDrawable glAutoDrawable, int i, int i1, int i2, int i3) {

//        final GL2 gl = glAutoDrawable.getGL().getGL2();
//        int height = i3;
//        int width = i2;
//        if(height <= 0)
//            height = 1;
//        final float h = ( float ) width / ( float ) height;
//        gl.glViewport( 0, 0, width, height );
//        gl.glMatrixMode( GL2.GL_PROJECTION );
//        gl.glLoadIdentity();
//        glu.gluPerspective( 45.0f, h, 1.0, 20.0 );
//        gl.glMatrixMode( GL2.GL_MODELVIEW );
//        gl.glLoadIdentity();


    }
/*
    public static void main(String[] args) {

    //getting the capabilities object of GL2 profile
//        final GLProfile profile = GLProfile.get( GLProfile.GL2 );
//        GLCapabilities capabilities = new GLCapabilities( profile );
//        // The canvas
//        final GLCanvas glcanvas = new GLCanvas( capabilities );
//        Main line3d = new Main();
//        glcanvas.addGLEventListener( line3d );
//        glcanvas.setSize( 400, 400 );
//        //creating frame
//        final JFrame frame = new JFrame (" 3d line");
//        //adding canvas to it
//        frame.getContentPane().add( glcanvas );
//        frame.setSize( frame.getContentPane().getPreferredSize() );
//        frame.setVisible( true );

        String rgbPath = "./01.jpg";
        String depthPath = "./01.png";
        KinectFace wgy = new KinectFace();
        List<Mat> res = wgy.creatNormals(rgbPath, depthPath,"438_Kinect_FE_1INFRARED.txt", true);
        for (int i=0; i<res.size(); i++) {
            Imgcodecs.imwrite(String.valueOf(i)+".jpg", res.get(i));
        }
    }
    */

}