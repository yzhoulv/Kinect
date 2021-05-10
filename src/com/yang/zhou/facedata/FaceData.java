package com.yang.zhou.facedata;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.*;

public class FaceData {

    private void read3dm(String facePath) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        File file = new File(facePath);
        int version = 0;
        int featureCount = -1;
        int cx = 0, cy = 0;
        int verCount = 0, traCount = 0;
        List<Point> colorFace = new ArrayList<>();
        List<Triangle> triangle = new ArrayList<>();
        int[][] map2dTo3d;
        List<Integer> map3dTo2d = new ArrayList<>();
        float maxDepth = -1, minDepth = 10000;

        try {
            FileInputStream input = new FileInputStream(file);
            byte[] tmpBytes = new byte[4];
            for (int i=0; i<17; i++) {
                int tmpValue = input.read();
                if (i == 6) {
                    version = tmpValue;
                }
            }
            //System.out.printf("version: VO%d\n", version-48);

            input.read(tmpBytes);
            verCount = byteArray2Int(tmpBytes);
            input.read(tmpBytes);
            traCount = byteArray2Int(tmpBytes);
            //System.out.printf("verCount:%d, traCount:%d\n", verCount, traCount);

            if (version == 50) {
                input.read(tmpBytes);
                featureCount = byteArray2Int(tmpBytes);
                //System.out.printf("feature count: %d\n", featureCount);
            }
            //image size
            input.read(tmpBytes);
            cx = byteArray2Int(tmpBytes);
            input.read(tmpBytes);
            cy = byteArray2Int(tmpBytes);
            //System.out.printf("image size: (%d, %d)\n", cx, cy);
            map2dTo3d = new int[cx][cy];
            for (int i=0; i<verCount; i++) {
                input.read(tmpBytes);
                float x = byteArray2Float(tmpBytes);
                input.read(tmpBytes);
                float y = byteArray2Float(tmpBytes);
                input.read(tmpBytes);
                float z = byteArray2Float(tmpBytes);
                if (z > maxDepth) {
                    maxDepth = z;
                }

                if (z < minDepth) {
                    minDepth = z;
                }
                Point tmpPoint = new Point(x, y, z);
                colorFace.add(tmpPoint);
            }

            for (int i=0; i<verCount; i++) {
                input.read(tmpBytes);
                int r = (int)byteArray2Float(tmpBytes);
                input.read(tmpBytes);
                int g = (int)byteArray2Float(tmpBytes);
                input.read(tmpBytes);
                int b = (int)byteArray2Float(tmpBytes);
                colorFace.get(i).setR(r);
                colorFace.get(i).setG(g);
                colorFace.get(i).setB(b);
            }

            for (int i=0; i<traCount; i++) {
                input.read(tmpBytes);
                float v1 = byteArray2Float(tmpBytes);
                input.read(tmpBytes);
                float v2 = byteArray2Float(tmpBytes);
                input.read(tmpBytes);
                float v3 = byteArray2Float(tmpBytes);
                Triangle tmpTra = new Triangle(v1, v2, v3);
                triangle.add(tmpTra);
            }

            for (int i=0; i<cx; i++) {
                for (int j=0; j<cy; j++) {
                    input.read(tmpBytes);
                    int tmpValue = byteArray2Int(tmpBytes);
                    map2dTo3d[i][j] = tmpValue;
                }
            }


            for (int i=0; i<verCount; i++) {
                input.read(tmpBytes);
                int x = byteArray2Int(tmpBytes);
                input.read(tmpBytes);
                int y = byteArray2Int(tmpBytes);
                map3dTo2d.add(x);
                map3dTo2d.add(y);
            }
            input.close();

//            Mat imgColor = new Mat(cy, cx, CvType.CV_8UC3);
//            Mat imgDepth = new Mat(cy, cx, CvType.CV_8UC1);
//
//            for (int i=0; i<cy; i++) {
//                for (int j=0; j<cx; j++) {
//                    double[] num = {0, 0, 0};
//                    imgColor.put(i, j, num);
//                    imgDepth.put(i,j,0);
//                }
//            }
//            for (int i=0; i<map3dTo2d.size(); i+=2) {
//                int r = colorFace.get(i/2).getR();
//                int g = colorFace.get(i/2).getG();
//                int b = colorFace.get(i/2).getB();
//                float z = colorFace.get(i/2).getZ();
//                //z = (z-minDepth)/(maxDepth-minDepth)*255;
//                double[] num = {b, g, r};
//                imgColor.put(cy - map3dTo2d.get(i+1), map3dTo2d.get(i), num);
//                imgDepth.put(cy - map3dTo2d.get(i+1), map3dTo2d.get(i), (int)z);
//            }
//
//            Imgcodecs.imwrite("rgb.jpg", imgColor);
//            Imgcodecs.imwrite("depth.jpg", imgDepth);

            File wFile = new File(facePath.replace("CurtinFace", "CurtinFace\\new"));
            if (!wFile.exists()) {
                wFile.createNewFile();
            }

            FileOutputStream fos = new FileOutputStream(facePath.replace("CurtinFace", "CurtinFace\\new"));
            for (int i=0; i<17; i++) {
                char tmpValue = 0;
                if (i == 6) {
                    tmpValue = '1';
                }
                if (i==5) {
                    tmpValue = '0';
                }
                if (i==4) {
                    tmpValue = 'V';
                }
                fos.write(oneToByte(tmpValue));
            }
            //System.out.printf("version: VO%d\n", version-48);
            fos.write(intToByteArray(verCount));
            fos.write(intToByteArray(traCount));
            //image size
            fos.write(intToByteArray(cx));
            fos.write(intToByteArray(cy));
            Random random = new Random();
            for (int i=0; i<verCount; i++) {
                Point tmpPoint = colorFace.get(i);
                fos.write(floatToByteArray(tmpPoint.getX()));
                fos.write(floatToByteArray(tmpPoint.getY()));
                float z = tmpPoint.getZ();
                z += (float) (random.nextGaussian()*2);
                fos.write(floatToByteArray(z));
            }
            for (int i=0; i<verCount; i++) {
                Point tmpPoint = colorFace.get(i);
                fos.write(floatToByteArray(tmpPoint.getR()));
                fos.write(floatToByteArray(tmpPoint.getG()));
                fos.write(floatToByteArray(tmpPoint.getB()));
            }

            for (int i=0; i<traCount; i++) {
                Triangle tmpTra = triangle.get(i);
                fos.write(floatToByteArray(tmpTra.getPoint1()));
                fos.write(floatToByteArray(tmpTra.getPoint2()));
                fos.write(floatToByteArray(tmpTra.getPoint3()));
            }

            for (int i=0; i<cx; i++) {
                for (int j=0; j<cy; j++) {
                    fos.write(intToByteArray(map2dTo3d[i][j]));
                }
            }

            for (int i=0; i<verCount; i++) {
                fos.write(intToByteArray(map3dTo2d.get(2*i)));
                fos.write(intToByteArray(map3dTo2d.get(2*i+1)));
            }

            fos.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private byte[] intToByteArray(int i) {
        byte[] result = new byte[4];
        result[3] = (byte)((i>>24));
        result[2] = (byte)((i>>16));
        result[1] = (byte)((i>>8));
        result[0] = (byte)(i);
        return result;
    }

    private byte[] charToByte(char c) {
        byte[] b = new byte[2];
        b[0] = (byte) ((c & 0xFF00) >> 8);
        b[1] = (byte) (c & 0xFF);
        return b;
    }

    private byte[] oneToByte(char c) {
        byte[] b = new byte[1];
        b[0] = (byte) (c & 0xFF);
        return b;
    }

    private int byteArray2Int(byte[] bytes) {
        int value = 0;
        for (int i=0; i<4; i++) {
            int shift = (i) * 8;
            value += (bytes[i] & 0xFF) << shift;
        }
        return value;
    }

    private float byteArray2Float(byte[] bytes) {
        int value = 0;
        for (int i=0; i<4; i++) {
            int shift = (i) * 8;
            value += (bytes[i] & 0xFF) << shift;
        }
        return Float.intBitsToFloat(value);
    }

    private byte[] floatToByteArray(float content) {
        int intbits = Float.floatToIntBits(content);//将float里面的二进制串解释为int整数
        return intToByteArray(intbits);
    }



    public static void main(String[] args) throws IOException {
        FaceData face = new FaceData();
        File f = new File("D:\\data\\CurtinFace\\list_3dm.txt");
        Scanner scan = new Scanner(f);
        int count = 0;
        while (scan.hasNext()) {
            String facePath = scan.next();
            facePath = "D:\\data\\CurtinFace\\"+facePath;
            System.out.println(facePath);
            face.read3dm(facePath);
            count += 1;
            if (count % 10 == 0) {
                System.out.printf("total:936 ----->now:%d\n", count);
            }
        }
            //break;
    }



}
