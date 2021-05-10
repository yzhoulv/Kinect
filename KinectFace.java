package com.yang.zhou.facedata;

import org.opencv.core.*;
import org.opencv.highgui.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.util.*;

public class KinectFace {

    KinectFace() {
        //Imgcodecs.imwrite()
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

    }

//    public List<Point> getFace(String rgbPath, String depthPath, String landmarkPath, int angle) {
//        List<Point> colorFace = new ArrayList<>();
//        List<Triangle> triangles;
//        Map<Integer, Integer> face2Index = new HashMap<>();
//        Mat rgbImage = loadImage(rgbPath, 0);
//        Mat depthImage = loadImage(depthPath, 1);
//        List<Integer> landmark = loadLandmark(landmarkPath);
//        List<Mat> croppedFaces = cropDepthImg(rgbImage, depthImage, landmark);
//        rgbImage = croppedFaces.get(0);
//        depthImage = croppedFaces.get(1);
//        depthImage = skinSmoothing(depthImage);
//        int[][] faceGrid = cropFace(rgbImage, depthImage, face2Index, colorFace, landmark);
//        if (angle != 0) {
//            double[][] matrix = {{Math.cos(Math.toRadians(angle)), 0, -Math.sin(Math.toRadians(angle))}, {0, 1, 0}, {Math.sin(Math.toRadians(angle)), 0, Math.cos(Math.toRadians(angle))}};
//            colorFace = rotateFace(colorFace, matrix);
//        }
//        triangles = calTri(faceGrid, face2Index);
//        return calNormals(colorFace, triangles);
//    }


    public List<Mat> creatNormals(String rgbPath, String depthPath, String landmarkPath, boolean argment) {
        List<Mat> res = new ArrayList<>();
        List<Triangle> triangles;
        List<Point> colorFace = new ArrayList<>();
        Map<Integer, Integer> face2Index = new HashMap<>();
        Mat rgbImage = loadImage(rgbPath, 0);
        Mat depthImage = loadImage(depthPath, 1);
        List<Integer> landmark = loadLandmark(landmarkPath);
        //depthImage = skinSmoothing(depthImage);
        noseTipCorrect(depthImage, landmark);
        int[][] faceGrid = cropFace(rgbImage, depthImage, face2Index, colorFace, landmark);
        triangles = calTri(faceGrid, face2Index);
        List<Point> normMap = calNormals(colorFace, triangles);
        Mat normImg = savaNorms(colorFace, normMap, faceGrid, face2Index);
        normImg = holeFilling(normImg);
        Imgproc.resize(normImg, normImg, new Size(128, 128));
        res.add(normImg);
        if (argment) {
            double[][][] matrix = {
                    {{Math.cos(Math.toRadians(75)), 0, -Math.sin(Math.toRadians(75))}, {0, 1, 0}, {Math.sin(Math.toRadians(75)), 0, Math.cos(Math.toRadians(75))}},
                    {{Math.cos(Math.toRadians(-75)), 0, -Math.sin(Math.toRadians(-75))}, {0, 1, 0}, {Math.sin(Math.toRadians(-75)), 0, Math.cos(Math.toRadians(-75))}},
                    {{Math.cos(Math.toRadians(60)), 0, -Math.sin(Math.toRadians(60))}, {0, 1, 0}, {Math.sin(Math.toRadians(60)), 0, Math.cos(Math.toRadians(60))}},
                    {{Math.cos(Math.toRadians(-60)), 0, -Math.sin(Math.toRadians(-60))}, {0, 1, 0}, {Math.sin(Math.toRadians(-60)), 0, Math.cos(Math.toRadians(-60))}},
                    {{Math.cos(Math.toRadians(45)), 0, -Math.sin(Math.toRadians(45))}, {0, 1, 0}, {Math.sin(Math.toRadians(45)), 0, Math.cos(Math.toRadians(45))}},
                    {{Math.cos(Math.toRadians(-45)), 0, -Math.sin(Math.toRadians(-45))}, {0, 1, 0}, {Math.sin(Math.toRadians(-45)), 0, Math.cos(Math.toRadians(-45))}},
                    {{Math.cos(Math.toRadians(30)), 0, -Math.sin(Math.toRadians(30))}, {0, 1, 0}, {Math.sin(Math.toRadians(30)), 0, Math.cos(Math.toRadians(30))}},
                    {{Math.cos(Math.toRadians(-30)), 0, -Math.sin(Math.toRadians(-30))}, {0, 1, 0}, {Math.sin(Math.toRadians(-30)), 0, Math.cos(Math.toRadians(-30))}},
                    {{Math.cos(Math.toRadians(15)), 0, -Math.sin(Math.toRadians(15))}, {0, 1, 0}, {Math.sin(Math.toRadians(15)), 0, Math.cos(Math.toRadians(15))}},
                    {{Math.cos(Math.toRadians(-15)), 0, -Math.sin(Math.toRadians(-15))}, {0, 1, 0}, {Math.sin(Math.toRadians(-15)), 0, Math.cos(Math.toRadians(-15))}},
                    {{1, 0, 0}, {0, Math.cos(Math.toRadians(15)), Math.sin(Math.toRadians(15))}, {0, -Math.sin(Math.toRadians(15)), Math.cos(Math.toRadians(15))}},
                    {{1, 0, 0}, {0, Math.cos(Math.toRadians(-15)), Math.sin(Math.toRadians(-15))}, {0, -Math.sin(Math.toRadians(-15)), Math.cos(Math.toRadians(-15))}},
                    {{1, 0, 0}, {0, Math.cos(Math.toRadians(30)), Math.sin(Math.toRadians(30))}, {0, -Math.sin(Math.toRadians(30)), Math.cos(Math.toRadians(30))}},
                    {{1, 0, 0}, {0, Math.cos(Math.toRadians(-30)), Math.sin(Math.toRadians(-30))}, {0, -Math.sin(Math.toRadians(-30)), Math.cos(Math.toRadians(-30))}},
            };
            for (int i = 0; i < 14; i++) {
                List<Point> newFace = rotateFace(colorFace, matrix[i]);
                normMap = calNormals(newFace, triangles);
                normImg = savaNorms(newFace, normMap, faceGrid, face2Index);
                normImg = holeFilling(normImg);
                Imgproc.resize(normImg, normImg, new Size(128, 128));
                res.add(normImg);
            }
        }
        return res;
    }

    private void noseTipCorrect(Mat image, List<Integer> lm) {
        int noseX = lm.get(4);
        int noseY = lm.get(5);
        int dstX=noseX, dstY=noseY;
        double depth = image.get(noseY, noseX)[0];
        int count = 3;
        double minDepth = depth;
        //有问题则调整，否则不变
        while ((depth == 0 || minDepth > 1500) && count < 40) {
//            System.out.println("####");
//            System.out.println(depth);
            if (minDepth == 0) {
                minDepth = 10000;
            }
            for (int i=noseX-count; i<noseX+count; i++) {
                for (int j=noseY-count; j<noseY+count; j++) {
                    try{
                        depth = image.get(j, i)[0];
                        if (depth < minDepth && depth > 0) {
                            minDepth = depth;
                            dstX = i;
                            dstY = j;
                        }
                    } catch (Exception e) {
                        //e.printStackTrace();
                        break;
                    }
                }
            }
            count ++;
        }
        lm.set(4, dstX);
        lm.set(5, dstY);
        //System.out.println(image.get(dstY, dstX)[0]);
    }

    private Mat loadImage(String fileName, int mod) {
        if (mod == 0) {
            return Imgcodecs.imread(fileName);
        } else {
            return Imgcodecs.imread(fileName, Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);
        }
    }

    private List<Integer> loadLandmark(String fileName) {
        List<Integer> res = new ArrayList<Integer>();
        try {
            File file = new File(fileName);
            BufferedReader br = new BufferedReader(new FileReader(file));
            String s;
            while ((s = br.readLine()) != null) {
                res.add(Integer.valueOf(s.split(" ")[0].split(":")[1]));
                res.add(Integer.valueOf(s.split(" ")[1].split(":")[1]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return res;
    }

    private List<ArrayList<Integer>> calLandmark() {
        return null;
    }


    private Mat imageAlign(Mat bgrImg, int offsetX, int offsetY) {
        Mat res = new Mat(609, 735, bgrImg.type());
        Mat tmpOrigin = new Mat(bgrImg.rows() / 2, bgrImg.cols() / 2, bgrImg.type());

        Imgproc.resize(bgrImg, tmpOrigin, new Size(bgrImg.cols(), bgrImg.rows()));

        for (int i = 0; i < 609; i++) {
            for (int j = 0; j < 735; j++) {
                if (i <= offsetX || i >= 540 + offsetX) {
                    res.put(i, j, new double[3]);
                } else {
                    //System.out.printf("(%d, %d)%n", i, j);
                    res.put(i, j, tmpOrigin.get(i - offsetX, j - offsetY));
                }
            }
        }
        Imgproc.resize(res, res, new Size(512, 424));

        return res;
    }

    private double getMedianLocus(Mat img, int x, int y, int size) {

        double[] a = new double[size * size];
        int m, n;
        for (int i = 0; i < size; i++) {

            for (int j = 0; j < size; j++) {
                m = Math.max(x+i-1, 0);
                m = Math.min(x+i-1, img.rows()-1);

                n = Math.max(y+j-1, 0);
                n = Math.min(y+j-1, img.cols()-1);
                a[i * size + j] = img.get(m, n)[0];
            }
        }
        Arrays.sort(a);
        return a[size * size / 2];
    }

    private Mat skinSmoothing(Mat image) {

        Mat dstImg = new Mat(image.rows(), image.cols(), image.type());
        //Imgproc.GaussianBlur(image, image, new Size(3, 3), 0);
        //Imgproc.blur(image, dstImg, new Size(3,3));
        Imgproc.medianBlur(image, dstImg, 3);
        //Imgproc.bilateralFilter(image, dstImg, 4, 16, 8);
//        double threshold = 5000;
//        for (int i=1; i<image.rows()-1; i++) {
//            for (int j=1; j<image.cols()-1; j++) {
//                double depth = image.get(i, j)[0];
//                double totalLength = 0;
//                for (int k=0; k<3; k++) {
//                    totalLength += Math.abs(depth - image.get(i-1+k, j-1)[0]);
//                    totalLength += Math.abs(depth - image.get(i-1+k, j)[0]);
//                    totalLength += Math.abs(depth - image.get(i-1+k, j+1)[0]);
//                }
//                //System.out.println(totalLength);
//                if(totalLength > threshold) {
//                    double value = getMedianLocus(image, i, j, 3);
//                    image.put(i, j, value);
//                }
//            }
//        }
        return dstImg;
    }

    private List<Mat> cropDepthImg(Mat rgbImg, Mat depthImg, List<Integer> lm) {
        List<Mat> res = new ArrayList<>();
        int noseX = lm.get(4);
        int noseY = lm.get(5);
        Mat rgbImage = new Mat(180, 180, rgbImg.type());
        Mat depthImage = new Mat(180, 180, depthImg.type());
        Mat rgbRoi = new Mat(rgbImg, new Rect(noseX - 90, noseY - 90, 180, 180));
        rgbRoi.copyTo(rgbImage);
        Mat depthRoi = new Mat(depthImg, new Rect(noseX - 90, noseY - 90, 180, 180));
        depthRoi.copyTo(depthImage);

        res.add(rgbImage);
        res.add(depthImage);

        for (int i = 0; i < lm.size(); i += 2) {
            int x = 90 - (noseX - lm.get(i));
            int y = 90 - (noseY - lm.get(i + 1));
            lm.set(i, x);
            lm.set(i + 1, y);
        }
        return res;
    }


    private int[][] cropFace(Mat rgbRoi, Mat depthRoi, Map<Integer, Integer> face2Index, List<Point> colorFace, List<Integer> lm) {
        Random random = new Random();
//        float fx = 525.0f;
//        float fy = 525.0f;
//        float cx = 319.5f;
//        float cy = 239.5f;
//        float fx = 518.0f / 1.437f;
//        float fy = 519.0f / 1.437f;
//        float cx = 325.5f / 1.437f;
//        float cy = 253.5f / 1.437f;
        float fx = 363.767f;
        float fy = 364.767f;
        float cx = 258.087f;
        float cy = 206.007f;
        int[][] faceGrid = new int[180][180];
        int noseY = lm.get(4);
        int noseX = lm.get(5);

        //double noseDepth = depthRoi.get(noseX-90, noseY-90)[0] - (depthRoi.get(noseX-90, noseY-90)[0] - depthRoi.get(noseX, noseY)[0]) / 2.0;
        //System.out.println(noseDepth);
        double noseDepth = depthRoi.get(noseX, noseY)[0]+180;
//        System.out.println(noseX);
//        System.out.println(noseY);
        int count = 0;
        for (int i = noseX-90; i < noseX+90; i++) {
            for (int j = noseY-90; j < noseY+90; j++) {
                // System.out.printf("%d %d\n", i, j);
                 if (depthRoi.get(i, j)[0] < noseDepth && depthRoi.get(i, j)[0] > 0) {
                    float z = (float) depthRoi.get(i, j)[0];
                    //z += (float) (random.nextGaussian()*random.nextInt(5));
                    faceGrid[i+90-noseX][j+90-noseY] = -1;
                    int b = (int) rgbRoi.get(i, j)[0];
                    int g = (int) rgbRoi.get(i, j)[1];
                    int r = (int) rgbRoi.get(i, j)[2];

                    float xw = (j - cx) * z / fx;
                    float yw = -(i - cy) * z / fy;
                    //z = z/1000f;
                    colorFace.add(new Point(xw, yw, z, r, g, b));
                    int tmpNum = count;
                    face2Index.put((i+90-noseX) * 180 + (j+90-noseY), tmpNum);
                    count += 1;
                }
            }
        }
        //System.out.println(depthRoi.get(noseX, noseY)[0]);
        int noseIndex = face2Index.get(90*180+90);
        float nx = colorFace.get(noseIndex).getX();
        float ny = colorFace.get(noseIndex).getY();
        float nz = colorFace.get(noseIndex).getZ();

        // 归一化
        for (Point tmpPoint : colorFace) {
            float x = tmpPoint.getX()-nx;
            float y = tmpPoint.getY()-ny;
            float z = tmpPoint.getZ()-nz;

            tmpPoint.setX(x);
            tmpPoint.setY(y);
            tmpPoint.setZ(-z);
        }
        return faceGrid;
    }


    private List<ArrayList<Integer>> postCorrect() {
        return null;
    }

    private List<Triangle> calTri(int[][] faceGrid, Map<Integer, Integer> face2Index) {
        List<Triangle> tris = new ArrayList<>();
        for (int i = 1; i < 180; i++) {
            for (int j = 0; j < 180 - 1; j++) {
                if (faceGrid[i][j] == -1 && faceGrid[i - 1][j + 1] == -1) {
                    if (faceGrid[i - 1][j] == -1) {
                        tris.add(new Triangle(face2Index.get(i * 180 + j), face2Index.get((i - 1) * 180 + j + 1), face2Index.get((i - 1) * 180 + j)));
                    }
                    if (faceGrid[i][j + 1] == -1) {
                        tris.add(new Triangle(face2Index.get(i * 180 + j), face2Index.get((i - 1) * 180 + j + 1), face2Index.get(i * 180 + j + 1)));
                    }
                }
            }
        }
        return tris;
    }

    private List<Point> calNormals(List<Point> face, List<Triangle> triangles) {
        //计算每个三角片法向量
        float p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z;
        float p1p2x, p2p3x, p1p2y, p2p3y, p1p2z, p2p3z;
        float a, b, c;
        float p1, p2, p3;

        List<Triangle> tmpTris = new ArrayList<>(triangles.size());
        List<Point> normMap = new ArrayList<>(face.size());

        for (Triangle triangle : triangles) {
            p1 = triangle.getPoint1();
            p2 = triangle.getPoint2();
            p3 = triangle.getPoint3();

            p1x = face.get((int) p1).getX();
            p1y = face.get((int) p1).getY();
            p1z = face.get((int) p1).getZ();

            p2x = face.get((int) p2).getX();
            p2y = face.get((int) p2).getY();
            p2z = face.get((int) p2).getZ();

            p3x = face.get((int) p3).getX();
            p3y = face.get((int) p3).getY();
            p3z = face.get((int) p3).getZ();

            p1p2x = p1x - p2x;
            p2p3x = p2x - p3x;
            p1p2y = p1y - p2y;
            p2p3y = p2y - p3y;
            p1p2z = p1z - p2z;
            p2p3z = p2z - p3z;

            //向量叉乘
            a = p1p2y * p2p3z - p2p3y * p1p2z;
            b = p1p2z * p2p3x - p2p3z * p1p2x;
            c = p1p2x * p2p3y - p2p3x * p1p2y;

            if (c < 0) {
                c = -c;
                a = -a;
                b = -b;
            }

            //System.out.printf("%f %f %f %n", a, b, c);

            float num = (float) Math.sqrt(a * a + b * b + c * c);

            Triangle tmpTri = new Triangle(a / num, b / num, c / num);
            tmpTris.add(tmpTri);

        }

        //初始化normalMap
        for (int i = 0; i < face.size(); i++) {
            Point p = new Point(0f, 0f, 0f);
            normMap.add(p);
        }

        //归一化每个点的法向量
        for (int i = 0; i < tmpTris.size(); i++) {

            int pp1 = (int) triangles.get(i).getPoint1();
            setNorm(tmpTris, normMap, i, pp1);

            int pp2 = (int) triangles.get(i).getPoint2();
            setNorm(tmpTris, normMap, i, pp2);

            int pp3 = (int) triangles.get(i).getPoint3();
            setNorm(tmpTris, normMap, i, pp3);

        }

        //转化成每个点的向量数据
        for (int i = 0; i < normMap.size(); i++) {
            a = normMap.get(i).getX();
            b = normMap.get(i).getY();
            c = normMap.get(i).getZ();

            float tmpNum = (float) Math.sqrt(a * a + b * b + c * c);
            int aa = (int) (((a / tmpNum) + 1) * 127.5);
            int bb = (int) (((b / tmpNum) + 1) * 127.5);
            int cc = (int) (((c / tmpNum) + 1) * 127.5);

            //System.out.printf("%d %d %d\n", aa, bb, cc);
            Point tmpPoint = new Point(face.get(i).getX(), face.get(i).getY(), face.get(i).getZ(), aa, bb, cc);
            normMap.set(i, tmpPoint);
        }
        return normMap;
    }

    private void setNorm(List<Triangle> tmpTris, List<Point> normMap, int i, int p) {
        float x = normMap.get(p).getX();
        float y = normMap.get(p).getY();
        float z = normMap.get(p).getZ();
        normMap.get(p).setX(tmpTris.get(i).getPoint1() + x);
        normMap.get(p).setY(tmpTris.get(i).getPoint2() + y);
        normMap.get(p).setZ(tmpTris.get(i).getPoint3() + z);
    }

    private Mat savaNorms(List<Point> face, List<Point> normMap, int[][] faceGrid, Map<Integer, Integer> face2Index) {
        Mat normImg = new Mat(500, 500, CvType.CV_8UC3);
        Mat depthImg = new Mat(500, 500, CvType.CV_32S);
        for (int i = 0; i < 500; i++) {
            for (int j = 0; j < 500; j++) {
                double[] num = {0, 0, 0};
                normImg.put(i, j, num);
                depthImg.put(i, j, 0);
            }
        }

        for (int i = 0; i < 180; i++) {
            for (int j = 0; j < 180; j++) {
                if (faceGrid[i][j] == -1) {
                    int index = face2Index.get(i * 180 + j);
                    double[] tmp = {normMap.get(index).getR(), normMap.get(index).getG(), normMap.get(index).getB()};
                    int y = Math.round(face.get(index).getX() / 2.8f) + 250;
                    int x = -Math.round(face.get(index).getY() / 2.8f) + 250;
                    int depth = (int) face.get(index).getZ();
                    x = Math.max(x, 0);
                    x = Math.min(x, 499);
                    y = Math.max(y, 0);
                    y = Math.min(y, 499);
                    //System.out.printf("%d %d\n", x, y);
                    if (normImg.get(x, y)[0] != 0) {
                        if (depth > depthImg.get(x, y)[0]) {
                            normImg.put(x, y, tmp);
                        }
                    } else {
                        normImg.put(x, y, tmp);
                        depthImg.put(x, y, depth);
                    }
                }
            }
        }
        return cropNorm(normImg);
    }

    private List<Point> rotateFace(List<Point> face, double[][] matrix) {
        List<Point> newFace = new ArrayList<>();
        for (Point p : face) {
            float x = p.getX();
            float y = p.getY();
            float z = p.getZ();
            int r = p.getR();
            int g = p.getG();
            int b = p.getB();

            float newX = (float) (x * matrix[0][0] + y * matrix[1][0] + z * matrix[2][0]);
            float newY = (float) (x * matrix[0][1] + y * matrix[1][1] + z * matrix[2][1]);
            float newZ = (float) (x * matrix[0][2] + y * matrix[1][2] + z * matrix[2][2]);

            Point tmpPoint = new Point(newX, newY, newZ, r, g, b);
            newFace.add(tmpPoint);
        }
        return newFace;
    }

    private Mat cropNorm(Mat normImg) {
        return new Mat(normImg, new Rect((250 - 35), (250 - 45), 70, 70));
        //return normImg;
    }


    private Mat holeFilling(Mat image) {
        List<Mat> mv = new ArrayList<>();
        Core.split(image, mv);
        Mat b, g, r;

        if (mv.size() == 3) {
            b = mv.get(0);
            g = mv.get(1);
            r = mv.get(2);
        } else {
            b = mv.get(0);
            g = mv.get(0);
            r = mv.get(0);
        }

        Mat threshold = new Mat(image.rows(), image.cols(), CvType.CV_8UC1);
        Imgproc.threshold(r, threshold, 10, 1, Imgproc.THRESH_BINARY);
        for (int i=0; i<threshold.rows(); i++) {
            for (int j=0; j<threshold.cols(); j++) {
                if (threshold.get(i,j)[0] == 1) {
                    threshold.put(i,j,200);
                }
            }
        }

        List<Integer> markedBg = markBackground(threshold);

        for (int i = 0; i < threshold.height(); i++) {
            for (int j = 0; j < threshold.width(); j++) {
                if (threshold.get(i, j)[0] == 0 && j >= markedBg.get(2 * i) && j <= markedBg.get(2 * i + 1)) {
                    double avgValueB = getAvgFill(b, i, j);
                    b.put(i, j, avgValueB);
                    double avgValueG = getAvgFill(g, i, j);
                    g.put(i, j, avgValueG);
                    double avgValueR = getAvgFill(r, i, j);
                    r.put(i, j, avgValueR);
                }
            }
        }


        Mat dest = new Mat();
        mv.set(0, b);
        mv.set(1, g);
        mv.set(2, r);
        Core.merge(mv, dest);
        return dest;
    }

    private double getAvgFill(Mat image, int x, int y) {
        double res = 0.0;
        int circle = 1;
        while (res == 0 && circle < 5) {
            int startX = Math.max(x - circle, 0);
            int startY = Math.max(y - circle, 0);
            int endX = Math.min(x + circle, image.rows() - 1);
            int endY = Math.min(y + circle, image.cols() - 1);
            double sumCircle = 0.0;
            int countCircle = 0;
            for (int i = startX; i <= endX; i++) {
                for (int j = startY; j <= endY; j++) {
                    if (image.get(i, j)[0] > 0) {
                        countCircle += 1;
                        sumCircle += image.get(i, j)[0];
                    }
                }
            }
            res = sumCircle / countCircle;
            circle += 1;
        }
        return res;
    }

    private List<Integer> markBackground(Mat image) {
        List<Integer> markedBg = new ArrayList<Integer>();
        int startIndex = image.cols() - 1;
        int endIndex = 0;
        for (int i = 0; i < image.rows(); i++) {
            for (int j = 0; j < image.cols(); j++) {
                if (image.get(i, j)[0] != 0) {
                    if (startIndex == image.cols() - 1) {
                        startIndex = j;
                    }
                    endIndex = j;
                }
            }
            int tmpStart = startIndex;
            int tmpEnd = endIndex;
            markedBg.add(tmpStart);
            markedBg.add(tmpEnd);
            startIndex = image.cols() - 1;
            endIndex = 0;
        }
        return markedBg;
    }

/*
    public static void main(String[] args) throws FileNotFoundException {
        File f = new File("F:\\Lock3DFace\\color_depth_landmark.txt");
        Scanner scan = new Scanner(f);

        String colorRoot = "F:\\Lock3DFace\\color\\";
        String depthRoot = "F:\\Lock3DFace\\depth\\";
        String lmRoot = "F:\\Lock3DFace\\landmark\\infrared\\";
        String resRoot = "D:\\data\\project\\java\\Kinect\\Lock3d\\sdd\\";
        KinectFace kf = new KinectFace();
        int count = 0;
        while (scan.hasNext()) {
            String rgbPath = scan.next();
            String depthPath = scan.next();
            String landmark = scan.next();

            String resPath = resRoot + rgbPath.split("\\\\")[0].replace("COLOR", "");
            File folder = new File(resPath);
            if (!folder.exists()) {
                if (folder.mkdirs()) {
                    System.out.println(resPath);
                } else {
                    System.err.println(resPath);
                }

            }

            String imageName = rgbPath.split("\\\\")[1].replace(".jpg", "");
            if(! new File(resPath + "\\59.jpg").exists()){

                if (rgbPath.contains("_NU_1") && Integer.parseInt(imageName) < 7) {
                    List<Mat> res = kf.creatNormals(colorRoot+rgbPath, depthRoot + depthPath, lmRoot + landmark, true);
                    for (int i = 0; i < res.size(); i++) {
                        Imgcodecs.imwrite(resPath + "\\" + imageName + "_" + String.valueOf(i) + ".jpg", res.get(i));
                    }
                } else {
                    List<Mat> res = kf.creatNormals(colorRoot+rgbPath, depthRoot+depthPath, lmRoot+landmark, false);
                    Imgcodecs.imwrite(resPath + "\\" + imageName + ".jpg", res.get(0));
                }
                //System.out.println(resPath);
                count += 1;
                if (count % 1000 == 0) {
                    System.out.printf("total:330518 ----->now:%d\n", count);
                }
            }

            //break;
        }

    }
*/

}
