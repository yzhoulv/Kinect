package com.yang.zhou.facedata;

public class Triangle {
    private float point1;
    private float point2;
    private float point3;

    public Triangle(float p1, float p2, float p3) {
        this.point1 = p1;
        this.point2 = p2;
        this.point3 = p3;
    }

    public float getPoint1() {
        return point1;
    }

    public void setPoint1(float point1) {
        this.point1 = point1;
    }

    public float getPoint2() {
        return point2;
    }

    public void setPoint2(float point2) {
        this.point2 = point2;
    }

    public float getPoint3() {
        return point3;
    }

    public void setPoint3(float point3) {
        this.point3 = point3;
    }
}
