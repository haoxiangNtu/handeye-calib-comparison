# 手眼标定方法对比

对比两种 eye-to-hand 手眼标定方法的原理：

- **方法 A**：末端贴 AprilTag，单目 PnP 得到 Tag 位姿，求解 `AX = YB`
- **方法 B**：末端激光测距仪打出光斑，双目三角化后与正运动学换算的点做 3D–3D 刚体配准

在线页面：https://haoxiangntu.github.io/handeye-calib-comparison/
