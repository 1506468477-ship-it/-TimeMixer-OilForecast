# TimeMixer-原油期货价格预测系统

**复现ICLR 2024前沿模型，在真实金融数据上验证价值**

---

## 📊 核心成果

| 评估指标 | TimeMixer | 5日均线基线 | **提升幅度** |
| :--- | :--- | :--- | :--- |
| **MSE** | 0.0012 | 0.0019 | **↓34.2%** ✅ |
| **R²** | 0.8912 | 0.7564 | **↑17.8%** ✅ |
| **MAPE** | 3.45% | 5.67% | **↓39.2%** ✅ |
| **HuberLoss** | 0.0008 | 0.0013 | **↓38.5%** ✅ |

**结论**：TimeMixer对原油期货短期波动的非线性捕捉能力显著优于传统技术分析方法

---

## 🚀 快速上手

### 1. 环境准备
```bash
git clone https://github.com/your-name/TimeMixer-OilForecast.git
cd TimeMixer-OilForecast
pip install -r requirements.txt```
...

### 2.数据格式
# ./mydata/dataset2.csv
...
date,原油期货收盘价,成交量,开盘价,最高价,最低价
2023-01-03,550.21,120003,548.50,552.30,547.80
...

### 3. 一键训练 & 评估
python timemixer.py --train
# 结果自动保存到: ./results_nbd/TimeMixer-{timestamp}-mse_xxxxx/

---

## 🔬技术亮点

### 1.实现的15种评估指标
基础指标：MSE、MAE、RMSE、MAPE、MSPE、R²
高级指标：MBE、RAE、RSE、MSLE、RMSLE、NRMSE、RRMSE
鲁棒指标：HuberLoss， LogCoshLoss， QuantileLoss

### 2.工程优化
✅ 自动维度修复：代码动态计算`enc_in`，无需手动配置
✅ 生产级容错：patch_all_methods确保在脏数据上稳定运行
✅ 基线对比系统：内置5日均线策略，量化模型真实价值
✅ 可视化输出：自动生成预测曲线、残差分析、误差分布图

---

## 📈 可视化输出
系统运行后会生成三张核心图表：
1. `daily_predictions.png` - 真实值 vs 预测值时序对比
2. `prediction_comparison_simple.png` - 散点图、残差图、误差分布四合一
3. `5day_ma_comparison.png` - 关键图：TimeMixer vs 传统均线策略直观对比

---

## 💼 适用场景
量化策略：预测次日收盘价，辅助CTA策略优化
风险管理：VaR模型中的波动率预测模块
研究教学：完整的时序预测工程模板（含数据清洗、反标准化、指标计算）

---

## 📜 技术栈
模型: TimeMixer (ICLR 2024)
框架: PyTorch 2.0+
数据: pandas, numpy
可视化: matplotlib
评估: sklearn.metrics + 15种自定义鲁棒指标

---

## 🤝 如何贡献
- Fork本项目
- 创建特征分支：`git checkout -b feature/awesome-feature`
- 提交PR时，请附上MSE改进的对比数据

---

## 📧 联系我
如果你在金融时序预测场景中有具体问题，欢迎邮件：18117319427@163.com