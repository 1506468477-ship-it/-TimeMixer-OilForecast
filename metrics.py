import numpy as np


def metric(pred, true):
    """
    计算预测指标
    pred: 预测值，形状 [samples, pred_len, 1]
    true: 真实值，形状 [samples, pred_len, 1]
    """
    # 展平为1维数组
    pred = pred.reshape(-1)
    true = true.reshape(-1)

    # 防止除零错误
    mask = true != 0
    pred = pred[mask]
    true = true[mask]

    # 计算各项指标
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / true)) * 100
    mspe = np.mean(np.square((pred - true) / true)) * 100

    return mae, mse, rmse, mape, mspe