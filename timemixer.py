import numpy as np
import random
import sys
import os
import types
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import datetime
import shutil
import torch

# 添加项目路径到系统路径
sys.path.append('./TimeMixer')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
fix_seed = 20050824
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def create_results_folder(mse_value, model_name="TimeMixer"):
    """创建带有时间戳和 MSE 值的结果文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mse_str = f"{mse_value:.8f}".replace('.', '')[:8]  # 使用8位MSE值便于文件夹命名
    folder_name = f"{model_name}-{timestamp}-mse_{mse_str}"

    # 创建结果目录
    results_base_dir = './results_nbd'
    if not os.path.exists(results_base_dir):
        os.makedirs(results_base_dir)

    results_dir = os.path.join(results_base_dir, folder_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"创建结果文件夹: {results_dir}")
    return results_dir


def calculate_5day_moving_average(values, dates):
    """计算5日均值预测作为基线"""
    if len(values) < 5:
        print("警告：数据量不足5天，使用简单均值作为基线")
        return np.full_like(values, np.mean(values))

    # 使用pandas的rolling计算5日均值
    df = pd.DataFrame({'value': values.flatten(), 'date': dates})
    ma5 = df['value'].rolling(window=5, min_periods=1).mean().values

    # 将5日均值向前移动一位作为预测（即用过去5天均值预测下一天）
    ma5_pred = np.roll(ma5, 1)
    ma5_pred[0] = ma5[0]  # 第一天用自身值

    return ma5_pred


def plot_5day_ma_comparison(dates, true_values, timemixer_preds, ma5_preds, results_dir):
    """绘制5日均值预测与TimeMixer的对比图"""
    plt.figure(figsize=(16, 10))

    # 主图：三种曲线对比
    plt.subplot(2, 2, 1)
    plt.plot(dates, true_values, 'b-', label='真实值', linewidth=2, alpha=0.8)
    plt.plot(dates, timemixer_preds, 'r-', label='TimeMixer预测', linewidth=2, alpha=0.8)
    plt.plot(dates, ma5_preds, 'g--', label='5日均值预测', linewidth=2, alpha=0.8)
    plt.title('预测结果对比: TimeMixer vs 5日均值', fontsize=14, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('原油期货收盘价')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # 子图2: 近期放大图（最后60天）
    plt.subplot(2, 2, 2)
    recent_days = min(60, len(dates))
    plt.plot(dates[-recent_days:], true_values[-recent_days:], 'b-', label='真实值', linewidth=2)
    plt.plot(dates[-recent_days:], timemixer_preds[-recent_days:], 'r-', label='TimeMixer', linewidth=2)
    plt.plot(dates[-recent_days:], ma5_preds[-recent_days:], 'g--', label='5日均值', linewidth=2)
    plt.title(f'近期{recent_days}天放大图', fontsize=14)
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # 子图3: 残差对比
    plt.subplot(2, 2, 3)
    timemixer_residuals = true_values - timemixer_preds
    ma5_residuals = true_values - ma5_preds
    plt.plot(dates, timemixer_residuals, 'r-', label='TimeMixer残差', alpha=0.7)
    plt.plot(dates, ma5_residuals, 'g--', label='5日均值残差', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    plt.title('残差对比', fontsize=14)
    plt.xlabel('日期')
    plt.ylabel('残差')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # 子图4: 误差分布对比
    plt.subplot(2, 2, 4)
    plt.hist(timemixer_residuals, bins=30, alpha=0.6, color='red', label='TimeMixer误差', density=True)
    plt.hist(ma5_residuals, bins=30, alpha=0.6, color='green', label='5日均值误差', density=True)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
    plt.title('误差分布对比', fontsize=14)
    plt.xlabel('误差值')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/5day_ma_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"5日均值对比图已保存到 {results_dir}/5day_ma_comparison.png")


def calculate_input_dims(args):
    """根据实际数据动态计算输入维度"""
    try:
        data_file = os.path.join(args.root_path, args.data_path)
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            print(f"DataFrame shape: {df.shape}")
            print(f"Data columns: {list(df.columns)}")

            # 排除目标列和日期列后的特征数量
            date_columns = ['date', 'Date', 'DATE', 'time', 'date']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break

            # 特征列：排除目标列和日期列
            feature_columns = [col for col in df.columns
                               if col != args.target and col != date_col]

            # 如果特征列为空，使用除目标列和日期列外的所有列
            if not feature_columns:
                feature_columns = [col for col in df.columns
                                   if col != args.target and col not in date_columns]

            actual_enc_in = len(feature_columns)

            print(f"Calculated input dimension: {actual_enc_in}")
            print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

            return actual_enc_in, feature_columns
    except Exception as e:
        print(f"Error calculating input dimension: {e}")
        return None, []


def validate_target_column(args, df):
    """验证目标列是否存在且位置正确"""
    if args.target not in df.columns:
        print(f"Error: Target column '{args.target}' not in data")
        print(f"Available columns: {list(df.columns)}")
        return False

    target_idx = df.columns.get_loc(args.target)
    print(f"Target column '{args.target}' is at column {target_idx}")

    return True


def load_data_with_safety_check(args):
    """带安全检查的数据加载"""
    data_file = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(data_file):
        print(f"Error: Data file does not exist: {data_file}")
        return None

    df = pd.read_csv(data_file)
    # 删除包含NaN值的行
    original_shape = df.shape
    df = df.dropna()  # 删除任何包含NaN值的行
    new_shape = df.shape
    if original_shape[0] != new_shape[0]:
        print(f"Removed {original_shape[0] - new_shape[0]} rows containing NaN values")
        print(f"Data shape before cleaning: {original_shape}")
        print(f"Data shape after cleaning: {new_shape}")

    print(f"Data loaded successfully, shape: {df.shape}")
    print(f"Data columns: {list(df.columns)}")

    # 验证目标列
    if not validate_target_column(args, df):
        return None

    # 动态设置维度
    actual_enc_in, feature_columns = calculate_input_dims(args)
    if actual_enc_in is not None:
        args.enc_in = actual_enc_in
        args.dec_in = actual_enc_in
        args.feature_columns = feature_columns
        print(f"Dynamically set input dimensions: enc_in={args.enc_in}, dec_in={args.dec_in}")
    else:
        print("Using default input dimensions, potential risk")
        args.feature_columns = []

    print(f"Feature columns: {args.feature_columns}")
    print(f"Target column: {args.target}")

    return df


def data_safety_check(args):
    """数据安全检查"""
    try:
        df = load_data_with_safety_check(args)
        if df is None:
            return False

        # 检查数据维度匹配
        if hasattr(args, 'enc_in'):
            feature_count = len(args.feature_columns)
            if args.enc_in != feature_count:
                print(f"Warning: Configured enc_in({args.enc_in}) does not match actual feature count({feature_count})")
                # 自动修正
                args.enc_in = feature_count
                args.dec_in = feature_count
                print(f"Automatically corrected to: enc_in={args.enc_in}, dec_in={args.dec_in}")

        # 检查目标列数据有效性
        target_data = df[args.target]
        if target_data.isna().all():
            print(f"Error: Target column '{args.target}' is all NaN values")
            return False

        print(
            f"Target column stats - Min: {target_data.min():.2f}, Max: {target_data.max():.2f}, Non-null: {target_data.count()}")
        return True

    except Exception as e:
        print(f"Data safety check failed: {e}")
        return False


# 创建参数配置 - 使用 TimeMixer 模型
class Args:
    def __init__(self):
        # 基本配置
        self.task_name = 'short_term_forecast'
        self.is_training = 1
        self.model_id = "my_custom_experiment"
        self.model = 'TimeMixer'
        self.frequency_map = {'B': 0}  # 工作日频率

        # 数据加载器
        self.data = 'custom'
        self.root_path = './mydata'
        self.data_path = 'dataset2.csv'
        self.features = 'M'
        self.target = '原油期货收盘价'
        self.freq = 'b'
        self.checkpoints = './checkpoints/'

        # 预报任务
        self.seq_len = 30
        self.label_len = 15
        self.pred_len = 1
        self.seasonal_patterns = "Monthly"
        self.inverse = True  # 启用反标准化

        # model define - 初始值会被动态更新
        self.top_k = 10
        self.num_kernels = 12
        self.enc_in = 1  # 初始值，会被动态更新
        self.dec_in = 1  # 初始值，会被动态更新
        self.c_out = 1
        self.d_model = 96
        self.n_heads = 128
        self.e_layers = 48
        self.d_layers = 16
        self.d_ff = 64
        self.moving_avg = 15  # 必须为奇数，且小于 seq_len
        self.factor = 1
        self.distil = True
        self.dropout = 0.1
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.output_attention = False
        self.channel_independence = 1
        self.decomp_method = 'moving_avg'
        self.use_norm = 0  # 关键修复：禁用标准化层
        self.down_sampling_layers = 0
        self.down_sampling_window = 1
        self.down_sampling_method = 'avg'
        self.use_future_temporal_feature = 0

        # 优化
        self.num_workers = 0
        self.itr = 1
        self.train_epochs = 10
        self.batch_size = 32
        self.patience = 15
        self.learning_rate = 0.01
        self.des = "test"
        self.loss = "MSE"
        self.lradj = 'TST'
        self.pct_start = 0.25
        self.use_amp = False
        self.comment = 'jupyter_run_fixed'

        # GPU
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = True
        self.devices = '0'
        self.device_ids = [0]  # 添加这行

        # 非固定投影仪参数
        self.p_hidden_dims = [128, 128]
        self.p_hidden_layers = 5


def metric_all(preds, trues, huber_delta=1.0, quantile=0.5):
    """
    计算所有评估指标（包含图中所有损失函数）
    """
    # 确保是一维数组
    preds = preds.reshape(-1)
    trues = trues.reshape(-1)

    # 移除 NaN 值
    mask = ~np.isnan(preds) & ~np.isnan(trues)
    preds = preds[mask]
    trues = trues[mask]

    if len(preds) == 0 or len(trues) == 0:
        return {
            'MAE': 0, 'MSE': 0, 'RMSE': 0, 'MAPE': 0, 'MSPE': 0, 'R2': 0,
            'MBE': 0, 'RAE': 0, 'RSE': 0, 'MSLE': 0, 'RMSLE': 0,
            'NRMSE': 0, 'RRMSE': 0, 'HuberLoss': 0, 'LogCoshLoss': 0,
            'QuantileLoss': 0
        }

    # 基础指标
    mae = np.mean(np.abs(preds - trues))
    mse = np.mean((preds - trues) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(trues, preds)

    # MBE (Mean Bias Error)
    mbe = np.mean(preds - trues)

    # RAE (Relative Absolute Error)
    rae = np.sum(np.abs(preds - trues)) / np.sum(np.abs(trues - np.mean(trues)))

    # RSE (Relative Squared Error)
    rse = np.sum((preds - trues) ** 2) / np.sum((trues - np.mean(trues)) ** 2)

    # MSLE (Mean Squared Logarithmic Error)
    # 确保值为正
    trues_msle = np.abs(trues) + 1e-8
    preds_msle = np.abs(preds) + 1e-8
    msle = np.mean((np.log1p(preds_msle) - np.log1p(trues_msle)) ** 2)

    # RMSLE (Root Mean Squared Logarithmic Error)
    rmsle = np.sqrt(msle)

    # NRMSE (Normalized RMSE)
    nrmse = rmse / (trues.max() - trues.min())

    # RRMSE (Relative RMSE)
    rrmse = rmse / np.mean(trues)

    # Huber Loss
    residuals = preds - trues
    huber_loss = np.mean(np.where(np.abs(residuals) <= huber_delta,
                                  0.5 * residuals ** 2,
                                  huber_delta * (np.abs(residuals) - 0.5 * huber_delta)))

    # LogCosh Loss
    logcosh_loss = np.mean(np.log(np.cosh(preds - trues)))

    # Quantile Loss
    error = trues - preds
    quantile_loss = np.mean(np.where(error >= 0, quantile * error, (quantile - 1) * error))

    # MAPE和MSPE（避免除以零）
    if np.any(trues == 0):
        mape = np.nan
        mspe = np.nan
    else:
        mape = np.mean(np.abs((preds - trues) / trues)) * 100
        mspe = np.mean(((preds - trues) / trues) ** 2) * 100

    return {
        'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'MSPE': mspe, 'R2': r2,
        'MBE': mbe, 'RAE': rae, 'RSE': rse, 'MSLE': msle, 'RMSLE': rmsle,
        'NRMSE': nrmse, 'RRMSE': rrmse, 'HuberLoss': huber_loss,
        'LogCoshLoss': logcosh_loss, 'QuantileLoss': quantile_loss
    }


def plot_daily_predictions(true_dates, true_values, pred_values, title="每日预测结果", results_dir="./results"):
    """
    绘制每日预测结果对比图
    """
    try:
        # 确保数据长度一致
        min_len = min(len(true_dates), len(true_values), len(pred_values))
        true_dates = true_dates[:min_len]
        true_values = true_values[:min_len]
        pred_values = pred_values[:min_len]

        print(f"Plotting data: dates {len(true_dates)}, true values {len(true_values)}, pred values {len(pred_values)}")
        print(f"True values range: {np.min(true_values):.2f} to {np.max(true_values):.2f}")
        print(f"Pred values range: {np.min(pred_values):.2f} to {np.max(pred_values):.2f}")

        # 计算 R²
        r2 = r2_score(true_values, pred_values)

        # 创建图形
        plt.figure(figsize=(14, 8))

        # 绘制真实值与预测值对比
        plt.plot(true_dates, true_values, 'b-', label='真实值', linewidth=2, alpha=0.8)
        plt.plot(true_dates, pred_values, 'r-', label='预测值', linewidth=2, alpha=0.8)

        # 添加填充区域显示误差
        plt.fill_between(true_dates, true_values, pred_values,
                         color='gray', alpha=0.2, label='误差区域')

        plt.title(f'{title}\nR² = {r2:.6f}', fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('原油期货收盘价', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 旋转 x 轴标签以便更好地显示日期
        plt.xticks(rotation=45)

        # 添加评估指标文本框
        mae_val = np.mean(np.abs(true_values - pred_values))
        mse_val = np.mean((true_values - pred_values) ** 2)
        rmse_val = np.sqrt(mse_val)

        textstr = f'R² = {r2:.6f}\nMSE = {mse_val:.6f}\nMAE = {mae_val:.6f}\nRMSE = {rmse_val:.6f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)

        plt.tight_layout()

        # 保存图片
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        plt.savefig(f'{results_dir}/daily_predictions.png', dpi=300, bbox_inches='tight')
        print(f"图片已保存到 {results_dir}/daily_predictions.png")
        plt.show()

        return r2, mse_val, mae_val, rmse_val
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0, 0


def plot_prediction_comparison_simple(true_dates, true_values, pred_values, title="预测结果对比",
                                      results_dir="./results"):
    """
    简化的预测对比图
    """
    try:
        # 确保数据长度一致
        min_len = min(len(true_dates), len(true_values), len(pred_values))
        true_dates = true_dates[:min_len]
        true_values = true_values[:min_len]
        pred_values = pred_values[:min_len]

        # 计算 R²
        r2 = r2_score(true_values, pred_values)

        # 创建图形
        plt.figure(figsize=(16, 10))

        # 子图1: 真实值与预测值对比
        plt.subplot(2, 2, 1)
        plt.plot(true_dates, true_values, 'b-', label='真实值', linewidth=2)
        plt.plot(true_dates, pred_values, 'r-', label='预测值', linewidth=2)
        plt.title('真实值与预测值对比', fontsize=14)
        plt.xlabel('日期')
        plt.ylabel('原油期货收盘价')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 子图2: 散点图
        plt.subplot(2, 2, 2)
        plt.scatter(true_values, pred_values, alpha=0.6)
        min_val = min(true_values.min(), pred_values.min())
        max_val = max(true_values.max(), pred_values.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想线')
        plt.title('真实值 vs 预测值散点图', fontsize=14)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 在散点图上添加 R²
        plt.text(0.05, 0.95, f'R² = {r2:.6f}', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                 verticalalignment='top')

        # 子图3: 残差图
        plt.subplot(2, 2, 3)
        residuals = true_values - pred_values
        plt.plot(true_dates, residuals, 'g-', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.title('残差图', fontsize=14)
        plt.xlabel('日期')
        plt.ylabel('残差')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 子图4: 误差分布
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
        plt.title('误差分布', fontsize=14)
        plt.xlabel('误差')
        plt.ylabel('频数')
        plt.grid(True, alpha=0.3)

        plt.suptitle(f'{title} - R² = {r2:.6f}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图片
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        plt.savefig(f'{results_dir}/prediction_comparison_simple.png', dpi=300, bbox_inches='tight')
        plt.show()

        return r2
    except Exception as e:
        print(f"Error in detailed comparison plot: {e}")
        import traceback
        traceback.print_exc()
        return 0


def ensure_3d_tensor(tensor):
    """
    确保张量是 3 维的 [batch_size, seq_len, features]
    """
    if tensor.dim() == 2:
        # 如果是2维，添加序列长度维度
        return tensor.unsqueeze(1)
    elif tensor.dim() == 1:
        # 如果是1维，添加批次和序列长度维度
        return tensor.unsqueeze(0).unsqueeze(-1)
    return tensor


def get_real_dates_from_data(args, test_size):
    """
    从实际数据中获取真实的日期信息 - 修复版本
    """
    try:
        # 读取原始数据文件
        data_file = os.path.join(args.root_path, args.data_path)
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)

            # 尝试找到日期列
            date_columns = ['date', 'Date', 'DATE', 'time', 'date']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break

            if date_col:
                # 转换日期格式
                df[date_col] = pd.to_datetime(df[date_col])

                # 获取测试集对应的日期（假设测试集是最后一部分）
                start_idx = max(0, len(df) - test_size)
                test_dates = df[date_col].iloc[start_idx:].values

                # 如果是工作日频率，过滤出工作日
                if args.freq.lower() == 'b':
                    # 将 numpy 数组转换为 pandas Series 以便过滤
                    date_series = pd.Series(test_dates)
                    # 过滤出工作日（周一至周五）
                    test_dates = date_series[date_series.dt.dayofweek < 5].values

                print(f"Retrieved {len(test_dates)} test dates from data file")
                print(f"Actual date range: {test_dates[0]} to {test_dates[-1]}")
                return test_dates[:test_size]  # 确保不超过测试集大小
    except Exception as e:
        print(f"Error getting dates from data file: {e}")

    # 如果无法从文件获取，生成基于工作日的日期
    print("Using generated business day dates")
    end_date = pd.Timestamp.now()
    start_date = end_date - BDay(test_size + 10)  # 多生成一些以确保有足够的工作日

    # 生成工作日日期范围
    date_range = pd.bdate_range(start=start_date, end=end_date)

    # 取最后 test_size 个日期
    if len(date_range) >= test_size:
        return date_range[-test_size:]
    else:
        return date_range


def get_daily_predictions(model, args, data_loader, device, dates, scaler=None):
    """
    获取每日预测结果
    """
    model.eval()
    daily_preds = []
    daily_trues = []
    daily_dates = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            # 跳过空批次
            if batch_x.nelement() == 0:
                continue

            # 确保数据是3维的
            batch_x = ensure_3d_tensor(batch_x.float().to(device))
            batch_y = ensure_3d_tensor(batch_y.float().to(device))

            # 处理时间标记数据
            if batch_x_mark is not None and batch_x_mark.nelement() > 0:
                batch_x_mark = ensure_3d_tensor(batch_x_mark.float().to(device))
            else:
                batch_x_mark = None

            if batch_y_mark is not None and batch_y_mark.nelement() > 0:
                batch_y_mark = ensure_3d_tensor(batch_y_mark.float().to(device))
            else:
                batch_y_mark = None

            try:
                # 前向传播
                outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                # 获取预测部分
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

                # 只取最后一个预测值（每日预测）
                pred = outputs[:, -1, :].detach().cpu().numpy()
                true = batch_y[:, -1, :].detach().cpu().numpy()

                batch_size = batch_x.size(0)

                # 获取对应的日期 - 确保长度匹配
                start_idx = i * args.batch_size
                end_idx = start_idx + batch_size

                # 确保不超出日期数组范围
                if end_idx > len(dates):
                    end_idx = len(dates)
                    # 如果日期不足，调整批次大小
                    batch_size = end_idx - start_idx
                    if batch_size <= 0:
                        break

                batch_dates = dates[start_idx:end_idx]

                # 确保日期数量与预测结果数量一致
                if len(batch_dates) == len(pred):
                    daily_preds.append(pred)
                    daily_trues.append(true)
                    daily_dates.extend(batch_dates)
                else:
                    print(f"Batch {i} date mismatch: dates {len(batch_dates)}, predictions {len(pred)}")
                    # 如果日期数量不匹配，截断预测结果以匹配日期数量
                    min_len = min(len(batch_dates), len(pred))
                    daily_preds.append(pred[:min_len])
                    daily_trues.append(true[:min_len])
                    daily_dates.extend(batch_dates[:min_len])

            except Exception as e:
                print(f"Prediction error in batch {i}: {e}")
                continue

    if not daily_preds:
        print("No valid prediction data")
        return [], [], []

    daily_preds = np.concatenate(daily_preds, axis=0)
    daily_trues = np.concatenate(daily_trues, axis=0)

    # 最终检查长度一致性
    min_len = min(len(daily_dates), len(daily_trues), len(daily_preds))
    daily_dates = daily_dates[:min_len]
    daily_trues = daily_trues[:min_len]
    daily_preds = daily_preds[:min_len]

    print(f"Final return: dates {len(daily_dates)}, trues {daily_trues.shape}, preds {daily_preds.shape}")

    return daily_dates, daily_trues, daily_preds


def get_original_data_stats(args):
    """
    获取原始数据的统计信息
    """
    try:
        data_file = os.path.join(args.root_path, args.data_path)
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)

            # 找到目标列
            if args.target in df.columns:
                target_data = df[args.target]
                target_idx = df.columns.get_loc(args.target)

                print(f"Target column '{args.target}' position in original data: column {target_idx}")
                print(
                    f"Target column '{args.target}' range in original data: {target_data.min():.2f} to {target_data.max():.2f}")
                print(f"Target column '{args.target}' mean: {target_data.mean():.2f}, std: {target_data.std():.2f}")

                return {
                    "target_idx": target_idx,
                    'min': target_data.min(),
                    'max': target_data.max(),
                    'mean': target_data.mean(),
                    'std': target_data.std(),
                    'columns': list(df.columns)
                }
            else:
                print(f"Warning: Target column '{args.target}' not found in data")
                return None
    except Exception as e:
        print(f"Error getting original data stats: {e}")
        return None


def manual_inverse_scale_global(data, original_stats):
    """
    使用全局原始数据统计信息进行手动反标准化
    """
    if original_stats is None:
        return data

    # 计算标准化前的数据范围
    data_min = data.min()
    data_max = data.max()

    print(f"Data range before manual inverse scaling: {data_min:.4f} to {data_max:.4f}")
    print(f"Original data global range: {original_stats['min']:.2f} to {original_stats['max']:.2f}")
    print(f"Original data global mean: {original_stats['mean']:.2f}, std: {original_stats['std']:.2f}")

    # 假设数据被标准化为标准正态分布 (均值为0，标准差为1)
    # 使用原始数据的均值和标准差进行反标准化
    scaled_data = data * original_stats['std'] + original_stats['mean']

    print(f"Data range after manual inverse scaling: {scaled_data.min():.2f} to {scaled_data.max():.2f}")
    return scaled_data


def extract_target_variable_safe(data, target_idx, original_columns, current_columns, data_name="data"):
    """
    安全地提取目标变量
    """
    if data.ndim == 1:
        # 如果数据已经是1维，直接返回
        print(f"{data_name} is already 1D, returning directly")
        return data

    # 验证列对应关系
    if hasattr(original_columns, '__len__') and hasattr(current_columns, '__len__'):
        print(f"Original data columns: {original_columns}")
        print(f"Current data column count: {len(current_columns)}")

    if data.shape[1] > target_idx:
        # 正常情况：目标索引在范围内
        extracted = data[:, target_idx]
        print(f"{data_name} successfully extracted target variable from column {target_idx}, shape: {extracted.shape}")
        return extracted
    else:
        # 索引越界：使用最后一列
        last_idx = data.shape[1] - 1
        print(f"Warning: {data_name} target index {target_idx} out of range (0-{last_idx}), using last column")
        extracted = data[:, last_idx]
        print(f"{data_name} using last column, shape: {extracted.shape}")
        return extracted


def patch_normalize_layers(exp_instance):
    """修补标准化层"""
    # 检查模型是否有标准化层
    if hasattr(exp_instance.model, 'normalize_layers'):
        print("Detected normalize layers, patching...")

        # 获取实际的输入维度
        actual_dim = exp_instance.args.enc_in
        print(f"Actual input dimension: {actual_dim}")

        # 修补每个标准化层
        for i, layer in enumerate(exp_instance.model.normalize_layers):
            if hasattr(layer, 'affine_weight'):
                current_dim = layer.affine_weight.shape[0]
                print(f"Normalize layer {i} current dimension: {current_dim}")

                if current_dim != actual_dim:
                    print(f"Normalize layer {i} dimension mismatch ({current_dim} vs {actual_dim}), reinitializing...")

                    # 重新初始化权重和偏置
                    layer.affine_weight = torch.nn.Parameter(torch.ones(actual_dim))
                    layer.affine_bias = torch.nn.Parameter(torch.zeros(actual_dim))

                    # 更新统计信息
                    if hasattr(layer, 'stdev'):
                        layer.stdev = torch.ones(actual_dim)
                    if hasattr(layer, 'means'):
                        layer.means = torch.zeros(actual_dim)

                    print(f"Normalize layer {i} reinitialized to dimension {actual_dim}")


def patch_model_initialization(exp_instance):
    """修补模型初始化方法"""
    original_build_model = exp_instance._build_model

    def new_build_model(self):
        print(
            f"Checking dimensions before building model: enc_in={self.args.enc_in}, dec_in={self.args.dec_in}, c_out={self.args.c_out}")
        model = original_build_model()

        # 检查模型参数
        if hasattr(model, 'normalize_layers'):
            print(f"Model normalize layer count: {len(model.normalize_layers)}")
            for i, layer in enumerate(model.normalize_layers):
                if hasattr(layer, 'affine_weight'):
                    print(f"Normalize layer {i} weight shape: {layer.affine_weight.shape}")

        return model

    exp_instance._build_model = types.MethodType(new_build_model, exp_instance)


def patch_all_methods(exp_instance):
    """修补所有有问题的验证和测试方法"""

    # 修补验证方法
    def new_vali(self, train_loader, vali_loader, criterion):
        self.model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # 跳过空批次
                if batch_x.nelement() == 0:
                    continue

                # 确保数据是3维的
                batch_x = ensure_3d_tensor(batch_x.float().to(self.device))
                batch_y = ensure_3d_tensor(batch_y.float().to(self.device))

                # 处理时间标记数据
                if batch_x_mark is not None and batch_x_mark.nelement() > 0:
                    batch_x_mark = ensure_3d_tensor(batch_x_mark.float().to(self.device))
                else:
                    batch_x_mark = None

                if batch_y_mark is not None and batch_y_mark.nelement() > 0:
                    batch_y_mark = ensure_3d_tensor(batch_y_mark.float().to(self.device))
                else:
                    batch_y_mark = None

                # 前向传播
                try:
                    outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                    # 计算损失 - 只使用预测部分
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    total_loss += loss.item() * batch_x.size(0)
                    total_samples += batch_x.size(0)
                except Exception as e:
                    print(f"Validation batch {i} error: {e}")
                    continue

        self.model.train()
        return total_loss / total_samples if total_samples > 0 else 0

    # 修补测试方法
    def new_test(self, setting, test=0, results_dir=None):
        if results_dir is None:
            results_dir = "./results"

        _, test_loader = self._get_data(flag="test")

        if test:
            print("Loading model")
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # 获取测试集大小
        test_size = len(test_loader.dataset)
        print(f"Test set size: {test_size}")

        # 获取真实的日期信息
        test_dates = get_real_dates_from_data(self.args, test_size)

        # 获取 scaler 用于反标准化
        scaler = None
        if hasattr(self, 'scaler'):
            scaler = self.scaler
            print(f"Found scaler: {type(scaler)}")
            if hasattr(scaler, 'mean_'):
                print(f"Scaler mean: {scaler.mean_}")
            if hasattr(scaler, 'std_'):
                print(f"Scaler std: {scaler.std_}")
        elif hasattr(test_loader.dataset, "scaler"):
            scaler = test_loader.dataset.scaler
            print(f"Found scaler from dataset: {type(scaler)}")

        # 获取原始数据统计信息
        original_stats = get_original_data_stats(self.args)

        # 获取每日预测结果
        daily_dates, daily_trues, daily_preds = get_daily_predictions(
            self.model, self.args, test_loader, self.device, test_dates)

        if len(daily_dates) == 0:
            print("No valid test data")
            return 0, "./results"

        print(f'Daily prediction data shape: trues {daily_trues.shape}, preds {daily_preds.shape}')
        print(f'Date count: {len(daily_dates)}')
        print(f'Date range: {daily_dates[0]} to {daily_dates[-1]}')

        # 安全地提取目标变量
        if original_stats:
            target_idx = original_stats['target_idx']
            original_columns = original_stats.get("columns", [])
            print(f"Using target index: {target_idx}")
        else:
            target_idx = 0
            original_columns = []
            print(f"Using default target index: {target_idx}")

        # 获取当前数据的列信息
        current_columns = []
        if hasattr(test_loader.dataset, 'data_columns'):
            current_columns = test_loader.dataset.data_columns
        elif hasattr(self.args, 'feature_columns'):
            current_columns = self.args.feature_columns + [self.args.target]

        # 安全地提取目标变量
        daily_trues_target = extract_target_variable_safe(daily_trues, target_idx, original_columns, current_columns,
                                                          "true values")
        daily_preds_target = extract_target_variable_safe(daily_preds, target_idx, original_columns, current_columns,
                                                          "pred values")

        print(f"Target variable shape - trues: {daily_trues_target.shape}, preds: {daily_preds_target.shape}")
        print(f"After normalization range - trues: {daily_trues_target.min():.4f} to {daily_trues_target.max():.4f}")
        print(f"After normalization range - preds: {daily_preds_target.min():.4f} to {daily_preds_target.max():.4f}")

        # 应用反标准化
        print("Applying global inverse scaling...")
        daily_preds_target = manual_inverse_scale_global(daily_preds_target, original_stats)
        daily_trues_target = manual_inverse_scale_global(daily_trues_target, original_stats)

        # 确保所有数组长度一致
        min_length = min(len(daily_dates), len(daily_trues_target), len(daily_preds_target))

        # 截断数组到相同长度
        daily_dates = daily_dates[:min_length]
        daily_trues_flat = daily_trues_target[:min_length]
        daily_preds_flat = daily_preds_target[:min_length]

        print(
            f"Adjusted array lengths - dates: {len(daily_dates)}, trues: {len(daily_trues_flat)}, preds: {len(daily_preds_flat)}")

        # 计算5日均值预测
        ma5_preds = calculate_5day_moving_average(daily_trues_flat, daily_dates)

        # 计算完整的评估指标
        metrics_timemixer = metric_all(daily_trues_flat, daily_preds_flat)
        metrics_ma5 = metric_all(daily_trues_flat, ma5_preds)

        print(f'\nTimeMixer Daily prediction evaluation metrics:')
        for key, value in metrics_timemixer.items():
            print(f'{key}: {value:.6f}')

        print(f'\n5-Day MA prediction evaluation metrics:')
        for key, value in metrics_ma5.items():
            print(f'{key}: {value:.6f}')

        # 创建带有时间戳和MSE值的结果文件夹
        final_results_dir = create_results_folder(metrics_timemixer['MSE'], model_name="TimeMixer")

        # 保存配置信息
        config_info = {
            'model': self.args.model,
            "target_column": self.args.target,
            "sequence_length": self.args.seq_len,
            'prediction_length': self.args.pred_len,
            "input_dimension": self.args.enc_in,
            "output_dimension": self.args.c_out,
            'feature_columns': getattr(self.args, 'feature_columns', []),
            'training_epochs': self.args.train_epochs,
            "batch_size": self.args.batch_size
        }

        config_df = pd.DataFrame([config_info])
        config_df.to_csv(f'{final_results_dir}/model_config.csv', index=False, encoding='utf-8')
        print("Model configuration saved to model_config.csv")

        # 绘制每日预测结果对比图
        print("\nGenerating daily prediction visualization...")
        try:
            r2_simple, mse_val, mae_val, rmse_val = plot_daily_predictions(
                daily_dates, daily_trues_flat, daily_preds_flat, "TimeMixer Daily Prediction Results",
                final_results_dir)

            # 检查 R²是否一致
            if abs(metrics_timemixer['R2'] - r2_simple) > 0.0001:
                print(
                    f"Warning: R² values inconsistent! Metric calc: {metrics_timemixer['R2']:.6f}, Plot calc: {r2_simple:.6f}")

            # 绘制详细对比图
            plot_prediction_comparison_simple(
                daily_dates, daily_trues_flat, daily_preds_flat, "TimeMixer Daily Prediction Detailed Analysis",
                final_results_dir)

            # 绘制5日均值对比图
            plot_5day_ma_comparison(daily_dates, daily_trues_flat, daily_preds_flat, ma5_preds, final_results_dir)
        except Exception as e:
            print(f"Error during plotting: {e}")
            import traceback
            traceback.print_exc()

        # 保存详细结果
        if not os.path.exists(final_results_dir):
            os.makedirs(final_results_dir)

        # 保存TimeMixer评估指标
        eval_timemixer_df = pd.DataFrame([metrics_timemixer])
        eval_timemixer_df.to_csv(f'{final_results_dir}/timemixer_metrics.csv', index=False, encoding='utf-8')
        print("\nTimeMixer metrics saved to timemixer_metrics.csv")

        # 保存5日均值评估指标
        eval_ma5_df = pd.DataFrame([metrics_ma5])
        eval_ma5_df.to_csv(f'{final_results_dir}/ma5_metrics.csv', index=False, encoding='utf-8')
        print("5-Day MA metrics saved to ma5_metrics.csv")

        # 保存预测值和真实值
        results_df = pd.DataFrame({
            'date': daily_dates,
            'true_value': daily_trues_flat,
            'timemixer_pred': daily_preds_flat,
            'ma5_pred': ma5_preds,
            'timemixer_error': daily_trues_flat - daily_preds_flat,
            'ma5_error': daily_trues_flat - ma5_preds
        })
        results_df.to_csv(f'{final_results_dir}/predictions_vs_true.csv', index=False, encoding='utf-8')
        print("Predictions and true values saved to predictions_vs_true.csv")

        # 保存原始数据统计信息
        if original_stats:
            stats_df = pd.DataFrame([original_stats])
            stats_df.to_csv(f'{final_results_dir}/original_data_statistics.csv', index=False, encoding='utf-8')
            print("Original data statistics saved to original_data_statistics.csv")

        # 保存结果到 TimeMixer 的测试结果目录
        folder_path = os.path.join(final_results_dir, 'test_results', setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保存指标
        try:
            with open(folder_path + 'daily_metrics.txt', 'w', encoding='UTF-8') as f:
                f.write(f"""TimeMixer Metrics:
MSE: {metrics_timemixer['MSE']:.6f}
MAE: {metrics_timemixer['MAE']:.6f}
R2: {metrics_timemixer['R2']:.6f}

5-Day MA Metrics:
MSE: {metrics_ma5['MSE']:.6f}
MAE: {metrics_ma5['MAE']:.6f}
R2: {metrics_ma5['R2']:.6f}

Improvement (MSE reduction): {((metrics_ma5['MSE'] - metrics_timemixer['MSE']) / metrics_ma5['MSE'] * 100):.2f}%
""")
        except UnicodeEncodeError:
            with open(folder_path + 'daily_metrics.txt', 'w', encoding='UTF-8') as f:
                f.write(
                    f'MSE: {metrics_timemixer["MSE"]:.6f}\nMAE: {metrics_timemixer["MAE"]:.6f}\nR2: {metrics_timemixer["R2"]:.6f}')

        # 保存预测值和真实值
        np.save(folder_path + 'daily_pred.npy', daily_preds)
        np.save(folder_path + 'daily_true.npy', daily_trues)

        # 保存完整的实验日志
        log_content = f"""
TimeMixer Experiment Completion Report
====================
Experiment time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Experiment setting: {setting}
Target variable: {self.args.target}
Data dimensions: input={self.args.enc_in}, output={self.args.c_out}

TimeMixer Metrics:
--------"""

        for key, value in metrics_timemixer.items():
            log_content += f"\n{key}: {value:.6f}"

        log_content += f"""

5-Day MA Metrics:
--------"""

        for key, value in metrics_ma5.items():
            log_content += f"\n{key}: {value:.6f}"

        log_content += f"""

Data Statistics:
--------
Test sample count: {len(daily_dates)}
Date range: {daily_dates[0]} to {daily_dates[-1]}
True values range: {daily_trues_flat.min():.2f} to {daily_trues_flat.max():.2f}
TimeMixer pred range: {daily_preds_flat.min():.2f} to {daily_preds_flat.max():.2f}
MA5 pred range: {ma5_preds.min():.2f} to {ma5_preds.max():.2f}

Feature columns ({len(getattr(self.args, 'feature_columns', []))}):
{getattr(self.args, 'feature_columns', [])}
"""

        with open(f'{final_results_dir}/experiment_report.txt', 'w', encoding='UTF-8') as f:
            f.write(log_content)

        print(f"\n所有结果保存到: {final_results_dir}")
        print(f"\n实验完成！TimeMixer MSE = {metrics_timemixer['MSE']:.6f}, 5-Day MA MSE = {metrics_ma5['MSE']:.6f}")
        print(
            f"TimeMixer相比5日均值提升了: {((metrics_ma5['MSE'] - metrics_timemixer['MSE']) / metrics_ma5['MSE'] * 100):.2f}%")

        return metrics_timemixer['MSE'], final_results_dir

    # 应用所有补丁
    exp_instance.vali = types.MethodType(new_vali, exp_instance)
    exp_instance.test = types.MethodType(new_test, exp_instance)

    # 修补模型初始化
    patch_model_initialization(exp_instance)

    # 修补标准化层
    patch_normalize_layers(exp_instance)

    print("Successfully patched all methods")


# 主执行代码
if __name__ == "__main__":
    # 创建参数实例
    args = Args()

    # 执行数据安全检查
    print("Performing data safety check...")
    if not data_safety_check(args):
        print("Data safety check failed! Please check data file and target column configuration.")
        sys.exit(1)
    else:
        print("Data safety check passed!")

    # 检查移动平均窗口大小是否有效
    if args.moving_avg >= args.seq_len:
        print(
            f"Warning: moving_avg ({args.moving_avg}) is greater than or equal to seq_len ({args.seq_len}), may cause errors")
        print("Automatically adjusting moving_avg to 13")
        args.moving_avg = 13

    if args.moving_avg % 2 == 0:
        print(f"Warning: moving_avg ({args.moving_avg}) should be odd")
        print("Automatically adjusting moving_avg to 13")
        args.moving_avg = 13

    # 选择实验类型
    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast

        Exp = Exp_Short_Term_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

        Exp = Exp_Long_Term_Forecast

    # 运行实验
    if args.is_training:
        for ii in range(args.itr):
            # 设置实验记录
            setting = '{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.comment,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)

            # 应用补丁
            patch_all_methods(exp)

            print('>>>>>>>> 开始训练 : {}>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>>> 开始测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mse_score, results_dir = exp.test(setting)
            torch.cuda.empty_cache()

            print(f"实验完成！最终MSE = {mse_score:.6f}")
            print(f"所有结果保存在: {results_dir}")
    else:
        ii = 0
        setting = '{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.comment,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)

        # 应用补丁
        patch_all_methods(exp)

        print('>>>>>>>>> 开始测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mse_score, results_dir = exp.test(setting, test=1)
        torch.cuda.empty_cache()

        print(f"测试完成！最终MSE = {mse_score:.6f}")
        print(f"所有结果保存在: {results_dir}")