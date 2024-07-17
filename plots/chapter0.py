import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from sklearn.linear_model import LinearRegression

plt.style.use('fivethirtyeight')


def fit_model(x_train, y_train):
    """
    使用线性回归模型拟合训练数据，从而找到最佳拟合的截距和系数（权重），以最小化预测值与实际值之间的误差
    Parameters
    ----------
    x_train
        训练数据的特征（自变量）
    y_train
        训练数据的目标值（因变量）

    Returns
    -------
    b_minimum : float
        最佳拟合的截距
    w_minimum : float
        最佳拟合的系数（权重）

    """
    # Fits a linear regression to find the actual b and w that minimize the loss
    regression = LinearRegression()
    regression.fit(x_train, y_train)
    # intercept:截距  coef:权重
    b_minimum, w_minimum = regression.intercept_[0], regression.coef_[0][0]
    return b_minimum, w_minimum


def find_index(b, w, bs, ws):
    # Looks for the closer indexes for the updated b and w inside their respective ranges
    # 查找bs和ws网格中最接近给定b和w值的索引
    b_idx = np.argmin(np.abs(bs[0, :] - b))
    w_idx = np.argmin(np.abs(ws[:, 0] - w))

    # Closest values for b and w
    fixedb, fixedw = bs[0, b_idx], ws[w_idx, 0]

    return b_idx, w_idx, fixedb, fixedw


def figure1(x_train, y_train, x_val, y_val):
    # 创建一个图形（fig），包含两个子图（ax）的布局
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制训练数据的散点图
    ax[0].scatter(x_train, y_train)
    # 设置x轴标签
    ax[0].set_xlabel('x')
    # 设置y轴标签
    ax[0].set_ylabel('y')
    # 设置y轴的显示范围
    ax[0].set_ylim([0, 3.1])
    # 设置子图标题
    ax[0].set_title('Generated Data - Train')

    ax[1].scatter(x_val, y_val, c='r')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_ylim([0, 3.1])
    ax[1].set_title('Generated Data - Validation')
    fig.tight_layout()

    return fig, ax


def figure2(x_train, y_train, b, w, color='k'):
    # Generates evenly spaced x feature
    # 生成101个从0到1的等差数列
    x_range = np.linspace(0, 1, 101)
    # Computes yhat
    # 计算预测值
    yhat_range = b + w * x_train

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0, 3])

    # Dataset
    ax.scatter(x_train, y_train)
    # Predictions
    # 绘制预测值线
    ax.plot(x_train, yhat_range, label='Model\'s predictions', c=color, linestyle='--')

    # Annotations
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b[0], w[0]), xy=(.2, .55), c=color)
    ax.legend(loc=0)
    fig.tight_layout()
    return fig, ax


def figure3(x_train, y_train, b, w):
    fig, ax = figure2(x_train, y_train, b, w)

    # First data point
    # 提取第一个数据点的横坐标 x0 和纵坐标 y0
    x0, y0 = x_train[0][0], y_train[0][0]
    # First data point
    # 在图中绘制第一个数据点，并用红色标记
    ax.scatter([x0], [y0], c='r')
    # Vertical line showing error between point and prediction
    # 在第一个数据点和预测值之间绘制一条垂直的虚线，以表示误差
    ax.plot([x0, x0], [b[0] + w[0] * x0, y0 - .03], c='r', linewidth=2, linestyle='--')
    ax.arrow(x0, y0 - .03, 0, .03, color='r', shape='full', lw=0, length_includes_head=True, head_width=.03)
    ax.arrow(x0, b[0] + w[0] * x0 + .05, 0, -.03, color='r', shape='full', lw=0, length_includes_head=True,
             head_width=.03)
    # Annotations
    ax.annotate(r'$error_0$', xy=(.8, 1.5))

    fig.tight_layout()
    return fig, ax


def figure4(x_train, y_train, b, w, bs, ws, all_losses):
    """
    使用matplotlib库来创建两个图表，用于可视化线性回归模型的损失曲面和参数空间
    Parameters
    ----------
    x_train:训练数据的特征集
    y_train:训练数据的目标值
    b:初始偏置项
    w:初始权重
    bs:偏置项的网格点
    ws:权重的网格点
    all_losses:损失函数值的网格

    Returns
    -------
    """
    # 使用线性回归模型来拟合给定的训练数据，获取最佳截距b和权重w
    b_minimum, w_minimum = fit_model(x_train, y_train)

    # 创建一个新的图形对象，尺寸为12x6英寸
    figure = plt.figure(figsize=(12, 6))

    # 1st plot
    # 创建1行2列的网格，并将子图（3D损失曲面图）放在第1列
    ax1 = figure.add_subplot(1, 2, 1, projection='3d')
    ax1.set_xlabel('b')
    ax1.set_ylabel('w')
    ax1.set_title('Loss Surface')

    # bs:x轴，ws:y轴，all_losses:z轴
    # rstride=1：行步幅为1
    # cstride=1：列步幅为1
    # alpha=.5：表面图的透明度，0表示完全透明，1表示完全不透明，这里设置为0.5表示半透明
    # cmap=plt.cm.jet：表面图的颜色映射为 jet 色彩映射
    # linewidth=0：表面图网格线的宽度为0，表示没有网格线
    # antialiased=True：启用抗锯齿，使表面图更加平滑
    surf = ax1.plot_surface(bs, ws, all_losses, rstride=1, cstride=1, alpha=.5, cmap=plt.cm.jet, linewidth=0,
                            antialiased=True)
    # 在上面3D损失曲面图上绘制等高线图
    ax1.contour(bs[0, :], ws[:, 0], all_losses, 10, offset=-1, cmap=plt.cm.jet)

    # 在bs和ws网格中找到最接近b_minimum和w_minimum的索引
    bidx, widx, _, _ = find_index(b_minimum, w_minimum, bs, ws)
    # 在损失曲面上标记最小损失点
    ax1.scatter(b_minimum, w_minimum, all_losses[bidx, widx], c='k')
    # 在3D图中添加文本注释
    ax1.text(-.3, 2.5, all_losses[bidx, widx], 'Minimum', zdir=(1, 0, 0))
    # Random start
    bidx, widx, _, _ = find_index(b, w, bs, ws)
    ax1.scatter(b, w, all_losses[bidx, widx], c='k')
    # Annotations
    ax1.text(-.2, -1.5, all_losses[bidx, widx], 'Random\n Start', zdir=(1, 0, 0))

    # 设置三维子图 ax1 的视角
    ax1.view_init(40, 260)

    # 2nd plot
    ax2 = figure.add_subplot(1, 2, 2)
    ax2.set_xlabel('b')
    ax2.set_ylabel('w')
    ax2.set_title('Loss Surface')

    # Loss surface
    CS = ax2.contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    ax2.clabel(CS, inline=1, fontsize=10)
    # Minimum
    ax2.scatter(b_minimum, w_minimum, c='k')
    # Random start
    ax2.scatter(b, w, c='k')
    # Annotations
    ax2.annotate('Random Start', xy=(-.2, 0.05), c='k')
    ax2.annotate('Minimum', xy=(.5, 2.2), c='k')

    figure.tight_layout()
    return figure, (ax1, ax2)


def figure5(x_train, y_train, b, w, bs, ws, all_losses):
    b_minimum, w_minimum = fit_model(x_train, y_train)

    b_idx, w_idx, fixedb, fixedw = find_index(b, w, bs, ws)

    b_range = bs[0, :]
    w_range = ws[:, 0]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title('Loss Surface')
    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    # Loss surface
    CS = axs[0].contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Minimum
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Starting point
    axs[0].scatter(fixedb, fixedw, c='k')
    # Vertical section
    axs[0].plot([fixedb, fixedb], w_range[[0, -1]], linestyle='--', c='r', linewidth=2)
    # Annotations
    axs[0].annotate('Minimum', xy=(.5, 2.2), c='k')
    axs[0].annotate('Random Start', xy=(fixedb + .1, fixedw + .1), c='k')

    axs[1].set_ylim([-.1, 15.1])
    axs[1].set_xlabel('w')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Fixed: b = {:.2f}'.format(fixedb))
    # Loss
    axs[1].plot(w_range, all_losses[:, b_idx], c='r', linestyle='--', linewidth=2)
    # Starting point
    axs[1].plot([fixedw], [all_losses[w_idx, b_idx]], 'or')

    fig.tight_layout()
    return fig, axs


def figure6(x_train, y_train, b, w, bs, ws, all_losses):
    b_minimum, w_minimum = fit_model(x_train, y_train)

    b_idx, w_idx, fixedb, fixedw = find_index(b, w, bs, ws)

    b_range = bs[0, :]
    w_range = ws[:, 0]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title('Loss Surface')
    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    # Loss surface
    CS = axs[0].contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Minimum
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Starting point
    axs[0].scatter(fixedb, fixedw, c='k')
    # Horizontal section
    axs[0].plot(b_range[[0, -1]], [fixedw, fixedw], linestyle='--', c='k', linewidth=2)
    # Annotations
    axs[0].annotate('Minimum', xy=(.5, 2.2), c='k')
    axs[0].annotate('Random Start', xy=(fixedb + .1, fixedw + .1), c='k')

    axs[1].set_ylim([-.1, 15.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Fixed: w = {:.2f}'.format(fixedw))
    # Loss
    axs[1].plot(b_range, all_losses[w_idx, :], c='k', linestyle='--', linewidth=2)
    # Starting point
    axs[1].plot([fixedb], [all_losses[w_idx, b_idx]], 'ok')

    fig.tight_layout()
    return fig, axs


def figure7(b, w, bs, ws, all_losses):
    b_range = bs[0, :]
    w_range = ws[:, 0]

    b_idx, w_idx, fixedb, fixedw = find_index(b, w, bs, ws)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_ylim([-.1, 6.1])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('MSE (loss)')
    axs[0].set_title('Fixed: b = {:.2f}'.format(fixedb))
    # Red rectangle
    # 绘制一个矩形区域，矩形左下角：(-.3,2.3) 宽：0.5 高：0.5
    rect = Rectangle((-.3, 2.3), .5, .5)
    # 在图中一次性添加多个形状
    pc = PatchCollection([rect], facecolor='r', alpha=.3, edgecolor='r')
    # 将这个矩形集合添加到第一个子图中
    axs[0].add_collection(pc)
    # Loss - fixed b
    axs[0].plot(w_range, all_losses[:, b_idx], c='r', linestyle='--', linewidth=2)
    # Starting point
    axs[0].plot([fixedw], [all_losses[w_idx, b_idx]], 'or')

    axs[1].set_ylim([-.1, 6.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('MSE (loss)')
    axs[1].set_title('Fixed: w = {:.2f}'.format(fixedw))
    axs[1].label_outer()
    # Black rectangle
    rect = Rectangle((.3, 2.3), .5, .5)
    pc = PatchCollection([rect], facecolor='k', alpha=.3, edgecolor='k')
    axs[1].add_collection(pc)
    # Loss - fixed w
    axs[1].plot(b_range, all_losses[w_idx, :], c='k', linestyle='--', linewidth=2)
    # Starting point
    axs[1].plot([fixedb], [all_losses[w_idx, b_idx]], 'ok')

    fig.tight_layout()
    return fig, axs


def loss_curves(b_idx, w_idx, b_idx_after, w_idx_after, all_losses):
    # BEFORE
    # Loss curve for b, given w is fixed
    # 提取在固定权重（索引为 w_idx）情况下，不同偏置值对应的损失
    loss_fixedw = all_losses[w_idx, :]
    # Loss curve for w, given b is fixed
    # 提取在固定偏置（索引为 b_idx）情况下，不同权重值对应的损失
    loss_fixedb = all_losses[:, b_idx]
    # Loss before
    # 提取当前偏置和权重组合下的损失值
    loss_before = all_losses[w_idx, b_idx]

    # AFTER
    # Loss after w is updated
    # 更新权重后，固定偏置情况下的损失值
    loss_after_w = all_losses[w_idx_after, b_idx]
    # Loss after b is updated
    # 更新偏置后，固定权重情况下的损失值
    loss_after_b = all_losses[w_idx, b_idx_after]
    return loss_fixedb, loss_fixedw, loss_before, loss_after_b, loss_after_w


def calc_gradient(parm_before, parm_after, loss_before, loss_after):
    # Computes changes in parm and loss
    # 计算参数变化量
    delta_parm = parm_after - parm_before
    # 计算损失变化量
    delta_loss = loss_after - loss_before
    # Computes gradient for parm
    # 计算梯度
    manual_grad = delta_loss / delta_parm
    return manual_grad, delta_parm, delta_loss


def figure8(b, w, bs, ws, all_losses):
    # 定义b和w的范围
    b_range = bs[0, :]
    w_range = ws[:, 0]

    # BEFORE 找到当前最接近 b和w的值及索引
    b_idx, w_idx, bs_before, ws_before = find_index(b, w, bs, ws)
    # AFTER 找到最接近修改后的 b和w的值及索引
    b_idx_after, w_idx_after, bs_after, ws_after = find_index(b + .12, w + .12, bs, ws)

    # 计算固定b和w时的损失曲线
    loss_fixedb, loss_fixedw, loss_before, loss_after_b, loss_after_w = loss_curves(b_idx, w_idx, b_idx_after,
                                                                                    w_idx_after, all_losses)

    # Computes gradient for b
    manual_grad_b, delta_b, delta_mse_b = calc_gradient(bs_before, bs_after, loss_before, loss_after_b)
    # Computes gradient for w
    manual_grad_w, delta_w, delta_mse_w = calc_gradient(ws_before, ws_after, loss_before, loss_after_w)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 设置 y 轴范围为 2.3 到 2.8
    axs[0].set_ylim([2.3, 2.8])
    # 设置 x 轴范围为 -0.3 到 0.2
    axs[0].set_xlim([-.3, .2])
    # 设置 x 轴标签为“w”
    axs[0].set_xlabel('w')
    # 设置 y 轴标签为“MSE (loss)”
    axs[0].set_ylabel('MSE (loss)')
    # 设置图表标题，显示固定的偏置值 b
    axs[0].set_title('Fixed: b = {:.2f}'.format(bs_before))
    # Loss curve
    # 绘制 w 变化时的损失曲线，曲线颜色为红色，线型为虚线，线宽为2
    axs[0].plot(w_range, loss_fixedb, c='r', linestyle='--', linewidth=2)
    # Point - before
    # 绘制更新前的点，颜色为红色，形状为圆形
    axs[0].plot([ws_before], [loss_before], 'or')
    # Point - after
    # 绘制更新后的点，颜色为红色，形状为圆形
    axs[0].plot([ws_after], [loss_after_w], 'or')

    # Arrows
    # 绘制一个红色水平箭头，表示参数 w 从更新后的值移动到更新前的值，箭头起点在 (ws_after, loss_before)
    axs[0].arrow(ws_after, loss_before, .01, 0, color='r', shape='full', lw=0, length_includes_head=True,
                 head_width=.01)
    # 绘制一个红色垂直箭头，表示损失从更新后的值移动到更新前的值，箭头起点在 (ws_before, loss_after_w)
    axs[0].arrow(ws_before, loss_after_w, 0, -0.01, color='r', shape='full', lw=0, length_includes_head=True,
                 head_width=.01)
    # 绘制一条红色水平线，连接更新前后的 w 值，表示参数变化
    axs[0].plot([ws_before, ws_after], [loss_before, loss_before], 'r-', linewidth=1.5)
    # 绘制一条红色垂直线，连接更新前后的损失值，表示损失变化
    axs[0].plot([ws_before, ws_before], [loss_after_w, loss_before], 'r-', linewidth=1.5)

    # Annotations
    # 在图表中 (0, 2.7) 位置添加注释，显示参数 w 的变化量
    axs[0].annotate(r'$\delta w = {:.2f}$'.format(delta_w), xy=(.0, 2.7), c='k', fontsize=15)
    # 在图表中 (-0.23, 2.45) 位置添加注释，显示损失的变化量
    axs[0].annotate(r'$\delta MSE = {:.2f}$'.format(delta_mse_w), xy=(-.23, 2.45), c='k', fontsize=15)
    # 在图表中 (-0.05, 2.6) 位置添加注释，显示手动计算的梯度值
    axs[0].annotate(r'$\frac{\delta MSE}{\delta w} \approx' + '{:.2f}$'.format(manual_grad_w), xy=(-.05, 2.6), c='k',
                    fontsize=17)

    axs[1].set_ylim([2.3, 2.8])
    axs[1].set_xlim([.3, .8])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('MSE (loss)')
    axs[1].set_title('Fixed: w = {:.2f}'.format(ws_before))
    # Loss Curve
    axs[1].plot(b_range, loss_fixedw, c='k', linestyle='--', linewidth=2)
    # Point - before
    axs[1].plot([bs_before], [loss_before], 'ok')
    # Point - after
    axs[1].plot([bs_after], [loss_after_b], 'ok')

    # Arrows
    axs[1].arrow(bs_after, loss_before, .01, 0, color='k', shape='full', lw=0, length_includes_head=True,
                 head_width=.01)
    axs[1].arrow(bs_before, loss_after_b, 0, -0.01, color='k', shape='full', lw=0, length_includes_head=True,
                 head_width=.01)
    axs[1].plot([bs_before, bs_after], [loss_before, loss_before], 'k-', linewidth=1.5)
    axs[1].plot([bs_before, bs_before], [loss_after_b, loss_before], 'k-', linewidth=1.5)

    # Annotations
    axs[1].annotate(r'$\delta b = {:.2f}$'.format(delta_b), xy=(.67, 2.7), c='k', fontsize=15)
    axs[1].annotate(r'$\delta MSE = {:.2f}$'.format(delta_mse_b), xy=(.45, 2.32), c='k', fontsize=15)
    axs[1].annotate(r'$\frac{\delta MSE}{\delta b} \approx' + '{:.2f}$'.format(manual_grad_b), xy=(.62, 2.6), c='k',
                    fontsize=17)

    axs[1].label_outer()

    fig.tight_layout()
    return fig, axs


def figure9(x_train, y_train, b, w):
    # Since we updated b and w, let's regenerate the initial ones
    # That's how using a random seed is useful, for instance
    np.random.seed(42)
    b_initial = np.random.randn(1)
    w_initial = np.random.randn(1)

    fig, ax = figure2(x_train, y_train, b_initial, w_initial)

    # Generates evenly spaced x feature
    x_range = np.linspace(0, 1, 101)
    # Model's predictions for updated paramaters
    yhat_range = b + w * x_range
    # Updated predictions
    ax.plot(x_range, yhat_range, label='Using parameters\nafter one update', c='g', linestyle='--')
    # Annotations
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b[0], w[0]), xy=(.2, .95), c='g')

    fig.tight_layout()
    return fig, ax


def figure10(b, w, bs, ws, all_losses, manual_grad_b, manual_grad_w, lr):
    b_range = bs[0, :]
    w_range = ws[:, 0]

    # BEFORE
    b_idx, w_idx, bs_before, ws_before = find_index(b, w, bs, ws)
    # AFTER
    new_b_idx, new_w_idx, bs_after, ws_after = find_index(bs_before - lr * manual_grad_b,
                                                          ws_before - lr * manual_grad_w,
                                                          bs,
                                                          ws)
    # Loss before
    loss_before = all_losses[w_idx, b_idx]
    loss_fixedb = all_losses[:, b_idx]
    loss_fixedw = all_losses[w_idx, :]
    loss_after_b = all_losses[w_idx, new_b_idx]
    loss_after_w = all_losses[new_w_idx, b_idx]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].set_ylim([-.1, 6.1])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('MSE (loss)')
    axs[0].set_title('Fixed: b = {:.2f}'.format(bs_before))

    # Loss curve for w, given fixed b
    axs[0].plot(w_range, loss_fixedb, c='r', linestyle='--', linewidth=2)
    # w before update
    axs[0].plot([ws_before], [loss_before], 'or')

    # Arrows
    axs[0].arrow(ws_after, loss_before, .1, 0, color='r', shape='full', lw=0, length_includes_head=True, head_width=.1)
    axs[0].plot([ws_before, ws_after], [loss_before, loss_before], 'r-', linewidth=1.5)
    axs[0].plot([ws_after], [loss_after_w], 'or')

    # Annotations
    axs[0].annotate(r'$\eta = {:.2f}$'.format(lr), xy=(1.6, 5.5), c='k', fontsize=17)
    axs[0].annotate(r'$-\eta \frac{\delta MSE}{\delta b} \approx' + '{:.2f}$'.format(-lr * manual_grad_w), xy=(1, 2),
                    c='k', fontsize=17)

    axs[1].set_ylim([-.1, 6.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('MSE (loss)')
    axs[1].set_title('Fixed: w = {:.2f}'.format(ws_before))
    axs[1].label_outer()

    # Loss curve for b, given fixed w
    axs[1].plot(b_range, loss_fixedw, c='k', linestyle='--', linewidth=2)
    # b before update
    axs[1].plot([bs_before], [loss_before], 'ok')

    # Arrows
    axs[1].arrow(bs_after, loss_before, .1, 0, color='k', shape='full', lw=0, length_includes_head=True, head_width=.1)
    axs[1].plot([bs_before, bs_after], [loss_before, loss_before], 'k-', linewidth=1.5)
    axs[1].plot([bs_after], [loss_after_b], 'ok')

    # Annotations
    axs[1].annotate(r'$\eta = {:.2f}$'.format(lr), xy=(0.6, 5.5), c='k', fontsize=17)
    axs[1].annotate(r'$-\eta \frac{\delta MSE}{\delta w} \approx' + '{:.2f}$'.format(-lr * manual_grad_b), xy=(1, 2),
                    c='k', fontsize=17)

    fig.tight_layout()
    return fig, axs


def figure14(x_train, y_train, b, w, bad_bs, bad_ws, bad_x_train):
    bad_b_range = bad_bs[0, :]
    bad_w_range = bad_ws[:, 0]

    # So we recompute the surface for X_TRAIN using the new ranges
    all_predictions = np.apply_along_axis(func1d=lambda x: bad_bs + bad_ws * x, axis=1, arr=x_train)
    all_errors = (all_predictions - y_train.reshape(-1, 1, 1))
    all_losses = (all_errors ** 2).mean(axis=0)

    # Then we compute the surface for BAD_X_TRAIN using the new ranges
    bad_all_predictions = np.apply_along_axis(func1d=lambda x: bad_bs + bad_ws * x, axis=1, arr=bad_x_train)
    bad_all_errors = (bad_all_predictions - y_train.reshape(-1, 1, 1))
    bad_all_losses = (bad_all_errors ** 2).mean(axis=0)

    b_idx, w_idx, fixedb, fixedw = find_index(b, w, bad_bs, bad_ws)

    b_minimum, w_minimum = fit_model(x_train, y_train)

    bad_b_minimum, bad_w_minimum = fit_model(bad_x_train, y_train)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    axs[0].set_title('Loss Surface - Before')

    # Loss surface - BEFORE
    CS = axs[0].contour(bad_bs[0, :], bad_ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Minimum point - BEFORE
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Initial random point
    axs[0].scatter(fixedb, fixedw, c='k')

    # Vertical cross section
    axs[0].plot([fixedb, fixedb], bad_w_range[[0, -1]], linestyle='--', c='r', linewidth=2)
    # Horizontal cross section
    axs[0].plot(bad_b_range[[0, -1]], [fixedw, fixedw], linestyle='--', c='k', linewidth=2)

    # Annotations
    axs[0].annotate('Minimum', xy=(.5, .35), c='k')
    axs[0].annotate('Random Start', xy=(fixedb - .6, fixedw - .3), c='k')

    axs[1].set_xlabel('b')
    axs[1].set_ylabel('w')
    axs[1].set_title('Loss Surface - After')

    # Loss surface - AFTER
    CS = axs[1].contour(bad_bs[0, :], bad_ws[:, 0], bad_all_losses, cmap=plt.cm.jet)
    axs[1].clabel(CS, inline=1, fontsize=10)
    # Minimum point - AFTER
    axs[1].scatter(bad_b_minimum, bad_w_minimum, c='k')
    # Initial random point
    axs[1].scatter(fixedb, fixedw, c='k')

    # Vertical cross section
    axs[1].plot([fixedb, fixedb], bad_w_range[[0, -1]], linestyle='--', c='r', linewidth=2)
    # Horizontal cross section
    axs[1].plot(bad_b_range[[0, -1]], [fixedw, fixedw], linestyle='--', c='k', linewidth=2)

    # Annotations
    axs[1].annotate('Minimum', xy=(.5, .35), c='k')
    axs[1].annotate('Random Start', xy=(fixedb - .6, fixedw - .3), c='k')

    fig.tight_layout()
    return fig, axs


def figure15(x_train, y_train, b, w, bad_bs, bad_ws, bad_x_train):
    bad_b_range = bad_bs[0, :]
    bad_w_range = bad_ws[:, 0]

    # So we recompute the surface for X_TRAIN using the new ranges
    all_predictions = np.apply_along_axis(func1d=lambda x: bad_bs + bad_ws * x, axis=1, arr=x_train)
    all_errors = (all_predictions - y_train.reshape(-1, 1, 1))
    all_losses = (all_errors ** 2).mean(axis=0)

    # Then we compute the surface for BAD_X_TRAIN using the new ranges
    bad_all_predictions = np.apply_along_axis(func1d=lambda x: bad_bs + bad_ws * x, axis=1, arr=bad_x_train)
    bad_all_errors = (bad_all_predictions - y_train.reshape(-1, 1, 1))
    bad_all_losses = (bad_all_errors ** 2).mean(axis=0)

    bad_b_idx, bad_w_idx, fixedb, fixedw = find_index(b, w, bad_bs, bad_ws)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_ylim([-.1, 15.1])
    axs[0].set_xlim([-1, 3.2])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Fixed: b = {:.2f}'.format(fixedb))

    # Loss curve for b, given fixed w - BEFORE
    axs[0].plot(bad_w_range, all_losses[:, bad_b_idx], c='r', linestyle='--', linewidth=1, label='Before')
    axs[0].plot([fixedw], [all_losses[bad_w_idx, bad_b_idx]], 'or')
    # Loss curve for b, given fixed w - AFTER
    axs[0].plot(bad_w_range, bad_all_losses[:, bad_b_idx], c='r', linestyle='--', linewidth=2, label='After')
    axs[0].plot([fixedw], [bad_all_losses[bad_w_idx, bad_b_idx]], 'or')

    axs[0].legend()

    axs[1].set_ylim([-.1, 15.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Fixed: w = {:.2f}'.format(fixedw))

    # Loss curve for w, given fixed b - BEFORE
    axs[1].plot(bad_b_range, all_losses[bad_w_idx, :], c='k', linestyle='--', linewidth=1, label='Before')
    axs[1].plot([fixedb], [all_losses[bad_w_idx, bad_b_idx]], 'ok')
    # Loss curve for w, given fixed b - AFTER
    axs[1].plot(bad_b_range, bad_all_losses[bad_w_idx, :], c='k', linestyle='--', linewidth=2, label='After')
    axs[1].plot([fixedb], [bad_all_losses[bad_w_idx, bad_b_idx]], 'ok')

    axs[1].legend()

    fig.tight_layout()
    return fig, axs


def figure17(x_train, y_train, scaled_bs, scaled_ws, bad_x_train, scaled_x_train):
    # So we recompute the surface for X_TRAIN using the new ranges
    all_predictions = np.apply_along_axis(func1d=lambda x: scaled_bs + scaled_ws * x, axis=1, arr=x_train)
    all_errors = (all_predictions - y_train.reshape(-1, 1, 1))
    all_losses = (all_errors ** 2).mean(axis=0)

    # So we recompute the surface for BAD_X_TRAIN using the new ranges
    bad_all_predictions = np.apply_along_axis(func1d=lambda x: scaled_bs + scaled_ws * x, axis=1, arr=bad_x_train)
    bad_all_errors = (bad_all_predictions - y_train.reshape(-1, 1, 1))
    bad_all_losses = (bad_all_errors ** 2).mean(axis=0)

    # Then we compute the surface for SCALED_X_TRAIN using the new ranges
    scaled_all_predictions = np.apply_along_axis(func1d=lambda x: scaled_bs + scaled_ws * x, axis=1, arr=scaled_x_train)
    scaled_all_errors = (scaled_all_predictions - y_train.reshape(-1, 1, 1))
    scaled_all_losses = (scaled_all_errors ** 2).mean(axis=0)

    b_minimum, w_minimum = fit_model(x_train, y_train)

    bad_b_minimum, bad_w_minimum = fit_model(bad_x_train, y_train)

    scaled_b_minimum, scaled_w_minimum = fit_model(scaled_x_train, y_train)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    axs[0].set_title('Loss Surface - Original')

    # Loss Surface - ORIGINAL
    CS = axs[0].contour(scaled_bs[0, :], scaled_ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Minimum point - ORIGINAL
    axs[0].scatter(b_minimum, w_minimum, c='k')

    # Annotations
    axs[0].annotate('Minimum', xy=(.3, 1.6), c='k')

    axs[1].set_xlabel('b')
    axs[1].set_ylabel('w')
    axs[1].set_title('Loss Surface - "Bad"')

    # Loss Surface - BAD
    CS = axs[1].contour(scaled_bs[0, :], scaled_ws[:, 0], bad_all_losses, cmap=plt.cm.jet)
    axs[1].clabel(CS, inline=1, fontsize=10)
    # Minimum point - BAD
    axs[1].scatter(bad_b_minimum, bad_w_minimum, c='k')

    # Annotations
    axs[1].annotate('Minimum', xy=(.3, -.25), c='k')

    axs[2].set_xlabel('b')
    axs[2].set_ylabel('w')
    axs[2].set_title('Loss Surface - Scaled')

    # Loss Surface - SCALED
    CS = axs[2].contour(scaled_bs[0, :], scaled_ws[:, 0], scaled_all_losses, cmap=plt.cm.jet)
    axs[2].clabel(CS, inline=1, fontsize=10)
    # Minimum point - SCALED
    axs[2].scatter(scaled_b_minimum, scaled_w_minimum, c='k')

    # Annotations
    axs[2].annotate('Minimum', xy=(1.3, .15), c='k')

    fig.tight_layout()
    return fig, axs


def figure18(x_train, y_train):
    b_minimum, w_minimum = fit_model(x_train, y_train)
    # Generates evenly spaced x feature
    x_range = np.linspace(0, 1, 101)
    # Computes yhat
    yhat_range = b_minimum + w_minimum * x_range

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0, 3.1])

    # Dataset
    ax.scatter(x_train, y_train)
    # Predictions
    ax.plot(x_range, yhat_range, label='Final model\'s predictions', c='k', linestyle='--')

    # Annotations
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b_minimum, w_minimum), xy=(.4, 1.5), c='k', rotation=34)
    ax.legend(loc=0)
    fig.tight_layout()
    return fig, ax
