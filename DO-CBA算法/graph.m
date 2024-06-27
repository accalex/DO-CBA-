x = -20:2:18;  % 横坐标范围为-20至18，每2一个点
y_ticks = 0:0.1:1;  % 纵坐标范围为0-1，每0.1一个点

% 创建图形
figure;
hold on;

% 绘制每一列数据的折线图
for i = 1:size(data, 2)
    plot(x, data(:, i), '-o', 'LineWidth', 2, 'DisplayName', ['Series ' num2str(i)]);
end

% 设置图例
legend show;

% 设置坐标轴
set(gca, 'XTick', x);
set(gca, 'YTick', y_ticks);

% 设置字体
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('信噪比 / dB', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('准确率', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'bold');

% 获取图例对象并设置字体
lgd = legend;
set(lgd, 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'bold');

% 设置图形标题和标签的中文字体
title('图形标题', 'FontName', 'SimSun', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('信噪比 / dB', 'FontName', 'SimSun', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('准确率', 'FontName', 'SimSun', 'FontSize', 10, 'FontWeight', 'bold');

% 显示网格
grid on;

% 保持图形
hold off;