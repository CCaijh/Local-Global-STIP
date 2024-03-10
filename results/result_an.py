import os,sys
import matplotlib.pyplot as plt

def size(a):
    if a == "weights":
        return -1

    x = int(a)
    return x


def DrawData(itr, data, title):
    x = range(1, len(itr) + 1)
    plt.plot(x, data,
             marker='o',  # 实心圆
             markersize=5,  # 实心圆大小
             markerfacecolor='red',  # 曲线颜色
             alpha=0.5, linestyle='--', linewidth=1
             )
    # 设置网格
    plt.grid(linewidth=1, alpha=0.7)
    plt.title(title)
    # 保存图片
    #plt.savefig('avg_mse.png')
    # 显示
    plt.show()

#找到所有迭代结果，一次迭代一个目录，性能数据存在data.txt
result_dir = "sf/20221018154529"
dir_list = next(os.walk(result_dir))[1]
dir_list.sort(key=size)

#提取结果
avg_mse = []
avg_psnr = []
avg_ssim = []
avg_lpips = []
itr = []
for dir in dir_list:
    if dir == "weights":
        continue

    one_itr_data_file = os.path.join(result_dir,dir,"data.txt")
    itr_data = open(one_itr_data_file, mode="r").readlines()
    avg_mse.append(float(itr_data[0]))
    avg_psnr.append(float(itr_data[2]))
    avg_ssim.append(float(itr_data[4]))
    avg_lpips.append(float(itr_data[6]))
    itr.append(int(dir))
    print("{} : {:.8f} {:.8f} {:.8f} {:.8f}".format(dir,avg_mse[-1],avg_psnr[-1],avg_ssim[-1],avg_lpips[-1]))

#绘制
DrawData(itr, avg_mse, "Avg MSE")
DrawData(itr, avg_psnr, "Avg PSNR")
DrawData(itr, avg_ssim, "Avg SSIM")
DrawData(itr, avg_lpips, "Avg LPIPS")





