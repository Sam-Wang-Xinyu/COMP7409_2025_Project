import tkinter as tk
import requests

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def fetch_prediction():

    # 收集被选中的勾选框的值
    selected_options = []
    if var1.get():
        selected_options.append("Momentum Feature")
    if var2.get():
        selected_options.append("Volatility Feature")
    if var3.get():
        selected_options.append("Trend Feature")
    if var4.get():
        selected_options.append("Volume Feature")
    
    
    # 将选项作为参数发送HTTP请求
    try:
        response = requests.get('http://127.0.0.1:4567/predict', params={'options': selected_options})
        data = response.json()        
        # 解析结果并分行输出
        result_text = ""
        if isinstance(data, dict):
            for key, value in data.items():
                if key != "price_data":
                    result_text += f"{key}: {value}\n"
        elif isinstance(data, list):
            result_text = "\n".join(data)

        result_var.set(result_text.strip())  # 去除末尾空白
        # 创建并显示折线图
        create_line_plot(data["price_data"])


    except Exception as e:
        result_var.set(f"Error fetching data: {str(e)}")
        
    
canvas = None 
def create_line_plot(data):
    global canvas
    # 示例数据
    x = [i for i in range(len(data))]
    y = data

    # 创建图形
    fig = Figure(figsize=(5, 4), dpi=100)
    plot = fig.add_subplot()
    # 绘制蓝色线条（除最后一个点外）
    plot.plot(x[:-1], y[:-1], color='blue', marker='o', label='Past Price')
    # 绘制最后一个点（红色）
    plot.plot(x[-2:], y[-2:], color='red', marker='o', linestyle='--',label='Avg Predict Price')

    # 设置标题和标签
    plot.set_title("stock trend")
    plot.set_xlabel("Date-axis")
    plot.set_ylabel("Price-axis")
    # 添加图例
    plot.legend()

    if canvas: canvas.get_tk_widget().pack_forget()  # remove previous image
    # 将图形嵌入 Tkinter 窗口
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
def center_window(width, height):
    # 获取屏幕尺寸
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # 计算居中位置
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)

    # 设置窗口大小和位置
    root.geometry(f"{width}x{height}+{x}+{y}")

# 设置主窗口
root = tk.Tk()
root.title("Prediction Result")
center_window(500, 600)  # 设置窗口大小并居中

# 创建结果显示标签
result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, wraplength=300)
result_label.pack(pady=20)

# 创建勾选框变量
var1 = tk.BooleanVar()
var2 = tk.BooleanVar()
var3 = tk.BooleanVar()
var4 = tk.BooleanVar()


# 创建一个框架以居中勾选框
frame = tk.Frame(root)
frame.pack(expand=False)

# 创建勾选框
check1 = tk.Checkbutton(root, text="Add Momentum Feature or Not", variable=var1)
check1.pack(anchor=tk.W, padx=(75, 0))
check2 = tk.Checkbutton(root, text="Add Volatility Feature or Not", variable=var2)
check2.pack(anchor=tk.W, padx=(75, 0))
check3 = tk.Checkbutton(root, text="Add Trend Feature or Not", variable=var3)
check3.pack(anchor=tk.W, padx=(75, 0))
check4 = tk.Checkbutton(root, text="Add Volume Feature or Not", variable=var4)
check4.pack(anchor=tk.W, padx=(75, 0))

# 创建按钮以获取预测
fetch_button = tk.Button(root, text="Fetch Prediction", command=fetch_prediction)
fetch_button.pack(pady=10)


# 启动GUI循环
root.mainloop()

