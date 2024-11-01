import tkinter as tk
import threading
import queue

class TextInputPopup:
    def __init__(self, title="Text Input", input_queue=None):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("400x300")  # 设置窗口大小

        self.label = tk.Label(self.root, text="Enter text:")
        self.label.pack(pady=10)

        # 使用 Text 控件代替 Entry 控件，以支持多行文本输入
        self.text_box = tk.Text(self.root, height=10, width=40)
        self.text_box.pack(pady=10)

        self.submit_button = tk.Button(self.root, text="Submit", command=self.on_submit)
        self.submit_button.pack(pady=10)

        self.input_queue = input_queue or queue.Queue()
        self.user_input = None

    def on_submit(self):
        self.user_input = self.text_box.get("1.0", tk.END).strip()  # 获取多行文本
        self.input_queue.put(self.user_input)
        self.root.quit()

    def show(self):
        self.root.mainloop()
        return self.user_input


if __name__ == "__main__":
    input_queue = queue.Queue()

    # 启动子线程，运行一个客户端循环
    from ClientLoop.ClientLoop import ClientLoop
    cl = ClientLoop()
    main_loop_thread = threading.Thread(target=cl.run, args=(input_queue,))
    main_loop_thread.start()

    # 在主线程监听弹窗监听用户文本输入
    popup = TextInputPopup(input_queue=input_queue)
    while True:
        popup.show()
