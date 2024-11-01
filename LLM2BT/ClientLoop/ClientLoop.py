
import queue
import carla

class ClientLoop(object):
    def __init__(self):
        self.carla_client = None

        try:
            self.carla_client = carla.Client('127.0.0.1', 2000)
        finally:
            if self.carla_client is None:
                print('Connecting failed')

    def run(self, input_queue):
        while True:
            try:
                user_input = input_queue.get_nowait() # 等待用户输入，超时为 1 秒
                #print(f"Received input:\n{user_input}")
                # 在这里处理用户输入
                print('')


            except queue.Empty:
                # 如果队列为空，继续等待
                pass

            # 主循环中的其他代码
            self.carla_client.get_world().wait_for_tick()


