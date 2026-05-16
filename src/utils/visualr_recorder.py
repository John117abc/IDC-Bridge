import os
import imageio
from datetime import datetime

from gpudrive.visualize.utils import img_from_fig

class VisualRecorder:
    """
    可视化记录器，用于保存仿真状态的帧并生成 GIF
    """
    
    def __init__(self, num_worlds, save_dir="/workspace/idc/gifs", fps=5):
        """
        初始化记录器
        
        参数:
            num_worlds: 环境数量
            save_dir: 保存 GIF 的目录
            fps: GIF 的帧率
        """
        self.num_worlds = num_worlds
        self.save_dir = save_dir
        self.fps = fps
        self.frames = {f"env_{i}": [] for i in range(num_worlds)}
        self.step_count = 0
        
    def record(self, env, epoch, step, zoom_radius=70,custom_fps=None):
        """
        记录当前状态（当 step 是 5 的倍数时）
        
        参数:
            env: 环境对象，包含 vis.plot_simulator_state 方法
            epoch: 当前训练轮次（时间步）
            step: 当前全局步数
            zoom_radius: 缩放半径
        """
        fps = custom_fps or self.fps
        if step % fps == 0:
            imgs = env.vis.plot_simulator_state(
                env_indices=list(range(self.num_worlds)),
                time_steps=[epoch] * self.num_worlds,
                zoom_radius=zoom_radius,
            )
            for i in range(self.num_worlds):
                self.frames[f"env_{i}"].append(img_from_fig(imgs[i]))
    
    def save_all_gifs(self, custom_save_dir=None, custom_fps=None):
        """
        将所有环境的帧保存为单独的 GIF 文件，自动创建日期子目录并带时间戳。
        """
        save_dir = custom_save_dir or self.save_dir
        fps = custom_fps or self.fps

        now = datetime.now()
        date_dir = os.path.join(save_dir, now.strftime('%Y%m%d'))
        os.makedirs(date_dir, exist_ok=True)

        ts = now.strftime('%H%M%S')
        print(f"开始保存到: {date_dir}")

        for env_name, frame_list in self.frames.items():
            path = os.path.join(date_dir, f"rollout_{env_name}_{ts}.gif")
            if frame_list:
                imageio.mimsave(path, frame_list, fps=fps)
                print(f"已保存: {path}")
            else:
                print(f"警告: {env_name} 没有记录任何帧")

        print("所有 GIF 保存完成")
    
    def reset(self):
        """重置 frames 字典，清空所有已记录的帧"""
        self.frames = {f"env_{i}": [] for i in range(self.num_worlds)}
        self.step_count = 0
        
    def get_frame_count(self):
        """获取每个环境记录的帧数"""
        return {env_name: len(frame_list) for env_name, frame_list in self.frames.items()}
    
    def __len__(self):
        """返回总帧数"""
        return sum(len(frame_list) for frame_list in self.frames.values())
    


if __name__ == "__main__":
    # 初始化记录器
    recorder = VisualRecorder(
        num_worlds=args.num_worlds,
        save_dir="/workspace/idc/gifs",
        fps=5
    )

    # 在训练循环中使用
    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            # ... 训练代码 ...
            
            # 记录当前状态
            recorder.record(env, epoch, step)

    # 训练结束后保存所有 GIF
    recorder.save_all_gifs()

    # 可选：查看记录的帧数
    print(recorder.get_frame_count())
    print(f"总帧数: {len(recorder)}")