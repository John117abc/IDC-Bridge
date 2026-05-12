```
idc-bridge/
├── config.yaml                 # 配置文件（路径、超参数）
├── requirements.txt            # 依赖
├── README.md                   # 说明
│
├── scripts/                    # 可执行脚本
│   ├── extract_data.py         # 步骤1：提取Waymo数据→HDF5
│   ├── pretrain.py             # 步骤2：预训练（BC+OCP）
│   └── export_weights.py       # 步骤3：导出CARLA可用权重
│
├── src/                        # 核心代码
│   ├── waymo_loader.py         # Waymo数据读取（唯一import tf的地方）
│   ├── state_extractor.py      # 核心：Waymo→OCP状态向量
│   ├── dataset.py              # PyTorch Dataset
│   ├── ocp_agent.py            # OCP智能体（从原项目复制，去掉env依赖）
│   ├── models/                 # 网络模型（从原项目复制）
│   │   ├── actor_critic.py
│   │   └── bicycle.py
│   └── utils.py                # 日志、保存/加载
│
├── data/                       # 数据目录 (gitignore)
│   ├── raw/                    # 下载的.tfrecord
│   └── processed/              # 提取后的.h5
│
├── checkpoints/                # 权重 (gitignore)
│   └── idc_pretrained.pt
│
└── logs/                       # 日志 (gitignore)
```