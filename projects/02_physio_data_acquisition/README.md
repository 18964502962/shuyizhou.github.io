# Multi-person Physiological Signal Data Acquisition System

分布式多人生理信号采集系统，实现视频录制与血氧仪数据的网络同步采集，支持多设备协同工作。

## 📋 项目概述

开发了一套分布式数据采集系统，用于多受试者生理信号研究。系统实现了视频录制与血氧仪数据的网络同步采集，支持多设备协同工作，同步精度优于 100ms。

**系统特点：**
- UDP 时间同步协议，精度 < 100ms
- FFmpeg 进程调度优化，硬件编码支持
- 60Hz 采样率 PPG/HR/SPO2 实时记录
- 已应用于 30 名实验人员，丢帧率 < 0.1%

## 🛠 技术栈

| 类别 | 技术 |
|------|------|
| 编程语言 | Python 3.9+ |
| 视频录制 | FFmpeg, OpenCV |
| 网络通信 | UDP Multicast, Socket Programming |
| 硬件接口 | HID (Human Interface Device) |
| 信号处理 | NumPy, SciPy |
| 并发控制 | Threading, asyncio |
| 数据处理 | Pandas |

## 📁 项目结构

```
physio-data-acquisition/
├── src/
│   ├── core/
│   │   ├── sync_server.py     # 时间同步服务器
│   │   ├── sync_client.py     # 时间同步客户端
│   │   ��── clock_sync.py      # 时钟校准算法
│   ├── capture/
│   │   ├── video_recorder.py  # 视频录制模块
│   │   ├── ppg_sensor.py      # 血氧仪数据采集
│   │   └── data_sync.py       # 数据同步模块
│   ├── network/
│   │   ├── udp_multicast.py   # UDP 多播通信
│   │   └── command_bus.py     # 指令总线
│   └── utils/
│       └── logger.py          # 日志工具
├── gui/
│   └── main_window.py         # Tkinter 主界面
├── tests/
│   ├── test_sync.py
│   ├── test_capture.py
│   └── test_network.py
├── scripts/
│   ├── run_experiment.py      # 实验运行脚本
│   └── analyze_data.py        # 数据分析脚本
├── config/
│   └── default.yaml           # 默认配置
├── data/                      # 数据存储目录
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 环境配置

```bash
pip install -r requirements.txt
```

**依赖：**
- Python 3.9+
- FFmpeg (系统安装)
- opencv-python
- numpy
- pandas
- pyserial (用于 HID 设备)
- tkinter (GUI)

### 硬件准备

- 网络摄像头 (Webcam)
- 血氧仪 (支持 USB/HID 接口)
- 多台电脑组成实验网络

### 系统部署

```bash
# 1. 启动时间同步服务器（主节点）
python src/core/sync_server.py --port 5000

# 2. 在从节点启动客户端
python src/core/sync_client.py --server 192.168.1.100

# 3. 启动数据采集 GUI
python gui/main_window.py
```

## 🔧 核心功能

### 时间同步

采用改良的 UDP 时间同步协议：

```python
# 客户端同步示例
sync = ClockSync(server_ip="192.168.1.100", port=5000)
sync.sync()  # 启动同步
print(f"Offset: {sync.offset_us} us")  # 偏移量
```

同步精度可达 **< 100μs**。

### 视频采集

```python
from src.capture.video_recorder import VideoRecorder

recorder = VideoRecorder(
    camera_id=0,
    resolution=(1280, 720),
    fps=30,
    output_path="./data/exp001.mp4"
)
recorder.start()
# ... 采集数据 ...
recorder.stop()
```

**优化特性：**
- FFmpeg 硬件编码支持 (NVENC)
- 防丢帧缓冲区管理
- 自动回滚机制

### 血氧仪数据采集

```python
from src.capture.ppg_sensor import PPGSensor

sensor = PPGSensor(port="COM3", baudrate=115200)
sensor.start()

for packet in sensor:
    hr = packet.heart_rate      # 心率
    spo2 = packet.spo2          # 血氧饱和度
    timestamp = packet.timestamp # 精确时间戳
```

## 📊 实验数据

### 性能指标

| 指标 | 值 |
|------|-----|
| 同步精度 | < 100μs |
| 视频帧率 | 30 fps |
| PPG 采样率 | 60 Hz |
| 丢帧率 | < 0.1% |
| 最大设备数 | 8 台 |

### 实验应用

系统已成功应用于：
- 多人群体呼吸率研究
- 睡眠监测实验
- 压力状态评估

**实验规模：** 30 名受试者，累计采集数据超过 100 小时

## 🔍 故障排除

### 常见问题

**Q: 时间同步失败**
- 检查网络连接
- 确认防火墙允许 UDP 端口

**Q: 视频丢帧**
- 降低分辨率或帧率
- 检查硬盘写入速度

**Q: 血氧仪无法识别**
- 检查 COM 端口设置
- 确认驱动已安装

### 日志查看

```bash
# 实时查看日志
tail -f logs/experiment.log
```

## 📝 引用

```bibtex
@misc{zhou2025multi,
  title={Multi-Person Physiological Signal Acquisition System},
  author={Zhou Shuyi},
  year={2025},
  note={Research Project, Macau University of Science and Technology}
}
```

## 📄 许可证

MIT License

---

> 本项目是澳门理工大学大数据与物联网硕士课程的研究项目，由周舒怡开发。
