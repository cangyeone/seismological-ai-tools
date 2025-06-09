# seismic-event-detection 使用说明

该目录包含用于地震波形震相拾取及事件关联的脚本，并提供多种预训练模型。

## 主要内容
- `pickers/`：已训练好的 `.jit` 与 `.onnx` 模型，可直接推断任意长度数据。
- `ckpt/`：PyTorch `.pt` 权重，用于迁移学习或重新训练。
- `picker.py`：遍历目录自动拾取三分量 100Hz 波形的震相。
- `fastlinker.py`、`gammalinker.py`、`reallinker.py`：三种震源关联算法实现。
- `config/`：拾取与关联的参数配置示例。

## 快速开始
1. 安装 `obspy`、`torch`、`onnxruntime` 等依赖。
2. 准备三分量 100Hz 波形数据，文件名格式示例 `NET.STA.LOC.CHN.XXXX.mseed`。
3. 运行下列命令进行批量拾取：
   ```bash
   python picker.py -i /path/to/waves -o result -m pickers/rnn.jit -d cuda:0
   ```
   将生成 `result.txt`（拾取结果）、`result.log`（处理日志）及 `result.err`（异常数据）。

### 拾取结果格式
```
#path/to/file
phase,relative_time,confidence,absolute_time,SNR,AMP,station,extra
```

## 震源关联
对拾取结果进行事件关联，可选择以下脚本：
- `fastlinker.py`：基于 LPPN 的快速关联。
- `reallinker.py`：REAL 算法实现。
- `gammalinker.py`：GaMMA 方法实现。

示例命令：
```bash
python fastlinker.py -i result.txt -o events.txt -s station.csv
```
其中 `station.csv` 格式：
```
network station LOC longitude latitude elevation
SC AXX 00 110.00 38.00 1000.00
```

## 其他说明
- `makejit.*.py` 与 `makeonnx.*.py` 为模型导出示例，可按需生成新的 jit/onnx 模型。
- 更多模型对比与详细参数说明见同目录 `README.md`。

许可证：GPLv3
