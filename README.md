# Tweener Stickman（火柴人插帧）

给 [Krita](https://krita.org/) 用的火柴人中间帧实验项目（v0.1）。

当前可用链路：

**结构化姿态请求（JSON）→ 本机 HTTP 服务（Transformer）→ PNG → Krita 图层**

> 已知限制：MediaPipe **不能**从火柴人线稿 PNG 可靠提取关键点；任意 PNG 输入尚未打通。

## 已完成

- v0–v4：数据、MLP、Transformer、Raster 实验与评测
- v5a–v5e：Python API、CLI、批量、HTTP 服务
- v5f（最小）：Krita 插件 Demo（health + interpolate → 写入当前文档图层）

## 环境

- Python 3.11+
- PyTorch（建议 CUDA）
- 正式权重：`checkpoints/sequence_transformer_k3_best.pt`

本仓库开发机常用 conda 环境；通用安装示例：

    pip install torch numpy pillow opencv-python mediapipe

## 启动服务

在项目根目录：

    python -m server.app \
      --device cuda \
      --port 8765 \
      --checkpoint checkpoints/sequence_transformer_k3_best.pt

健康检查：

    curl -s http://127.0.0.1:8765/health

默认只监听 `127.0.0.1`，不要改成 `0.0.0.0`。

主要接口：

- `GET /health`
- `POST /v1/stickman/interpolate`
- `POST /v1/stickman/interpolate-batch`

示例请求：`krita_plugin/examples/tweener_request.json`

## Krita 插件

1. 复制到 Krita 的 `pykrita` 目录：
   - `krita_plugin/tweener_health.desktop`
   - `krita_plugin/tweener_health/`
2. macOS 常见路径：`~/Library/Application Support/krita/pykrita/`
3. 复制示例请求到桌面：

    cp krita_plugin/examples/tweener_request.json ~/Desktop/tweener_request.json

4. 确保本机可访问服务（同机直连；远程 GPU 可用端口转发）。
5. 重启 Krita，启用 **Tweener Health**。
6. 菜单：**工具 → 脚本**
   - `Tweener: Check Server`
   - `Tweener: Interpolate Demo`（写入当前文档图层）

插件默认 `http://127.0.0.1:8766`；若服务在 `8765`，请改 `__init__.py` 里的 `BASE_URL`。

## 仓库说明

| 路径 | 说明 |
| --- | --- |
| `stickman/` | 模型、数据、渲染与请求契约 |
| `server/` | 本机 HTTP 服务 |
| `krita_plugin/` | Krita 最小插件 |
| `checkpoints/` | 正式 Transformer 权重 |
| `infer_stickman*.py` 等 | CLI / 评测 / 烟雾测试 |
| `火柴人插帧.md` | 设计与阶段说明 |
| `交接_20260903.md` | 当前交接与断点 |

未纳入版本库（见 `.gitignore`）：`data/`、`runs/`、Krita AppImage。

## 下一步（未做）

- 时间轴多帧动画闭环
- 自研「渲染图 → 17 关键点」以支持 PNG 输入
- 任意 `n` / 连续时间插帧
- 单元测试

## License

MIT。详见仓库根目录 `LICENSE`。
