# Kaggle production demo

## Phạm vi

Kaggle demo qualification cung cấp một Kaggle notebook mỏng để tái tạo VoxCPM2 API runtime đã qualification từ một workspace hoàn toàn mới.

Notebook:

```text
notebooks/kaggle-production-demo.ipynb
```

Notebook chỉ làm orchestration. Runtime vẫn nằm trong package của dự án và `scripts/kaggle_production_demo.py`; notebook không sao chép scheduler, worker, backend hay API implementation.

## Thiết lập trên Kaggle

Để chạy đầy đủ matrix Kaggle demo qualification:

1. Tạo một Kaggle Notebook mới.
2. Chọn **Accelerator: GPU T4 x2**.
3. Attach Kaggle model **dangkhoa2016/openbmb-voxcpm2**.
4. Bật Internet cho bước bootstrap source/dependency.
5. Upload hoặc import `notebooks/kaggle-production-demo.ipynb`.
6. Trước **Run All**, chọn execution profile bằng biến `VOXCPM_DEMO_PROFILE`. Nếu không đặt, notebook dùng `auto`.

Các canonical profile:

```text
cpu
cuda-single
cuda-replica
auto
```

Một notebook T4x2 có thể kiểm tra cả bốn profile. Profile `cpu` buộc runtime dùng CPU dù accelerator của Kaggle vẫn đang bật.

## Fresh-workspace contract

Mỗi lần Run All sẽ xóa và tạo lại workspace demo đã chọn. Sau đó notebook:

1. clone project vào workspace mới;
2. checkout `VOXCPM_DEMO_REF`;
3. đọc upstream source commit đã pin từ `provenance/voxcpm-source-lock.json`;
4. clone và checkout đúng upstream commit đó;
5. tạo virtual environment mới dùng lại CUDA/PyTorch stack có sẵn của Kaggle;
6. cài project/API và upstream dependencies;
7. verify Kaggle model đã attach trong offline mode;
8. khởi động execution profile đã chọn;
9. chờ `/readyz`;
10. gửi real authenticated `POST /v1/tts`;
11. kiểm tra WAV mono 48 kHz trả về;
12. shutdown server và xác nhận API port cùng GPU process state trở về baseline.

Notebook không đọc hay tái sử dụng virtual environment, log, generated audio hoặc project checkout của T4x2 qualification/execution-profile qualification.

## Ranh giới network của source và model

Internet chỉ được dùng trong bootstrap để clone public source và cài Python dependencies.

Sau bootstrap, model runtime chạy local/offline:

```text
VOXCPM_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_ENDPOINT=http://127.0.0.1:9
HTTP_PROXY=http://127.0.0.1:9
HTTPS_PROXY=http://127.0.0.1:9
ALL_PROXY=http://127.0.0.1:9
```

Loopback vẫn được cho phép để notebook gọi FastAPI server local.

Model được discovery qua Kaggle deployment adapter, không hard-code model leaf vào reusable runtime code.

## Ref có thể tái lập

Khi qualification hoặc review, nên dùng project SHA bất biến:

```bash
export VOXCPM_DEMO_REF=<project-commit-sha>
```

Public demo có thể tái lập được bind vào release tag bất biến `v1.0.0`, không bind vào branch thay đổi liên tục.

Dùng `main` thuận tiện khi phát triển nhưng không tạo evidence binding bất biến tương đương.

## Real Kaggle qualification matrix

Cả bốn profile đã được chạy qua notebook, mỗi profile dùng một fresh workspace riêng, trên cùng một Kaggle T4x2 session.

| Profile | Topology resolve | Ready workers | Startup | TTS request | Kết quả |
| --- | --- | ---: | ---: | ---: | --- |
| `auto` | CUDA GPU 0,1 / 2 worker | 2 | 81.053 s | 5.994 s | HTTP 200 |
| `cuda-single` | CUDA GPU 0 / 1 worker | 1 | 33.041 s | 3.933 s | HTTP 200 |
| `cuda-replica` | CUDA GPU 0,1 / 2 worker | 2 | 62.572 s | 5.360 s | HTTP 200 |
| `cpu` | CPU / 1 worker | 1 | 28.586 s | 28.886 s | HTTP 200 |

Mọi response đều là WAV mono 48 kHz hợp lệ. Các timing trên chỉ là quan sát của qualification run, không phải benchmark guarantee. Benchmark chính thức thuộc qualification sau.

Trong mọi run:

- model identity là `openbmb/VoxCPM2`;
- model revision là `32279effe8c19989596f05d353d1447f51d9e915`;
- mirror config SHA-256 là `405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c`;
- runtime model acquisition ở offline/blackholed mode;
- cleanup đưa GPU compute-process state về baseline;
- port 8090 đã đóng;
- cgroup OOM và OOM-kill counters luôn bằng 0.

## Evidence files

Runner ghi structured JSON evidence dưới fresh workspace:

```text
<workspace>/evidence/kaggle-<profile>-evidence.json
```

Evidence/log runtime không được commit. Public qualification summary nằm tại:

```text
`provenance/QUALIFICATION.md`
```

## Ranh giới

Tài liệu này qualification fresh Kaggle production-style demo và Run All reproducibility cho bốn profile execution-profile qualification trên môi trường đã đo.

Kaggle demo qualification không qualification:

- permanent production service;
- autoscaling;
- automatic worker respawn;
- tensor parallelism hoặc model sharding;
- mọi loại Kaggle accelerator;
- benchmark guarantee;
- model training hoặc fine-tuning.
