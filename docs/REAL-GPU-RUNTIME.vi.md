# Qualification runtime single-GPU thật

## Trạng thái

M5 qualification backend VoxCPM2 thật cho TTS chuẩn trên đúng một GPU CUDA được chọn tường minh.

Host nhìn thấy hai Tesla T4 trước khi isolate. Process M5 canonical dùng CUDA_VISIBLE_DEVICES=0, vì vậy runtime của dự án chỉ thấy đúng một thiết bị CUDA.

~~~text
VOXCPM_DEVICE=cuda
VOXCPM_GPU_DEVICES=0
VOXCPM_WORKERS=1
effective_device=cuda
selected_gpu_indices=[0]
worker_count=1
upstream_device=cuda:0
~~~

Đây là qualification single-GPU, không phải qualification T4x2 hay multi-GPU.

## Authority

~~~text
M5 starting SHA        ec28285ca5dcb50e7f45ccb1984e19529fb21f88
M5 implementation SHA  d1e91b99c0457401c2cbf0c46eeb576aa08350ee
implementation CI      36214646774 (PASS)
VERSION                 1.0.0
upstream commit         f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69
installed voxcpm        2.0.3.post33+gf772e498a
~~~
Model là openbmb/VoxCPM2 tại revision 32279effe8c19989596f05d353d1447f51d9e915. Config SHA-256 là 405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c. Đường dẫn model được chủ động loại khỏi report public; M2 bind run vào mirror local đã attach với zero remote acquisition.

## Môi trường runtime

~~~text
Python                  3.12.13
PyTorch                 2.10.0+cu128
torchaudio              2.10.0+cu128
CUDA runtime            12.8
CUDA driver API         13.0
GPU                     Tesla T4
runtime visible GPUs    1
GPU memory              15,636,037,632 B (14.562 GiB)
compute capability      7.5
configured dtype        bfloat16
upstream optimize       false
denoiser                false
~~~

Dtype checkpoint không bị thay đổi. M5 không thêm override FP16 và không qualification torch.compile.

Kaggle image này không có binary nvidia-smi. Vì vậy identity GPU và allocator memory được ghi qua PyTorch, còn driver API được query qua libcuda.so.1.

## Các run thật

| Run | Exit | Load | Synthesize | Wall | Peak allocated VRAM | Peak reserved VRAM | Host peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Load-only | 0 | 45.283 s | n/a | 48.35 s | 4.946 GiB | 5.004 GiB | 11.015 GiB |
| Canonical TTS | 0 | 27.080 s | 7.138 s | 37.63 s | 5.194 GiB | 5.363 GiB | 11.003 GiB |
| Offline-proof TTS | 0 | 27.168 s | 5.611 s | 35.86 s | 5.184 GiB | 5.363 GiB | 11.033 GiB |
Text canonical: "Xin chào từ VoxCPM2."

Canonical synthesis tạo 2.40 giây audio trong 7.138 giây, tương ứng real-time factor đo được khoảng 2.97. Đây là một phép đo trên T4, không phải cam kết latency và không phải tuyên bố real-time.

## Device placement

~~~text
parameter_count           888
parameter_device_types    ["cuda"]
buffer_count              10
buffer_device_types       ["cuda"]
expected_device           "cuda"
all_on_expected_device    true
cpu_only                  false
~~~

Backend map một runtime GPU đã chọn thành cuda:0. Real backend từ chối plan có nhiều GPU bằng multi_gpu_not_qualified; M5 không âm thầm fallback về CPU.

## Kết quả WAV

~~~text
channels              1
sample width          2 bytes (PCM16)
sample rate           48,000 Hz
frames                115,200
duration              2.40 s
file size             230,444 B
SHA-256               8a5b4641f6ddc9b08bc733baa5071a1fc851ab71f13e3e8ab132bd6b649515a2
peak abs PCM16        15,446
RMS PCM16             2,034.97
nonzero samples       114,130 / 115,200
~~~

Project validator và lần đọc độc lập bằng Python wave đồng ý về cấu trúc WAV. Tín hiệu không rỗng và không suy biến. M5 không thực hiện nghe đánh giá, ASR transcription, chấm speaker fidelity hay qualification chất lượng audio chủ quan.
Sampler upstream có tính stochastic. Offline-proof run tạo WAV hợp lệ khác digest và duration; M5 không tuyên bố tái lập byte-for-byte.

## Offline proof

Một full TTS run thứ hai dùng HF_HOME rỗng mới, các offline flag và endpoint/proxy loopback blackhole.

~~~text
exit code                       0
local_files_only                true
HF_HOME files after run         0
network/proxy/download errors   0
valid WAV                       yes
~~~

Điều này chứng minh run đã đo dùng model local được attach mà không cần remote model acquisition. Phương pháp dùng endpoint/proxy blackhole thay vì network namespace riêng.

## Bằng chứng resource

Với load-only, canonical TTS và offline-proof TTS, delta cgroup đều bằng zero cho high, low, max, oom, oom_kill và oom_group_kill. Tất cả process đều exit zero. Không quan sát thấy GPU OOM, host OOM, signal kill hay allocation failure.

## Lưu ý qualification environment

Qualification venv kế thừa CUDA stack hệ thống Kaggle bằng --system-site-packages. Tạo venv chuẩn thất bại ở ensurepip, nên M5 dùng --without-pip --system-site-packages và tái sử dụng module pip hệ thống qua interpreter của venv.

Global pip check báo các conflict có sẵn không liên quan của Kaggle. Import VoxCPM đúng pinned commit PASS, toàn bộ dependency non-torch upstream khai báo đều hiện diện và CUDA PyTorch identity không đổi. Các conflict này là limitation môi trường, không phải lỗi runtime VoxCPM.
## M5 gates

~~~text
M5_PRE_IMPLEMENTATION_AUDIT=PASS
M5_SINGLE_GPU_PLAN_GATE=PASS
M5_CUDA_DEVICE_MAPPING_GATE=PASS
M5_REAL_GPU_LOAD_GATE=PASS
M5_REAL_GPU_TTS_GATE=PASS
M5_GPU_DEVICE_PLACEMENT_GATE=PASS
M5_GPU_MEMORY_MEASUREMENT=PASS
M5_WAV_VALIDATION_GATE=PASS
M5_OFFLINE_LOCAL_MODEL_GATE=PASS
M5_NO_CPU_FALLBACK_GATE=PASS
M5_M0_M4_REGRESSION_GATE=PASS
M5_REAL_SINGLE_GPU_RUNTIME=PASS
M5_INVESTIGATION_COMPLETENESS=PASS
M5_SINGLE_T4_RUNTIME=PASS
~~~

## Những gì M5 không qualification

M5 không qualification T4x2/multi-GPU, nhiều worker, scheduler, torch.compile, override FP16, quantization, streaming, voice design, cloning, continuation, hành vi API, cam kết real-time, hỗ trợ phổ quát cho GPU 16 GB, toàn bộ NVIDIA GPU, production readiness hay chất lượng audio chủ quan.
