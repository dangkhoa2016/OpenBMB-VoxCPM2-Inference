# Runtime CPU thật

Tài liệu này ghi lại điều M4 thực sự đo được khi bản triển khai upstream VoxCPM2 đã ghim được nạp và
chạy trên CPU với một model cục bộ thật. Đây là báo cáo điều tra, không phải cam kết hỗ trợ.

Hãy đọc trước:

- Việc nạp model cục bộ thật trên CPU và tổng hợp TTS chuẩn thật trên CPU đều thành công trên host này.
- Dung lượng cư trú đỉnh đo được là `10.819 GiB`. Con số đó **không phải** là tuyên bố hỗ trợ 16 GB.
  Xem [Trạng thái 16 GB](#trạng-thái-16-gb).
- Tổng hợp trên CPU chậm hơn thời gian thực khoảng `48 lần`. Đây **không phải** là tuyên bố CPU thời gian
  thực.
- Báo cáo là bằng chứng riêng cho môi trường đó. Nó không định danh GPU, T4, thiết kế giọng, nhân bản
  giọng, nối tiếp âm thanh, phát trực tuyến, batching, hay cách ly worker.

## Phạm vi

M4 bổ sung đường thực thi thật đầu tiên đặt sau ranh giới backend của M3:

| Hạng mục | Trạng thái |
| --- | --- |
| Nạp model cục bộ thật trên CPU | Đủ điều kiện trên host này |
| TTS chuẩn thật xuất WAV | Đủ điều kiện trên host này |
| Ghi và kiểm tra WAV | Đủ điều kiện |
| Chứng minh không dùng CUDA cho lần chạy chuẩn | Đủ điều kiện |
| Thiết kế giọng / nhân bản / nối tiếp | Chưa triển khai (M5 trở đi) |
| Phát trực tuyến | Chưa triển khai (M6 trở đi) |
| GPU / T4 / runtime một GPU | Chưa đủ điều kiện (M5) |
| API, bộ lập lịch, cách ly worker, đa GPU | Chưa bắt đầu (M7-M10) |

## Kiến trúc

`voxcpm_runtime` giữ một ranh giới cứng giữa lõi hợp đồng không phụ thuộc và bản triển khai thật:

```text
voxcpm_runtime/backend.py          chỉ hợp đồng, không import upstream
voxcpm_runtime/backend_types.py    chỉ kiểu dữ liệu thuộc dự án
voxcpm_runtime/errors.py           phân cấp lỗi đã chuẩn hóa
voxcpm_runtime/fake_backend.py     xác định, hermetic, không cần model
voxcpm_runtime/pytorch_backend.py  module DUY NHẤT chạm tới upstream VoxCPM
voxcpm_runtime/wav_io.py           bộ ghi và kiểm tra WAV PCM16 thuộc dự án
voxcpm_runtime/runtime_metrics.py  telemetry RSS, thiết bị, CUDA, phiên bản
```

`pytorch_backend.py` import gói upstream đúng một lần, thông qua
`importlib.import_module("voxcpm")`, bên trong đường nạp của nó. Không có import cấp module nào của
`torch`, `numpy`, `transformers`, hay `voxcpm` ở bất kỳ đâu trong `voxcpm_runtime`. Điều này được CI
kiểm tra bằng một gate riêng thay vì để dựa vào quy ước.

Gói của dự án không khai báo phụ thuộc runtime nào. Torch, torchaudio, transformers và gói upstream
nằm trong một môi trường định danh riêng, nên việc import dự án này không bao giờ kéo theo cả ngăn
xếp ML.

## Chính sách tối ưu hóa CPU

M4 buộc truyền `optimize=False` lên upstream khi chạy CPU. Đường `optimize=True` của upstream cần
ngăn xếp `torch.compile` và một cuộc điều tra biên dịch thực sự thuộc milestone sau, nên milestone
này không bao giờ bật nó.

Một yêu cầu tối ưu hóa không phải CPU bị từ chối có chủ đích thay vì bị hạ xuống âm thầm:

```text
BackendUnsupportedError
code = optimization_semantics_not_qualified
```

Chạy âm thầm một chế độ tối ưu hóa khác sẽ khiến các con số thời gian và bộ nhớ trở nên không thể
kiểm chứng, nên backend thành từ chối.

## Cách dùng

```bash
export VOXCPM_MODEL_PATH=/path/to/local/voxcpm2
export VOXCPM_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
export VOXCPM_DEVICE=cpu

# Nạp model thật, ghi nhận sự thật về thiết bị và bộ nhớ, không sinh âm thanh.
voxcpm-generate --load-only --report load.json

# TTS chuẩn thật ra WAV mono PCM16 kèm báo cáo runtime JSON.
voxcpm-generate \
  --text "Xin chào từ VoxCPM2." \
  --output out.wav \
  --report report.json
```

Môi trường định danh CPU chuẩn luôn được khởi chạy với cả `CUDA_VISIBLE_DEVICES=""` và
`VOXCPM_DEVICE=cpu`.

## Báo cáo chứa gì

Báo cáo JSON được thiết kế để có thể chia sẻ. Nó ghi lại thiết bị hiệu lực và kế hoạch thực thi,
trạng thái sẵn có và khởi tạo của CUDA, các kiểu thiết bị của tham số và buffer upstream, sự thật về
bộ nhớ host và cgroup, kích thước tập trú trước khi nạp, sau khi nạp và sau khi sinh, RSS đỉnh, thời
gian từng pha, danh tính model và digest cấu hình, phiên bản phụ thuộc đã phân giải, siêu dữ WAV và
SHA-256, cùng một khối kiểm tra WAV độc lập.

Các đường dẫn tệp tuyệt đối, thư mục home, vị trí mount model, và URL đều được che. Báo cáo nói rõ
rằng đường dẫn đã bị giữ lại thay vì lặng lẽ bỏ qua, và ghi rõ rằng trọng số model không bao giờ được
commit vào kho lưu trữ.

## Kết quả đo được

Host và môi trường cho mọi con số dưới đây:

```text
platform            Linux-6.12.90+-x86_64-with-glibc2.35
python              3.12.13 (CPython)
cpu                 4 logical, 2 physical
host ram            33,659,379,712 B (31.348 GiB)
cgroup limit        32,212,254,720 B (30.000 GiB)
torch               2.14.0+cpu
torchaudio          2.11.0+cpu
gói upstream        2.0.3.post33+gf772e498a
commit upstream     f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69
model               openbmb/VoxCPM2
revision model      32279effe8c19989596f05d353d1447f51d9e915  (do M2 ràng buộc)
config sha256 model 405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c
tệp model           10 tệp, 4,960,734,297 B
```

Ba lần chạy thật đã được thực hiện với model. RSS đỉnh được báo cáo hai lần có chủ đích: một lần từ
telemetry trong tiến trình, và một lần từ harness bên ngoài `/usr/bin/time -v`, vì telemetry trong
tiến trình không thể báo cáo sau khi bị OOM kill cứng.

| Lần chạy | Exit | Nạp | Tổng hợp | Wall | RSS đỉnh (bên ngoài) |
| --- | --- | --- | --- | --- | --- |
| Dò tải chỉ-đọc | `0` | 78.57 s | không áp dụng | 82.61 s | 11,595,091,968 B (10.799 GiB) |
| TTS CPU chuẩn | `0` | 43.15 s | 100.66 s | 148.08 s | 11,616,759,808 B (10.819 GiB) |
| Chứng minh offline | `0` | 34.32 s | 101.38 s | 139.96 s | 11,717,046,272 B (10.912 GiB) |

Chi tiết bộ nhớ cho lần chạy chuẩn:

```text
rss trước khi nạp        232,341,504 B (221.6 MiB)
rss sau khi nạp         6,710,083,584 B (6.249 GiB)
rss sau khi sinh         6,895,919,104 B (6.422 GiB)
rss đỉnh               11,616,759,808 B (10.819 GiB)
```

Nạp chiếm phần lớn tập trú. Model `4.62 GiB` cộng AudioVAE giải thích phần lớn bước nhảy từ
`221.6 MiB` lên `6.249 GiB`; khoảng cách giữa RSS ổn định và RSS đỉnh là bộ nhớ làm việc tạm thời
trong lúc giải mã, không phải trọng số lưu thêm.

Âm thanh đã sinh, lần chạy chuẩn:

```text
channels                1 (mono)
sample rate             48,000 Hz
sample width            2 bytes (PCM16)
frames                  99,840
duration                2.08 s
kích thước tệp          199,724 B
wav sha256              5a7608fe4af915c33dacc84f2193b3fade8dcf5054d3fe70e356a296a4e3df53
peak abs pcm16          32,342
rms pcm16               5,918.34
```

Tần số `48,000 Hz` được đọc từ model upstream đã nạp tại thời gian chạy. Nó không được hard-code trong
backend, nên một revision model khác báo cáo tần số khác sẽ được ghi nhận chứ không bị ghi đè.

Việc kiểm tra WAV được thực hiện hai lần và độc lập: một lần bởi bộ kiểm tra của dự án trong lúc chạy,
và một lần sau đó bởi module `wave` của thư viện chuẩn Python đọc tệp từ đĩa. Cả hai đều thống nhất về
số kênh, độ rộng, tần số, số khung, và SHA-256.

### Tổng hợp rất xa thời gian thực

`100.66 s` tổng hợp ở lần chạy chuẩn tạo ra `2.08 s` âm thanh, tương đương hệ số thời gian thực khoảng
`48.4`. Host này có 2 nhân CPU vật lý và model chạy ở `bfloat16` trên CPU, không phải là đường tối ưu
thông lượng.

Số đo này được ghi lại để chặn việc tuyên bố quá mức. Nó không phải mục tiêu hiệu năng, và không có
cam kết nào về độ trễ hay thời gian thực.

## Chứng minh không dùng CUDA

CUDA không chỉ vắng mặt khỏi báo cáo; lần chạy chuẩn được khởi chạy với mọi công tắc liên quan đều được
đặt để loại trừ nó, và model sau đó được kiểm tra sau khi nạp:

```text
CUDA_VISIBLE_DEVICES            ""            (rỗng)
VOXCPM_DEVICE                   cpu
execution_plan.effective_device cpu
execution_plan.selected_gpu_indices   []
torch.cuda.is_available()       false
torch.cuda.is_initialized()     false
torch.cuda.device_count()       0
kiểu thiết bị tham số model     ["cpu"]
kiểu thiết bị buffer model      ["cpu"]
model cpu_only                  true
```

Các API cấp phát bộ nhớ CUDA đã cố ý không bao giờ được gọi, vì chỉ dò một số trong số đó có thể khởi
tạo một CUDA context và làm mất hiệu lực của phép chứng minh.

Đường dẫn model trong lần chạy chuẩn là `explicit-path`, nên trường `revision` của chính báo cáo là
`null` theo cấu trúc. Danh tính và revision model được ràng buộc vào bản ghi uy quyền của M2, là
nguồn chân lý đã được ghi nhận cho revision mirror và digest cấu hình.

## Chứng minh không tải về

Lần chạy chứng minh offline lặp lại toàn bộ quá trình sinh với mọi đường mạng bị chặn:

```text
VOXCPM_OFFLINE=1, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
HF_HOME                = một thư mục mới hoàn toàn rỗng
HF_ENDPOINT            = http://127.0.0.1:1   (cổng đã đóng)
http_proxy/https_proxy = http://127.0.0.1:1   (cổng đã đóng)
HTTP_PROXY/HTTPS_PROXY = http://127.0.0.1:1
```

Lần chạy thoát với mã `0`, tạo ra một WAV hợp lệ, và không ghi lại lỗi kết nối, proxy, hay tải về. Sau
lần chạy, `HF_HOME` chứa `0` tệp. Kết hợp với `local_files_only=true` trong báo cáo, điều này chứng
minh rằng cả việc nạp lẫn việc sinh đều không cần mạng.

Lần chạy chuẩn và lần chạy chứng minh offline tạo ra digest WAV khác nhau với số khung giống hệt. Cơ
chế lấy mẫu của upstream có tính ngẫu nhiên, điều này là dự kiến; M4 không khẳng định điều gì về khả
năng tái tạo từng byte cho byte của quá trình sinh thật.

## Trạng thái 16 GB

```text
M4_CPU_16GB_SUFFICIENCY=NOT_QUALIFIED
```

RSS đỉnh đo được là `10.819 GiB`, thấp hơn `16 GB`. Nhưng host này có giới hạn cgroup `30 GiB` và
`31.348 GiB` RAM, và **không có ràng buộc 16 GB nào được áp dụng cho lần chạy**. Một tiến trình vừa
với `31 GiB` chưa từng được chứng minh là vừa với `16 GB`.

Để đủ điều kiện cho gate đó một cách trung thực, cùng thao tác nạp và sinh phải được chạy lại dưới giới
hạn `16 GB` bị cưỡng chế, với một harness OOM theo dõi cgroup. Cho đến lúc đó, con số này là kết quả
đo, không phải tuyên bố hỗ trợ.

## Xử lý bộ nhớ và OOM

RSS đỉnh được thu thập bởi hai cơ chế độc lập:

1. Telemetry trong tiến trình lấy mẫu kích thước tập trú trước khi nạp, sau khi nạp, sau khi sinh, và
   theo dõi mực nước cao nhất.
2. Harness bên ngoài `/usr/bin/time -v` thu thập RSS tối đa của tiến trình con, thời gian wall, và mã
   thoát, và vẫn tồn tại được sau một lần kill cứng.

Các bộ đếm `memory.events` của cgroup được lấy mẫu trước và sau mỗi lần chạy. Trên cả ba lần chạy
thật, `oom`, `oom_kill`, `max`, `high`, và `low` đều bằng `0`, và mọi lần chạy đều thoát với mã `0`. Không
lần chạy nào bị kill, và không lần chạy nào ném ra lỗi cấp phát.

Mã thoát khác 0 không được coi là OOM chỉ dựa vào bản thân nó. Phân loại là OOM đòi hỏi bằng chứng như
bộ đếm `oom` hoặc `oom_kill` tăng, một lỗi cấp phát thật do runtime ném ra, hoặc bằng chứng kết thúc OOM
tường minh từ môi trường. Nếu không, kết quả được ghi là tín hiệu hoặc chấm dứt tài nguyên không giải
thích được.

## Kiểm thử

Model thật không bao giờ được nạp trong CI thông thường. Bộ kiểm thử giả lập import upstream, nên toàn bộ
bộ kiểm thử vẫn nhanh và hermetic.

```text
mốc trước M4          226 passed
sau M4                334 passed  (60 backend, 23 WAV, 25 metrics)
bộ kiểm thử M3         77 passed  (contract, errors, fake backend)
ruff check            PASS
compileall            PASS
```

Các kiểm thử riêng cho M4 bao phủ ranh giới import lười, ánh xạ CPU `optimize=False`, cách xử lý
tham số vị trí của upstream, việc truyền lại tần số mẫu từ model, từ chối dạng sóng rỗng và không
hữu hạn, vòng đời và lỗi trạng thái, mọi thao tác không được hỗ trợ, tính nguyên tử của WAV, cắt bão
hoà, SHA-256, kiểm tra RIFF, theo dõi RSS đỉnh, và che giấu trong báo cáo.

## Những gì M4 không định danh

- 16 GB hay bất kỳ mức sàn bộ nhớ nào khác như một tuyên bố hỗ trợ
- CPU thời gian thực, phát trực tuyến, hay độ trễ tương tác
- Thực thi GPU, T4, một GPU, hay đa GPU
- Thiết kế giọng, nhân bản giọng, nối tiếp âm thanh, hay bất kỳ tính năng VoxCPM2 nào khác
- Chất lượng âm thanh, độ trung thành giọng nói, hay khả năng hiểu
- Hành vi worker đa tiến trình, đa tiến trình, hay bộ lập lịch
- Bất kỳ bề mặt API nào

Những điều đó thuộc M5 trở đi, và mỗi điều cần đo lường và bằng chứng riêng.

## Tài liệu liên quan

- [`docs/BACKEND-CONTRACT.vi.md`](BACKEND-CONTRACT.vi.md) cho ranh giới hợp đồng M3
- [`docs/CONFIGURATION.md`](CONFIGURATION.md) cho hợp đồng môi trường và chính sách thiết bị
- [`docs/MODEL-RESOLUTION.md`](MODEL-RESOLUTION.md) cho hợp đồng resolver M2
- [`provenance/M4-REAL-CPU-RUNTIME-EVIDENCE.md`](../provenance/M4-REAL-CPU-RUNTIME-EVIDENCE.md) cho bản
  ghi bằng chứng chi tiết theo từng gate
- `docs/REAL-CPU-RUNTIME.md` cho bản tiếng Anh của tài liệu này
