# Phạm vi đồ án tốt nghiệp

Căn cứ: Phiếu giao đề tài VLU.ĐAKLTN.08 do sinh viên cung cấp ngày 10/09/2026.
Tên đề tài: Nghiên cứu ảnh hưởng của sai số world model đến quyết định của tác tử trong học tăng cường.

## Đối chiếu yêu cầu và sản phẩm

| Yêu cầu phiếu giao đề tài | Thực hiện trong bản đồ án |
|---|---|
| Đo sai số dự báo và tác động lên quyết định | L1 của phân phối chuyển trạng thái, MSE của trạng thái dự báo, sai số Q, đổi hành động, giá trị sửa chữa |
| Cơ sở lý thuyết về margin | Xấp xỉ bậc nhất cho độ dịch margin, gồm số hạng của hành động cạnh tranh, và ngưỡng $t_{crit}$; kiểm chứng số trên MDP giải chính xác |
| Khảo sát action-value margin | GridWorld và Fork-MDP với Q tối ưu; CartPole với điểm tham chiếu thủ công, ghi rõ giới hạn |
| Thí nghiệm có kiểm soát, nhiều **dạng** sai số | Nhiễu hỗn hợp Dirichlet, cặp phản chiếu ngẫu nhiên, cặp phản chiếu cực trị theo $V^*$, mạng học từ dữ liệu, nhiễu successor trong planning |
| Đánh giá trên **một số môi trường tiêu chuẩn** | Gymnasium CartPole-v1, Hopper-v4, Walker2d-v4, HalfCheetah-v4; cùng GridWorld và Fork-MDP tự xây dựng |
| Baseline RL/MBRL cơ bản | Q-learning và Dyna-Q, cùng 5.000 tương tác thật |
| Hiệu năng tác tử | Return của chính sách greedy trong MDP thật và rollout CartPole-v1 |
| Nhiều seed và thống kê | 30 seed, CI Student t theo seed; bootstrap cụm khi đơn vị suy diễn là MDP hoặc quỹ đạo |
| Tái lập | Scripts, CSV, cấu hình, manifest SHA-256, tests, hướng dẫn |

## Bảy nhóm thí nghiệm

1. GridWorld 5×5: nhiễu L1 có kiểm soát và cặp phản chiếu khả thi.
2. CartPole-v1: ensemble MLP học động học, chấm điểm trên hộp trạng thái đều.
3. Baseline GridWorld 4×4: Q-learning so với Dyna-Q dưới ngân sách tương tác chung.
4. Họ 25 Fork-MDP: hướng sai số có dấu dưới severity đã ghép cặp.
5. Audit 12 ô trên ba môi trường MuJoCo: cổng tín hiệu quyết định tiền đăng ký.
6. CartPole-v1 trên quỹ đạo held-out: điểm đặc quyền so với không đặc quyền.
7. Kiểm chứng xấp xỉ bậc nhất cho độ nhạy margin và ngưỡng đổi quyết định trên GridWorld giải chính xác.

## Quyết định về phần nghiên cứu mở rộng

Thư mục nghiên cứu `drl-thesis` chứa một bản thảo tạp chí rộng hơn đề tài này.
Việc đưa từng phần vào đồ án được quyết định theo ba tiêu chí: có nằm trong
phạm vi phiếu giao không, có kiểm chứng lại được không, và sinh viên có bảo vệ
được không.

**Đã đưa vào** (nhóm 4, 5, 6). Cả ba được cài đặt lại trong repo này theo văn
phong và giao diện sẵn có, không sao chép script nghiên cứu, rồi đối chiếu với
số liệu gốc:

| Nhóm | Kiểm chứng |
|---|---|
| Fork-MDP | 1.782 cặp, hiệu số 0,00894, CI [0,00718; 0,01083]; bản gốc cho 0,00894, CI [0,00716; 0,01087] |
| MuJoCo | mọi hệ số Spearman tính lại khớp bảng đóng băng tới sai lệch 0,00e+00 |
| CartPole held-out | thí nghiệm mới, dùng lại ensemble và agent sẵn có của nhóm 2 |

Phần đắt của nhóm 5 (huấn luyện SAC và tính tín hiệu trên GPU) không chạy lại
trong đồ án. Bảng theo từng trạng thái được đóng băng trong
`results/mujoco_signal_audit/` (khoảng 1 MB) và mọi thống kê báo cáo được tính
lại từ đó trên CPU trong khoảng 25 giây. Xuất xứ phần đóng băng nằm trong
`docs/mujoco_provenance/`, gồm mã GPU, phiên bản thư viện và SHA-256 của từng
file nguồn; `verify_results.py` băm lại các file này và đối chiếu từng hệ số.

**Không đưa vào.** So sánh học MBPO/VaGraM/VAML trên MuJoCo; định lý phân tách
minimax; ranking sửa mô hình theo ngân sách; bridge phản chiếu quanh mô hình
được fit; học hàm mất mát mới; bảo đảm an toàn triển khai robot.

Lý do loại phần so sánh học: phiếu giao yêu cầu thuật toán RL/MBRL **cơ bản**,
trong khi VaGraM và VAML không thuộc nhóm đó; chi phí tái lập khoảng 400 giờ
GPU RTX 5090 nên không kiểm chứng lại được; và chính bản thảo gốc xếp phần này
là thăm dò với return "không so sánh được với MBPO đã công bố". Ranking sửa mô
hình bị loại vì nhóm 5 đã trả lời cùng câu hỏi đó trên ba môi trường tiêu chuẩn
với quy tắc quyết định cố định trước.

**Mức công khai.** Nhóm 4 và nhóm 5 trùng một phần với bản thảo tạp chí đang
soạn trong `drl-thesis`. Việc giữ chúng trong bản đồ án công khai là lựa chọn
có chủ đích, không phải sơ suất: chúng lấp đúng yêu cầu "một số môi trường học
tăng cường tiêu chuẩn" của phiếu giao, và cả hai đều kiểm chứng lại được trên
máy thường. Khi nộp bản thảo, cần khai báo khóa luận này như prior disclosure
theo chính sách của venue.

Đây là nghiên cứu thực nghiệm về quan hệ sai số và quyết định, không phải đề
xuất thuật toán DRL mới.

Báo cáo LaTeX nằm riêng ở thư mục `VLU___KLTN_HK261_TTNT_Minh` cạnh repo.
Repo công khai chỉ chứa mã nguồn, dữ liệu, hình và tài liệu tái lập;
không đưa phiếu giao đề tài hay phiếu đánh giá cá nhân lên GitHub.
