# Thiết kế thực nghiệm thesis-v3

Tài liệu mô tả bộ chạy cuối cùng của đồ án, không phải đăng ký trước độc lập.
Không chọn seed hoặc cấu hình dựa trên chiều kết luận mong muốn.
Seed cố định: 11k với k = 1..30, tức 11, 22, ..., 330. Năm seed đầu trùng
thesis-v1. Đơn vị tổng hợp bất định là seed.

## Khác biệt so với thesis-v1

thesis-v2 chỉ tăng độ chính xác, không đổi đại lượng ước lượng, quy tắc quyết
định hay hộp trạng thái kiểm tra:

- Seed 5 → 30. Seed là đơn vị suy diễn, nên đây là đòn bẩy chính thu hẹp CI.
- CartPole: 400 → 2.000 điểm kiểm tra mỗi seed. Ở tỷ lệ flip quan sát được,
  một seed 400 điểm có thể không chứa flip nào và AUROC của seed đó không xác
  định. Mức 2.000 làm thống kê theo seed tính được ở mọi seed.
- Cặp phản chiếu: quét qua cùng bốn mức yêu cầu như nhánh nhiễu ngẫu nhiên
  thay vì chỉ mức 0,35. Sai số thực tế bão hòa nên chỉ có hai mức phân biệt.
- Số episode rollout giữ nguyên 20: CI phụ thuộc biến thiên giữa seed, không
  phụ thuộc số episode trong một seed.

Cấu hình mở rộng được cố định trước và chỉ chạy một lần. Không lặp lại việc
tăng cỡ mẫu cho tới khi đạt ý nghĩa thống kê. Các kết luận vẫn không phân giải
được sau khi mở rộng đều được báo cáo nguyên trạng.

## GridWorld

Lưới 5×5, đích (4,4), xuất phát (0,0), vùng phạt (1,2), (2,2), (3,2).
Bốn hành động; đi đúng hướng 0,85; trượt sang mỗi hướng vuông góc 0,075.
Đụng biên giữ nguyên vị trí. Reward R(s,a) là reward **kỳ vọng** cố định:
-0,1 + 10 P(goal|s,a) - 5 P(hazard|s,a); đích hấp thụ reward 0.
Khi tiêm nhiễu P, giữ R cố định để cô lập sai số chuyển trạng thái.
Gamma 0,95. Value iteration tol 1e-12; policy evaluation giải hệ Bellman.

Thí nghiệm ngẫu nhiên: mọi trạng thái không phải đích, chỉ sửa dòng của
hành động greedy thật. Tạo u~Dirichlet(1,...,1); delta = alpha(u-p), với
alpha=min(magnitude/||u-p||1,1). Mức yêu cầu 0,05/0,15/0,35/0,70;
5 lần lặp mỗi state/mức/seed, tổng 14.400 dòng. Lưu sai số **thực tế**.
Ngưỡng nhóm margin là trung vị 24 trạng thái của MDP thật, không chọn theo flip.
So sánh nhóm là mô tả có điều kiện trên một bố cục; không cô lập nhân quả của margin.

Cặp phản chiếu: chọn ngẫu nhiên hai successor trong support với V(low)<V(high).
Delta chuyển t=min(magnitude/2,p(low),p(high)) từ high sang low;
nhánh còn lại dùng -delta. Hai phân phối cùng khả thi, cùng L1 và đối xứng.
Quét bốn mức yêu cầu 0,05/0,15/0,35/0,70. Vì t bị chặn bởi p(low) và p(high),
L1 thực tế chỉ nhận hai giá trị 0,05 và 0,15; ba mức yêu cầu từ 0,15 trở lên
cho cùng một can thiệp. Trần bão hòa này được báo cáo, không bị che đi.
24×4×5×30=14.400 cặp, không phải 14.400 MDP độc lập.
Nhãn down/up là dấu delta·V*, không cam kết dấu biến dạng margin sau tái tối ưu.
Đo flip tại state can thiệp, số flip toàn lưới và loss từ state xuất phát.
Đổi hành động và mất return là hai đại lượng khác nhau.

## CartPole

Gymnasium CartPole-v1 chuẩn dùng để thu thập và rollout. Hàm động học nội bộ
được đối chiếu với Gymnasium gồm reward=1 ở bước kết thúc. Có 4 trạng thái,
2 hành động; Euler dt=0,02. Hàm tham chiếu W(s)=50/(1+s' D s),
D=diag(1;0,5;10;1), gamma=0,98. Điểm hành động là 1+gamma W(s_next),
hoặc 1 khi kết thúc. Đây không phải V*, nghiệm LQR, hay critic được huấn luyện.

Mỗi seed: 20 episode × tối đa 30 bước, 20% hành động ngẫu nhiên, còn lại
greedy theo điểm tham chiếu dùng mô hình thật. 5 MLP khởi tạo độc lập, cùng
minibatch (không bootstrap), mỗi MLP có 2 lớp ẩn 64 ReLU, dự báo delta trạng thái.
Adam 1e-3, batch 32, 120 epoch, loss MSE thô, không chuẩn hóa các thành phần.
PyTorch CPU 1 thread, manual_seed và deterministic algorithms được bật.

2.000 điểm kiểm tra/seed lấy độc lập từ hộp đều: x, x_dot, theta_dot∈[-0,8;0,8],
theta∈[-0,12;0,12]. Đánh giá cả hai hành động bằng mô hình thật và mô hình học.
Tập kiểm tra này là hộp trạng thái đã chỉ định, không phải occupancy của policy.
Không dùng test để chọn mạng, epoch, hàm tham chiếu hay ngưỡng.
Phân vị 30% của margin chỉ dùng mô tả; không là ngưỡng dự báo đã huấn luyện.
AUROC của -margin và MSE đo riêng theo seed; seed chỉ có một lớp thì không xác định.
Với 2.000 điểm/seed, cả 30 seed đều có đủ hai lớp trong bộ chạy hiện tại.
Rollout: 20 reset seed giống nhau cho hai policy ở mỗi training seed; tối đa 500 bước.
Tập rollout tách khỏi tập thu thập và hộp test. Báo cả return và hiệu ứng ghép cặp.
Không dùng reference-score regret làm thay thế return thật.

## Baseline

Lưới 4×4, cùng cơ chế reward/transition, gamma 0,95. Q-learning so với Dyna-Q
lưu transition gần nhất cho mỗi (s,a), gồm reward, successor, terminal.
Mô hình last-sample của Dyna-Q không chính xác hoàn toàn trong MDP ngẫu nhiên,
kể cả khi không thêm nhiễu. Mỗi update thật có 5 update planning.
Planning chọn đều state đã thấy rồi đều action đã thấy tại state đó.
Nhiễu với xác suất 0/0,15/0,35 thay successor bởi một state đều ngẫu nhiên;
reward và terminal label của transition ghi nhớ được giữ nguyên. Đây là
mô hình hỏng thông tin successor, không phải 15%/35% L1 hay MSE.
Nếu terminal=True, không bootstrap, nên thay successor không tác động target đó.

Mỗi cấu hình/seed dùng 5.000 tương tác thật, epsilon 0,15, alpha 0,15.
Reset sau đích hoặc 100 bước; truncation không là terminal Bellman.
RNG cho planning tách khỏi RNG tương tác. Các policy có thể tạo quỹ đạo khác nhau,
nên seed ghép cặp không đồng nghĩa hoàn toàn cùng dữ liệu.
Mỗi 100 bước đánh giá chính sách greedy bằng policy evaluation trong MDP thật.
Không làm mượt đường cong. Báo chi phí planning (25.000 update phụ ở Dyna-Q).

## Xấp xỉ bậc nhất (nhóm 7)

Phát biểu kiểm chứng, với reward giữ nguyên khi tiêm nhiễu P:

    Qhat(s,a) - Q(s,a) = gamma * [ delta . Vhat + P(.|s,a) . (Vhat - V*) ]
                      ~= gamma * (delta . V*)

Chỉ số hạng delta.(Vhat-V*) là bậc hai. Số hạng P.(Vhat-V*) là bậc nhất, tỉ lệ
với trọng số quay lại chiết khấu c(s) của chính dòng bị can thiệp, nên phát biểu
chặt hơn là

    Qhat - Q = gamma * (delta . V*) * (1 + gamma c(s)) + O(delta^2)

Cả hai dự đoán đều được ghi để đo mức cải thiện.

Quyết định do margin chi phối, không phải Q(s,a*). Can thiệp dòng của a* làm
Vhat đổi, nên Q(s,a2) của hành động xếp thứ hai cũng đổi dù dòng của nó không
bị đụng. Đạo hàm bậc nhất của margin vì vậy là

    d[margin] = gamma (delta . V*) * kappa(s),
    kappa(s)  = 1 + gamma c_{a*}(s) - gamma c_{a2}(s)

Trong bố cục này kappa xuống tới ~0,14 vì gamma c_{a2} đạt 0,94 ở nơi
gamma c_{a*} chỉ 0,15. Ngưỡng đổi quyết định:

    t_crit = m(s) / (gamma * (V*(s_high) - V*(s_low)) * kappa(s))

Phiên bản đặt kappa = 1 cũng được ghi để đối chiếu: nó đạt 99,0% độ khớp nhưng
chỉ vì ba state duy nhất vượt trần khả thi đều có kappa gần 1.

Quét trên lưới đều 25 mức từ 4% đến 100% trần khả thi, tại mọi trạng thái không
phải đích, dùng hành động greedy thật. Lưới lấy theo tỉ lệ trần chứ không theo
tỉ lệ t_crit: trần khả thi ở bố cục này bằng 0,0713 tại mọi trạng thái trong khi
t_crit lớn gấp 3-6 lần ở phần lớn trạng thái, nên lưới theo t_crit sẽ để đa số
trạng thái không có điểm khả thi.

Cả hai vế đều tính được chính xác bằng quy hoạch động, nên đây là kiểm chứng
số chứ không phải ước lượng thống kê; không có seed và không có khoảng tin cậy.
`verify_results.py` yêu cầu tương quan > 0,99, không có flip dưới 0,8·t_crit và
tỷ lệ flip > 90% trên 1,5·t_crit.

Đây là xấp xỉ bậc nhất, không phải chặn đúng với mọi delta.

Giới hạn: phát biểu chỉ mô tả một dòng bị can thiệp trên MDP bảng, không mở
rộng sang mô hình nơ-ron hay sang tổn thất return toàn cục.

## Fork-MDP (nhóm 4)

Hai mươi lăm Fork-MDP sinh độc lập, seed 42..66. Mỗi thiết kế rút theo đúng thứ
tự: độ dài nhánh {2,3}; p_left~U(0,80;0,95); p_right=clip(p_left-U(0,02;0,12));
r_left~U(0,9;1,1); r_right=clip(r_left-U(0,02;0,20)); step_cost~U(0,005;0,02);
gamma∈{0,90;0,95;0,98}. Bố cục là hàm của thứ tự rút, nên thứ tự này cố định.

Với mỗi (s,a) không phải terminal và mỗi mức trong chín mức 0,02..0,35, khối
lượng được chuyển giữa hai successor cực trị theo V*, chặn ở 95% của successor
nhỏ hơn. Bỏ qua dòng có support < 2, dòng có V*(high) ≤ V*(low)+1e-8, và khối
lượng < 1e-4.

Nhãn hai nhánh theo tác động lên **margin**, không theo hướng giá trị: giảm
Q(s,a) chỉ đóng margin khi a là hành động greedy. Gán nhãn theo hướng giá trị
làm đảo dấu tổng hợp vì phần lớn (s,a) không greedy.

Đại lượng: giá trị sửa chữa J(pi*) - J(pi_hat) trong MDP thật, **không kẹp về 0**
vì dấu chính là thứ so sánh. Đơn vị suy diễn là MDP; CI dùng bootstrap lấy lại
nguyên cụm MDP, 10.000 lần lặp.

## Audit MuJoCo (nhóm 5)

Quy tắc quyết định đóng băng trong `docs/mujoco_provenance/T1A_PREREGISTRATION.md`
ngày 07/09/2026, trước khi tính hoặc đọc bất kỳ output nào. Thiết kế đầy đủ 12 ô:
{Hopper-v4, Walker2d-v4, HalfCheetah-v4} × seed {42,43} × checkpoint {50k,150k}.
Mỗi ô 2.000 trạng thái trên 25 quỹ đạo.

Tín hiệu ứng viên S_cont = sigma_grad_aQ/(h_hat + eps_H). Đối chứng: sigma_dyn
(phân tán dự báo trạng thái) và sigma_Q (phân tán giá trị). Nhãn: y_A sai lệch
hành động cảm sinh (chính), y_B sai số động học, y_C sai số giá trị.

Phần GPU **không chạy lại**. Bảng theo từng trạng thái đóng băng trong
`results/mujoco_signal_audit/`; mọi thống kê tính lại từ đó trên CPU. Bootstrap
lấy lại nguyên cụm quỹ đạo, 2.000 lần lặp. `verify_results.py` băm lại file
đóng băng và đối chiếu từng hệ số Spearman với bảng gốc, dừng nếu lệch.

Giới hạn: đây là quan hệ xếp hạng trên phân phối trạng thái do policy host sinh
ra, không phải phát biểu về hiệu năng học. Không kết luận gì về MBPO hay các
biến thể value-aware từ nhóm này.

## CartPole trên quỹ đạo held-out (nhóm 6)

Giữ nguyên quy trình thu thập và huấn luyện của nhóm 2. Đánh giá trên 20 episode
held-out mỗi seed, reset seed = seed*1000 + 700 + episode, tách khỏi cả tập thu
thập (offset 0) lẫn tập rollout (offset 500). Quỹ đạo sinh bởi chính chính sách
tham chiếu nên tập trạng thái không phụ thuộc điểm số đang được chấm.

Hai loại điểm báo riêng. Đặc quyền: -m của tham chiếu thật, không dùng được khi
triển khai vì cần chính tham chiếu mà mô hình phải thay thế. Không đặc quyền:
sigma_dyn/(m_hat + 1e-3) và sigma_dyn, chỉ dùng đại lượng mô hình tự có.

## Thống kê và giới hạn kết luận

Tính trung bình trong seed trước; CI 95% dùng Student t với n-1 bậc tự do.
Hiệu ứng ghép cặp lấy chênh lệch trong seed rồi mới tính CI.
CI t được giữ nguyên, có thể vượt [0,1] với tỷ lệ/AUROC hoặc vượt cận return;
đây là xấp xỉ kém khi ít seed hoặc ít biến cố, không là xác suất hợp lệ ngoài miền.
Không diễn giải CI [0,0] do mọi seed không có flip thành rủi ro bằng 0.
Không coi các state/repetition trong cùng seed là quan sát độc lập để tính p-value.
CI đường cong là pointwise, không là dải đồng thời. Không kiểm định hàng loạt checkpoint.
Ba mươi seed cho CI hẹp hơn bản thăm dò năm seed nhưng vẫn chỉ áp dụng cho
một bố cục lưới và hai môi trường nhỏ; không tuyên bố ưu thế thuật toán phổ quát.
