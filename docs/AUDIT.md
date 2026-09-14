# Kiểm tra bản tách đồ án

Bản tách ban đầu có ba lỗi đã được tái hiện bằng regression test trước khi sửa:

1. Cặp sai số được chiếu riêng về simplex, phá tính phản chiếu. Bản sửa tạo
   hai hướng khả thi trên support chung, lưu L1 thực tế cho cả hai nhánh.
2. Dyna-Q không lưu terminal flag; planning bootstrap sai ở transition kết thúc.
   Bản sửa lưu cờ và dùng target reward-only cho terminal.
3. CartPole nội bộ trả reward 0 khi kết thúc, khác Gymnasium mặc định.
   Bản sửa trả 1 và có test đối chiếu state/reward/terminal.

Các sửa đổi phương pháp: cố định PyTorch seed; RNG planning riêng; baseline
cùng số tương tác thật; đánh giá greedy return trong môi trường thật; cặp
CartPole rollout cùng reset; thống kê theo seed; báo AUROC thiếu một lớp.

Kết quả cũ không được dùng để hỗ trợ các kết luận mới. Hình và bảng mới đọc
trực tiếp CSV của bộ chạy hiện tại. Nguồn và CSV được băm trong manifest.
Báo cáo giữ các kết quả không thuận lợi và không điền thay đánh giá giảng viên.
