# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 6/11/2026**

**Runtime Colab:** CPU

**Python: 3.13.18; PyTorch: 2.11.0+cpu; Ultralytics: 8.4.145**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` 

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh

1. Record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`= 1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về cả bức ảnh. Vì mô hình Image Classification thực hiện phân loại ở cấp độ ảnh, tức là đưa ra một lớp đại diện cho nội dung tổng thể của ảnh, chứ không xác định vị trí hay phân loại riêng từng vật thể.
3. Danh sách tên lớp được định nghĩa sẵn trong taxonomy đi kèm với checkpoint của mô hình. Model không tự tạo ra tên lớp mới mà chỉ dự đoán trong danh sách lớp đã được huấn luyện, ví dụ ImageNet-1K.
4. Cần ghi cả ba để xác định chính xác nhãn:
`class_id`: Mã số định danh của lớp.
`class_name`: Tên dễ đọc của lớp.
`taxonomy_name`: Hệ thống phân loại mà lớp đó thuộc về.
Chỉ ghi tên lớp có thể gây nhầm lẫn khi các taxonomy khác nhau có tên lớp giống nhau nhưng ý nghĩa hoặc mã ID khác nhau. Ghi đủ ba trường giúp kiểm tra và đối chiếu nhãn chính xác hơn.
5. Guideline cần quy định rõ tiêu chí chọn lớp đại diện cho toàn bộ ảnh, chẳng hạn dựa trên chủ thể chính, nội dung chiếm ưu thế hoặc mục đích của ảnh. Đồng thời cần có quy tắc xử lý khi ảnh chứa nhiều chủ đề ngang nhau hoặc không đủ thông tin để xác định lớp phù hợp.
6. Vì score chỉ thể hiện mức độ tin tưởng của mô hình đối với prediction, không chứng minh prediction đó là đúng. Ground truth phải được tạo bởi annotator dựa trên ảnh thực tế và guideline đã thống nhất, sau đó được reviewer kiểm tra chất lượng.

## 2. Phát hiện vật thể

1. Chép một record: `class_name`= cup ; `score`= 0.381542 ; `bbox_xyxy`=[119.95,272.2,141.78,304.16] ; `bbox_width`= 21.84 ; `bbox_height`= 31.96
2. Box có góc trên-trái tại tọa độ (119.95, 272.2) và góc dưới-phải tại tọa độ (141.78, 304.16).
Các tọa độ được tính theo pixel, với gốc tọa độ (0, 0) nằm ở góc trên-trái ảnh.
Box có chiều rộng khoảng 21.84 pixel và chiều cao khoảng 31.96 pixel, bao quanh vật thể được dự đoán là cup.
3. Ngưỡng 0.20: 17 predictions; Ngưỡng 0.60: 6 predictions.
Khi sử dụng ngưỡng 0.20, mô hình giữ lại nhiều prediction hơn vì chấp nhận các kết quả có score thấp hơn. Ngược lại, ngưỡng 0.60 loại bỏ nhiều prediction có score thấp, nên số lượng kết quả thường ít hơn.

4. Reviewer cần kiểm tra nhiều prediction hơn, đặc biệt là các box có confidence thấp. Cần phát hiện false positive (nhận diện nhầm vật thể), box bị trùng lặp, box sai vị trí hoặc prediction không phù hợp với vật thể thực tế. Những trường hợp không chắc chắn cần được đối chiếu với guideline hoặc đưa lên escalation.
5. Mỗi vật thể được gán một bounding box riêng, bao sát phần vật thể nhìn thấy được, không bao gồm các vùng nền hoặc vật thể khác. Tọa độ box phải được xác định nhất quán theo thứ tự [x_min, y_min, x_max, y_max], với gốc tọa độ ở góc trên-trái ảnh và đơn vị là pixel.
6. Trên ảnh `kitchen`, phần lò nướng (oven) bị che đi bởi phần bàn ăn (đining table). Tôi sẽ đưa trường hợp này lên Lab Coach để escalation nếu chưa có quy tắc rõ ràng trong guideline. Vì vật thể bị che khuất hoặc cắt khỏi khung hình có thể khiến việc xác định phạm vi bounding box không thống nhất. Không nên tự suy đoán phần vật thể không nhìn thấy hoặc tự đặt ra quy tắc gán nhãn.

## 3. Phân đoạn theo từng đối tượng

1. Chép một record: `instance_id`= kitchen-001 ; `class_name`= person ; `score`= 0.899318 ; số điểm polygon=`polygon_point_count`= 348 ; 3–5 điểm đầu của `polygon_xy`= [446.0,70.0],[445.0,71.0],[444.0,71.0],[443.0,72.0],[442.0,72.0]
2. Bounding box chỉ thể hiện hình chữ nhật bao quanh vật thể, nên có thể chứa cả vùng nền không thuộc vật thể.
Polygon mô tả đường biên chi tiết hơn bằng nhiều điểm tọa độ, giúp thể hiện hình dạng thực tế của vật thể và phân biệt vùng thuộc vật thể với vùng nền xung quanh.
3. `instance_id` dùng để định danh riêng từng đối tượng xuất hiện trong ảnh, kể cả khi nhiều đối tượng có cùng class.
Nó khác với:
`class_id`: Mã định danh của loại vật thể, ví dụ person hoặc cup.
`tracking_id`: Mã định danh một đối tượng được theo dõi xuyên suốt nhiều frame trong video.
4. Tôi sẽ sử dụng guideline để xử lý nếu guideline đã quy định rõ cách gán mask cho vật thể bị che khuất. Tuy nhiên, nếu chưa có quy tắc cụ thể về việc chỉ vẽ phần nhìn thấy hay suy đoán toàn bộ hình dạng của lò nướng, tôi sẽ đưa trường hợp này lên Lab Coach để escalation.
Lý do là cần đảm bảo các annotator áp dụng cùng một quy tắc và không tự suy đoán phần biên bị che khuất.
5. Trên ảnh `kitchen`, phần lò nướng (oven) bị che đi bởi bàn ăn (dining table). 
Tôi sẽ sử dụng guideline để xử lý nếu guideline đã quy định rõ cách gán mask cho vật thể bị che khuất. Tuy nhiên, nếu chưa có quy tắc cụ thể về việc chỉ vẽ phần nhìn thấy hay suy đoán toàn bộ hình dạng của lò nướng, tôi sẽ đưa trường hợp này lên Lab Coach để escalation.
Lý do là cần đảm bảo các annotator áp dụng cùng một quy tắc và không tự suy đoán phần biên bị che khuất.

## 4. Vòng đời và kiểm tra chất lượng

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một class ID và class name đại diện cho toàn bộ ảnh, thuộc taxonomy đã quy định. | Ảnh có nhiều vật thể hoặc nhiều chủ đề; prediction top-1 có thể không đại diện chính xác cho nội dung chính. (segmentation_prediction.png) | Chọn một lớp theo tiêu chí trong guideline. | Kiểm tra lớp được chọn có phù hợp với nội dung ảnh và taxonomy không. |
| Phát hiện vật thể | Mỗi object có một class và một bounding box gồm 4 tọa độ [x_min, y_min, x_max, y_max]. | Object nhỏ, bị che khuất hoặc cắt mép; box có thể sai vị trí hoặc bỏ sót vật thể. Ví dụ prediction cup có score 0.381542. | Gán class và box cho từng object theo quy tắc thống nhất. | Kiểm tra class, vị trí box, độ bao sát, vật thể bị bỏ sót và box trùng lặp. |
| Instance segmentation | Mỗi instance có class, instance ID và polygon mô tả đường biên vật thể. | Vật thể bị che khuất khiến đường biên không rõ. Ví dụ oven bị dining table che khuất. | Vẽ polygon theo phần biên được quy định trong guideline; escalation nếu chưa rõ. | Kiểm tra polygon, đường biên, vùng mask, instance ID và tính nhất quán giữa các nhãn. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng dữ liệu công khai được cho phép trong phạm vi bài lab, không tải lên hoặc chia sẻ ảnh cá nhân, dữ liệu nội bộ hay thông tin nhạy cảm khi chưa được cấp quyền.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ dừng xử lý dữ liệu đó, không tự ý tải xuống hoặc chia sẻ, và báo cho Lab Coach hoặc người phụ trách để được hướng dẫn.