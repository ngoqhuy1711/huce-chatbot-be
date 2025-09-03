# HUCE CHATBOT BACKEND - MÔ TẢ DỰ ÁN

## Tổng quan dự án

**HUCE Chatbot Backend** là một hệ thống chatbot thông minh được phát triển cho trường Đại học Xây dựng Hà Nội (HUCE), sử dụng công nghệ xử lý ngôn ngữ tự nhiên (NLP) và trí tuệ nhân tạo để hỗ trợ sinh viên và người dùng tìm hiểu thông tin về tuyển sinh, đào tạo và các dịch vụ của trường.

## Mục tiêu dự án

- Xây dựng hệ thống chatbot tự động trả lời các câu hỏi thường gặp về tuyển sinh đại học
- Hỗ trợ sinh viên tìm hiểu thông tin về ngành học, điều kiện tuyển sinh, học phí
- Giảm tải cho bộ phận tư vấn tuyển sinh
- Cung cấp thông tin chính xác và nhanh chóng 24/7

## Kiến trúc hệ thống

### 1. Cấu trúc thư mục
```
huce-chatbot-be/
├── api/                    # API endpoints (FastAPI)
├── configs/               # Cấu hình hệ thống
├── data/                  # Dữ liệu training và responses
├── nlu/                   # Natural Language Understanding
│   ├── entity/           # Xử lý entities
│   └── intent/           # Phát hiện ý định người dùng
├── test/                  # Test files
└── requirements.txt       # Dependencies
```

### 2. Công nghệ sử dụng

#### Backend Framework
- **FastAPI**: Framework web hiện đại, hiệu suất cao cho Python
- **Uvicorn**: ASGI server để chạy FastAPI

#### Xử lý ngôn ngữ tự nhiên
- **spaCy**: Thư viện NLP cho tiếng Việt (vi_core_news_sm)
- **Sentence Transformers**: Mô hình embedding cho semantic search
- **scikit-learn**: Machine learning và đánh giá hiệu suất

#### Xử lý dữ liệu
- **Pandas**: Xử lý và phân tích dữ liệu CSV
- **PyYAML**: Đọc file cấu hình YAML
- **underthesea**: Thư viện NLP tiếng Việt

## Các thành phần chính

### 1. Hệ thống NLU (Natural Language Understanding)

#### Intent Detection
- **Mô hình**: Sử dụng sentence-transformers/all-MiniLM-L6-v2
- **Phương pháp**: Semantic similarity với threshold tối ưu
- **Tính năng**: 
  - Phát hiện ý định người dùng từ văn bản đầu vào
  - Tự động tìm threshold tối ưu cho từng dataset
  - Xử lý trường hợp "unknown" khi độ tin cậy thấp

#### Entity Recognition
- **Mục đích**: Trích xuất thông tin cụ thể từ câu hỏi
- **Ví dụ**: Tên ngành học, điểm số, năm học, v.v.

### 2. Cơ sở dữ liệu

#### Dữ liệu tuyển sinh
- **admission_conditions.csv**: Điều kiện tuyển sinh các ngành
- **admission_methods.csv**: Phương thức tuyển sinh
- **admission_quota.csv**: Chỉ tiêu tuyển sinh
- **major_intro.csv**: Giới thiệu ngành học
- **tuition.csv**: Học phí các ngành
- **scholarships_huce.csv**: Học bổng nội bộ
- **scholarships_international.csv**: Học bổng quốc tế

#### Dữ liệu training
- **intent.csv**: Câu hỏi mẫu và ý định tương ứng
- **entity.csv**: Entities và synonyms
- **responses.json**: Các câu trả lời mẫu

### 3. API Endpoints

#### Main API (`api/main.py`)
- Xử lý requests từ frontend
- Tích hợp với hệ thống NLU
- Trả về responses phù hợp

## Quy trình hoạt động

### 1. Training Pipeline
```
1. Thu thập dữ liệu câu hỏi và câu trả lời
2. Preprocessing và labeling dữ liệu
3. Training mô hình intent detection
4. Xây dựng index cho semantic search
5. Đánh giá hiệu suất mô hình
```

### 2. Inference Pipeline
```
1. Người dùng gửi câu hỏi
2. Hệ thống phân tích ý định (intent)
3. Trích xuất entities nếu cần
4. Tìm câu trả lời phù hợp
5. Trả về response cho người dùng
```

## Đánh giá hiệu suất

### 1. Metrics sử dụng
- **Precision**: Độ chính xác của dự đoán
- **Recall**: Khả năng phát hiện đúng ý định
- **F1-Score**: Trung bình điều hòa của precision và recall
- **Confusion Matrix**: Ma trận nhầm lẫn
- **Threshold Analysis**: Phân tích ngưỡng tối ưu

### 2. Báo cáo đánh giá
Hệ thống tự động tạo báo cáo chi tiết bao gồm:
- Tổng quan dataset
- Phân tích threshold
- Hiệu suất tổng thể
- Phân tích từng class
- Ma trận nhầm lẫn
- Phân tích lỗi
- Khuyến nghị cải thiện

## Cài đặt và chạy

### 1. Yêu cầu hệ thống
- Python 3.8+
- RAM: 4GB+ (cho mô hình NLP)
- Storage: 2GB+ (cho models và data)

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Chạy hệ thống
```bash
# Chạy API server
uvicorn api.main:app --reload

# Training mô hình
python nlu/intent/build_index.py

# Đánh giá mô hình
python nlu/intent/evaluate.py
```

## Tính năng nổi bật

### 1. Tự động tối ưu hóa
- Tự động tìm threshold tối ưu cho mỗi dataset
- Adaptive learning từ dữ liệu mới
- Dynamic response selection

### 2. Xử lý tiếng Việt
- Hỗ trợ đầy đủ tiếng Việt
- Xử lý từ đồng nghĩa và biến thể
- Nhận diện entities tiếng Việt

### 3. Khả năng mở rộng
- Kiến trúc modular dễ mở rộng
- Hỗ trợ thêm ngôn ngữ mới
- API RESTful chuẩn

## Kế hoạch phát triển

### Phase 1 (Hiện tại)
- ✅ Hệ thống NLU cơ bản
- ✅ Intent detection
- ✅ Entity recognition
- ✅ API endpoints

### Phase 2 (Tương lai)
- 🔄 Tích hợp với database thực
- 🔄 Hệ thống logging và monitoring
- 🔄 Authentication và authorization
- 🔄 Multi-language support

### Phase 3 (Dài hạn)
- 📋 Machine learning nâng cao
- 📋 Sentiment analysis
- 📋 Conversation flow management
- 📋 Integration với các hệ thống khác

## Đóng góp và phát triển

### Cách đóng góp
1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Tạo Pull Request

### Coding standards
- Tuân thủ PEP 8
- Viết docstring cho functions
- Test coverage > 80%
- Type hints cho Python 3.8+

## Liên hệ và hỗ trợ

- **Repository**: [GitHub Link]
- **Documentation**: [Wiki Link]
- **Issues**: [GitHub Issues]
- **Email**: [Contact Email]

---

*Dự án được phát triển bởi sinh viên Đại học Xây dựng Hà Nội*
*Phiên bản: 1.0.0*
*Cập nhật lần cuối: [Ngày hiện tại]*
