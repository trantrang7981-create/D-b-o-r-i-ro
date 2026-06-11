import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 0. CẤU HÌNH TRANG WEB STREAMLIT ĐẦU TIÊN
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Phát Hiện Giao Dịch Gian Lận tại Agribank",
    page_icon="❤💕"

# ==========================================
# 1. CÁC HÀM CACHE DÙNG CHUNG
# ====================f======================
@st.cache_data
def load_data(file_bytes, file_name):
    """
    Nạp dữ liệu từ bytes của file tải lên để đảm bảo tính hashable trong Streamlit cache.
    Hàm này tự động nhận diện đuôi file mở rộng .csv hoặc .xlsx.
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            df = pd.read_excel(file_bytes)
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# Định nghĩa danh sách các biến đặc trưng X được sử dụng trong notebook
FEATURES = [f"X_{i}" for i in range(1, 15)]
TARGET = "default"

# Khởi tạo session state lưu trữ trạng thái huấn luyện toàn cục
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "evaluation_metrics" not in st.session_state:
    st.session_state.evaluation_metrics = None
if "trained_features" not in st.session_state:
    st.session_state.trained_features = None

# ==========================================
# 2. THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải file dữ liệu huấn luyện mẫu lên hệ thống
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu mẫu (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Hãy chọn file dữ liệu chứa các cột từ X_1 đến X_14 và cột mục tiêu 'default' như trong Notebook."
    )
    
    st.divider()
    st.subheader("Tham số mô hình AI")
    
    # Các siêu tham số của thuật toán RandomForest được cấu hình mặc định tương ứng với mô hình huấn luyện trong Notebook
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, max_value=500, value=100, step=10,
        help="Số lượng cây quyết định trong rừng cây ngẫu nhiên."
    )
    
    criterion = st.selectbox(
        "Tiêu chí đánh giá (criterion)",
        options=["gini", "entropy", "log_loss"],
        index=0,
        help="Hàm đo lường chất lượng phân tách các nhánh cây."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa (max_depth)",
        min_value=1, max_value=50, value=15, step=1,
        help="Độ sâu tối đa của mỗi cây quyết định (Bỏ trống hoặc chọn cao để tự động mở rộng)."
    )
    
    random_state = st.number_input(
        "Trạng thái ngẫu nhiên (random_state)",
        min_value=0, max_value=9999, value=42, step=1,
        help="Giá trị seed giúp cố định kết quả xáo trộn và phân tách dữ liệu giống nhau qua các lần chạy."
    )
    
    with st.expander("Tham số cấu hình nâng cao"):
        min_samples_split = st.slider("Min samples split", min_value=2, max_value=20, value=2, step=1)
        min_samples_leaf = st.slider("Min samples leaf", min_value=1, max_value=20, value=1, step=1)

    st.divider()
    
    # Nút bấm hành động kích hoạt luồng xử lý huấn luyện mô hình duy nhất trên app
    btn_train = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)


# ==========================================
# 3. THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("Ứng Dụng Phát Hiện Giao Dịch Gian Lận & Rủi Ro Tín Dụng")
st.caption("Ứng dụng thông minh hỗ trợ phân tích dữ liệu lịch sử giao dịch nhằm phát hiện các hành vi gian lận tài chính sử dụng thuật toán Học máy ngẫu nhiên (Random Forest Classifier).")

df_main = None
if uploaded_file is None:
    st.info("💡 Vui lòng tải lên file dữ liệu mẫu ở thanh Sidebar bên trái để bắt đầu phân tích và huấn luyện mô hình.")
    st.stop()
else:
    # Đọc dữ liệu đã cache
    df_main = load_data(uploaded_file, uploaded_file.name)
    if df_main is not None:
        st.caption(f"📊 Đang sử dụng tệp dữ liệu: **{uploaded_file.name}**")
    else:
        st.error("❌ Không thể nạp dữ liệu. Vui lòng kiểm tra lại định dạng tệp đầu vào.")
        st.stop()

st.divider()

# ==========================================
# 4. KHỐI XỬ LÝ HUẤN LUYỆN (Lưu vào st.session_state)
# ==========================================
if btn_train:
    # Kiểm tra tính toàn vẹn của cấu trúc dữ liệu theo Schema trong notebook
    required_cols = FEATURES + [TARGET]
    missing_cols = [col for col in required_cols if col not in df_main.columns]
    
    if missing_cols:
        st.error(f"❌ Tệp dữ liệu thiếu các cột bắt buộc sau: {', '.join(missing_cols)}")
    else:
        with st.spinner("⏳ Hệ thống đang xử lý phân tách dữ liệu và huấn luyện mô hình AI..."):
            # Trích xuất tập đặc trưng và biến đích
            X = df_main[FEATURES]
            y = df_main[TARGET]
            
            # Thực hiện phân tách tập Train/Test theo tỷ lệ 80/20 tương tự logic Notebook
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
            
            # Thiết lập và huấn luyện mô hình Random Forest Classifier
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
            model.fit(X_train, y_train)
            
            # Dự báo kết quả trên tập kiểm định test để đánh giá chỉ số
            y_pred = model.predict(X_test)
            
            # Tính toán các metric đánh giá chi tiết
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "cm": confusion_matrix(y_test, y_pred).tolist(),
                "report": classification_report(y_test, y_pred, output_dict=True)
            }
            
            # Đồng bộ kết quả vào Session State toàn app
            st.session_state.trained_model = model
            st.session_state.evaluation_metrics = metrics
            st.session_state.trained_features = FEATURES
            
            st.success("🎉 Huấn luyện mô hình thành công! Hãy chuyển sang Tab 'Kết quả huấn luyện & kiểm định' để xem chi tiết.")


# ==========================================
# 5. KHỞI TẠO CÁC PHÂN VÙNG GIAO DIỆN TABS
# ==========================================
tabs = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả huấn luyện & Kiểm định", 
    "🔮 Sử dụng mô hình dự báo"
])

# --- TAB 1: TỔNG QUAN DỮ LIỆU ---
with tabs[0]:
    st.subheader("📋 Phân tích Thống kê Dữ liệu Thô")
    
    # Tính toán dung lượng file tải lên
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Số lượng dòng (Samples)", f"{df_main.shape[0]:,}")
    with col2:
        st.metric("Số lượng cột (Features)", f"{df_main.shape[1]:,}")
    with col3:
        st.metric("Dung lượng file", f"{file_size_mb:.2f} MB")
        
    st.write("📂 **Xem nhanh 5 hàng đầu tiên của tập dữ liệu:**")
    st.dataframe(df_main.head(), use_container_width=True)
    
    st.write("📊 **Mô tả thống kê của các biến đặc trưng đưa vào mô hình (X & y):**")
    available_model_cols = [col for col in FEATURES + [TARGET] if col in df_main.columns]
    if available_model_cols:
        st.dataframe(df_main[available_model_cols].describe(), use_container_width=True)


# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tabs[1]:
    st.subheader("📈 Phân Tích Trực Quan Biến Hệ Thống")
    
    # Ưu tiên hiển thị Biến mục tiêu 'default' đầu tiên
    if TARGET in df_main.columns:
        st.write("🎯 **Phân phối nhãn của Biến mục tiêu (0: Bình thường, 1: Gian lận/Nợ xấu)**")
        target_counts = df_main[TARGET].value_counts().reset_index()
        target_counts.columns = ['Trạng thái', 'Số lượng dòng']
        target_counts['Trạng thái'] = target_counts['Trạng thái'].map({0: 'Bình thường (0)', 1: 'Gian lận/Rủi ro (1)'})
        
        fig_target = px.bar(
            target_counts, x='Trạng thái', y='Số lượng dòng', 
            color='Trạng thái', text_auto=True,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_target.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_target, use_container_width=True)
        
    st.write("🔍 **Biểu đồ phân phối cấu trúc các biến đầu vào**")
    
    # Bộ lọc lựa chọn xem động các đặc trưng tránh quá tải màn hình
    selected_features = st.multiselect(
        "Chọn các biến đặc trưng bạn muốn trực quan hóa:",
        options=FEATURES,
        default=FEATURES[:4],
        help="Mặc định chọn 4 biến đầu tiên, bạn có thể thêm hoặc bớt biến."
    )
    
    if selected_features:
        # Tổ chức lưới hiển thị 2x2
        num_cols = 2
        for i in range(0, len(selected_features), num_cols):
            cols_grid = st.columns(num_cols)
            for j in range(num_cols):
                if i + j < len(selected_features):
                    feat = selected_features[i + j]
                    if feat in df_main.columns:
                        with cols_grid[j]:
                            fig_feat = px.histogram(
                                df_main, x=feat, color=TARGET if TARGET in df_main.columns else None,
                                barmode='overlay', marginal='box',
                                title=f"Phân phối tần suất đặc trưng {feat}",
                                color_discrete_sequence=px.colors.qualitative.Safe
                            )
                            fig_feat.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                            st.plotly_chart(fig_feat, use_container_width=True)


# --- TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH ---
with tabs[2]:
    st.subheader("🎯 Đánh Giá Độ Chính Xác Của Mô Hình Thuật Toán")
    
    if st.session_state.trained_model is None:
        st.info("💡 Chưa tìm thấy mô hình đã huấn luyện. Vui lòng quay lại thanh Sidebar bên trái và ấn nút 'Huấn luyện mô hình' để xem chi tiết kết quả kiểm định.")
    else:
        metrics = st.session_state.evaluation_metrics
        
        # Hiển thị các chỉ số cốt lõi dưới dạng Metric Card
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Độ chính xác toàn cục (Accuracy)", f"{metrics['accuracy']:.4f}")
        with m_col2:
            st.metric("Độ chính xác xác thực (Precision)", f"{metrics['precision']:.4f}")
        with m_col3:
            st.metric("Tỷ lệ bắt sót rủi ro (Recall)", f"{metrics['recall']:.4f}")
        with m_col4:
            st.metric("Chỉ số F1-Score cân bằng", f"{metrics['f1']:.4f}")
            
        st.divider()
        
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.write("📊 **Ma trận nhầm lẫn (Confusion Matrix):**")
            cm = np.array(metrics['cm'])
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế"),
                x=['Bình thường (0)', 'Gian lận (1)'],
                y=['Bình thường (0)', 'Gian lận (1)'],
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(height=350)
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with c_right:
            st.write("📋 **Báo cáo phân loại chi tiết (Classification Report):**")
            report_df = pd.DataFrame(metrics['report']).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)

        st.divider()
        st.write("📈 **Độ quan trọng của các biến tính năng (Feature Importance):**")
        importance = st.session_state.trained_model.feature_importances_
        importance_df = pd.DataFrame({
            'Tính năng': FEATURES,
            'Mức độ đóng góp': importance
        }).sort_values(by='Mức độ đóng góp', ascending=True)
        
        fig_imp = px.bar(
            importance_df, x='Mức độ đóng góp', y='Tính năng', orientation='h',
            title='Mức độ ảnh hưởng của các biến đến quyết định phân loại rủi ro gian lận',
            color='Mức độ đóng góp', color_continuous_scale='Viridis'
        )
        fig_imp.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)


# --- TAB 4: SỬ DỤNG MÔ HÌNH DỰ BÁO ---
with tabs[3]:
    st.subheader("🔮 Chẩn Đoán & Dự Báo Rủi Ro Gian Lận Giao Dịch")
    
    if st.session_state.trained_model is None:
        st.info("💡 Ứng dụng yêu cầu có mô hình làm lõi xử lý. Vui lòng nhấn nút 'Huấn luyện mô hình' tại Sidebar trước khi tiến hành dự báo trực tiếp.")
    else:
        model = st.session_state.trained_model
        
        mode = st.radio(
            "Chọn phương thức kiểm tra rủi ro:",
            options=["Nhập liệu trực tiếp từ 01 khách hàng", "Dự báo hàng loạt từ File danh sách mới (X_new)"],
            horizontal=True
        )
        
        if mode == "Nhập liệu trực tiếp từ 01 khách hàng":
            st.write("✍️ **Điền các thông số kỹ thuật của giao dịch cụ thể:**")
            
            # Tạo biểu mẫu nhập liệu tự động dựa trên phân bổ min/max/median của tập dữ liệu gốc
            with st.form("single_prediction_form"):
                form_cols = st.columns(3)
                input_data = {}
                
                for idx, feat in enumerate(FEATURES):
                    # Phân bổ đều các cột input trên giao diện lưới 3 cột
                    col_target = form_cols[idx % 3]
                    
                    # Lấy giá trị mặc định là trung vị và tính min, max tương ứng của cột tính năng đó
                    min_val = float(df_main[feat].min())
                    max_val = float(df_main[feat].max())
                    default_val = float(df_main[feat].median())
                    
                    with col_target:
                        input_data[feat] = st.number_input(
                            f"Đặc trưng {feat}",
                            min_value=min_val - 10.0,
                            max_value=max_val + 10.0,
                            value=default_val,
                            format="%.6f",
                            help=f"Giá trị trong tập dữ liệu mẫu: Min={min_val:.4f}, Max={max_val:.4f}"
                        )
                        
                submit_predict = st.form_submit_button("🛡️ Kiểm tra mức độ rủi ro", use_container_width=True)
                
                if submit_predict:
                    # Chuyển đổi dữ liệu input sang cấu trúc DataFrame chuẩn hóa tính năng
                    input_df = pd.DataFrame([input_data])
                    
                    # Tiến hành dự đoán nhãn phân lớp và tính toán xác suất phân bổ lớp rủi ro
                    pred_class = model.predict(input_df)[0]
                    pred_proba = model.predict_proba(input_df)[0]
                    
                    st.subheader("Kết quả phân tích:")
                    p_col1, p_col2 = st.columns(2)
                    with p_col1:
                        if pred_class == 1:
                            st.error("🚨 **CẢNH BÁO: Giao dịch có dấu hiệu GIAN LẬN / RỦI RO CAO!**")
                        else:
                            st.success("✅ **AN TOÀN: Giao dịch được đánh giá Bình thường.**")
                    with p_col2:
                        st.metric("Xác suất xuất hiện Gian lận (Lớp 1)", f"{pred_proba[1]*100:.2f} %")
                        st.progress(float(pred_proba[1]))
                        
        elif mode == "Dự báo hàng loạt từ File danh sách mới (X_new)":
            st.write("📂 **Tải lên tệp chứa danh sách các giao dịch tổng hợp mới cần chấm điểm rủi ro:**")
            new_file_upload = st.file_uploader(
                "Tải lên tệp kiểm tra (Chứa chính xác các cột từ X_1 đến X_14)", 
                type=["csv", "xlsx"],
                key="bulk_uploader"
            )
            
            if new_file_upload is not None:
                df_new = load_data(new_file_upload, new_file_upload.name)
                if df_new is not None:
                    # Kiểm tra tính đồng bộ của các cột đầu vào đặc trưng
                    missing_features = [col for col in FEATURES if col not in df_new.columns]
                    
                    if missing_features:
                        st.error(f"❌ Tệp tải lên thiếu các trường thông tin cột đặc trưng bắt buộc sau: {', '.join(missing_features)}")
                    else:
                        st.success("✅ Cấu trúc file hợp lệ. Đang xử lý chấm điểm chuỗi dữ liệu...")
                        
                        # Trích xuất đúng tập thuộc tính đầu vào
                        X_new_data = df_new[FEATURES]
                        
                        # Thực hiện dự báo hàng loạt
                        bulk_preds = model.predict(X_new_data)
                        bulk_probas = model.predict_proba(X_new_data)[:, 1]
                        
                        # Tạo tập kết quả mới
                        df_result = df_new.copy()
                        df_result["Dự_Báo_Kết_Quả"] = bulk_preds
                        df_result["Xác_Suất_Gian_Lận"] = bulk_probas
                        
                        st.write("📋 **Kết quả dự đoán và chấm điểm hàng loạt:**")
                        st.dataframe(df_result, use_container_width=True)
                        
                        # Thống kê tổng hợp số vụ việc bất thường
                        total_cases = len(df_result)
                        fraud_cases = int(np.sum(bulk_preds == 1))
                        
                        sc1, sc2 = st.columns(2)
                        sc1.metric("Tổng số lượng giao dịch rà soát", f"{total_cases:,}")
                        sc2.metric("Số lượng nghi vấn gian lận phát hiện", f"{fraud_cases:,}", 
                                  delta=f"{(fraud_cases/total_cases)*100:.2f}% tổng số", delta_color="inverse")
                        
                        # Chuyển đổi DataFrame sang CSV định dạng UTF-8-SIG để tránh lỗi font chữ tiếng Việt khi tải về mở bằng Excel
                        csv_data = df_result.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                        
                        st.download_button(
                            label="📥 Tải xuống bảng kết quả dự báo (.CSV)",
                            data=csv_data,
                            file_name="Ket_Qua_Du_Bao_Gian_Lan_Hang_Loat.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
