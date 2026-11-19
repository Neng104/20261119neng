import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# 設定網頁標題
st.set_page_config(page_title="中文產品評論情緒分析器", page_icon="💬")
st.title("💬 中文產品評論情緒分析器")
st.write("請在下方輸入您對產品的評論，系統會模擬產品使用的好壞並自動判定 1 至 5 顆星的等評！")

# 載入模型與分詞器 (快取確保效能)
@st.cache_resource
def load_model():
    model_name = "LiYuan/amazon-review-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

try:
    with st.spinner("正在初始化情緒分析模型，請稍候..."):
        tokenizer, model = load_model()
    st.success("模型載入成功！")
except Exception as e:
    st.error(f"模型載入失敗，錯誤訊息: {e}")

# 使用者文字輸入
user_input = st.text_area(
    "輸入您的產品評論：", 
    placeholder="例如：這個產品非常好用，品質超越預期，強烈推薦！",
    height=150
)

# 當使用者點擊按鈕時進行情緒分析
if st.button("開始進行情緒分析"):
    if user_input.strip() == "":
        st.warning("請輸入文字後再點擊按鈕！")
    else:
        with st.spinner("分析中..."):
            # 1. Tokenize 文字
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
            
            # 2. 模型預測
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 3. 溫度縮放處理 (0.5) 與 期望值計算
            temperature = 0.5
            scaled_logits = outputs.logits / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1).numpy()[0]
            
            # 各等級對應的分數
            star_values = np.array([1, 2, 3, 4, 5])
            expected_star = np.sum(probs * star_values)
            final_rating = int(round(expected_star))
            
            # 4. 顯示結果
            st.subheader("📊 分析結果")
            
            # 顯示星等圖案
            stars_display = "⭐" * final_rating + "☆" * (5 - final_rating)
            st.markdown(f"### 評估星等： {stars_display} ({final_rating} 顆星)")
            st.info(f"精確情緒加權期望分數：{expected_star:.2f} / 5.0")
            
            # 顯示各星等預估概率分佈
            st.write("**各星級機率分佈：**")
            for i, p in enumerate(probs):
                st.progress(float(p), text=f"{i+1} 顆星: {p*100:.1f}%")
