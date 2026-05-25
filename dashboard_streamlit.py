import streamlit as st
import pandas as pd
import json
import re
import os
import plotly.express as px

# ==========================================
# [설정] 데이터가 저장된 폴더 경로
# ==========================================
# app.py 파일과 같은 위치에 'data' 폴더를 만들고 jsonl 파일들을 넣어두세요.
DATA_FOLDER = "data"

def get_local_jsonl_files(folder_path):
    """지정된 폴더 내의 모든 .jsonl 파일 목록을 가져옴"""
    if not os.path.exists(folder_path):
        # 폴더가 없으면 에러 방지를 위해 자동 생성
        os.makedirs(folder_path)
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
    return sorted(files)

def load_jsonl_to_dataframe(file_path):
    """선택한 로컬 JSONL 파일을 읽어와 DataFrame으로 변환"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    raw_data = [json.loads(line) for line in lines]
    return pd.DataFrame(raw_data)

# 뉴스 텍스트에서 수치 지표 추출 헬퍼 함수
def extract_macro_info(prompt_text):
    macro_data = {}
    # 원/달러 환율 추출
    fx_match = re.search(r"원/달러 환율:\s*([\d\.]+)원", prompt_text)
    if fx_match:
        macro_data['FX_Rate'] = float(fx_match.group(1))
    
    # RSI 추출
    rsi_match = re.search(r"현재 RSI:\s*([\d\.]+)", prompt_text)
    if rsi_match:
        macro_data['RSI'] = float(rsi_match.group(1))
        
    return macro_data

# ==========================================
# Streamlit UI 레이아웃 시작
# ==========================================
st.set_page_config(
    page_title="LLM 주가 예측 결과 통합 분석기",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LLM 주가 예측 결과 통합 분석 대시보드")
st.caption("data/ 폴더 내부의 JSONL 파일을 자동으로 탐색하여 여러 종목의 LoRA 백테스팅 성과를 분석합니다.")

# 1. 로컬 데이터 폴더 내 파일 목록 실시간 스캔
jsonl_files = get_local_jsonl_files(DATA_FOLDER)

if not jsonl_files:
    st.warning(f"⚠️ `{DATA_FOLDER}/` 폴더 내에 `.jsonl` 파일이 존재하지 않습니다. 파일을 폴더에 넣어주세요.")
else:
    # 2. 사이드바에서 분석할 파일(종목) 선택
    with st.sidebar:
        st.header("📂 종목 및 결과 선택")
        
        # 파일명에서 종목명만 깔끔하게 추출해서 보여주기 (예: '삼성전자_lora_llm_results.jsonl' -> '삼성전자')
        file_display_names = {f.split('_')[0]: f for f in jsonl_files}
        
        selected_stock = st.selectbox(
            "분석할 주식 종목을 고르세요",
            options=list(file_display_names.keys())
        )
        
        selected_file_name = file_display_names[selected_stock]
        file_path = os.path.join(DATA_FOLDER, selected_file_name)
        st.success(f"🔄 로드 완료: {selected_file_name}")

    # 3. 데이터 로드 및 파싱
    df = load_jsonl_to_dataframe(file_path)
    
    # 수치 정보 정규식 파싱 및 결합
    if 'prompt' in df.columns:
        extracted_features = [extract_macro_info(p) for p in df['prompt']]
        ext_df = pd.DataFrame(extracted_features)
        df = pd.concat([df, ext_df], axis=1)
    
    # 날짜 정렬
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

    # 4. 상단 핵심 KPI 메트릭 지표
    st.write("---")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric("총 테스트 일수", f"{len(df)} 일")
    with kpi2:
        model_name = df['model'].iloc[0].split('+')[0] if 'model' in df.columns else "Unknown"
        st.metric("테스트 모델", model_name)
    with kpi3:
        if 'predicted_label' in df.columns and 'actual_label' in df.columns:
            accuracy = (df['predicted_label'] == df['actual_label']).mean() * 100
            st.metric("예측 정확도 (Accuracy)", f"{accuracy:.2f} %")
    with kpi4:
        if 'actual_return_pct' in df.columns:
            avg_return = df['actual_return_pct'].mean()
            st.metric("7일 후 평균 실제 수익률", f"{avg_return:.2f} %")

    # 5. 메인 레이아웃 (좌측: 데이터 시각화 / 우측: 상세 추론 내용)
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("📊 백테스팅 데이터 트렌드")
        
        chart_type = st.radio(
            "시각화할 지표를 선택하세요",
            ["실제 수익률 추이", "예측 라벨 vs 실제 라벨 분포", "환율 및 RSI 추이"],
            horizontal=True
        )
        
        if chart_type == "실제 수익률 추이" and 'actual_return_pct' in df.columns:
            fig = px.line(df, x='date', y='actual_return_pct', markers=True,
                          title=f"{selected_stock} 7일 후 실제 수익률(%) 추이",
                          labels={'actual_return_pct': '수익률 (%)', 'date': '기준 날짜'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "예측 라벨 vs 실제 라벨 분포":
            c1, c2 = st.columns(2)
            with c1:
                fig_pred = px.pie(df, names='predicted_label', title='LLM의 예측 라벨 분포', hole=0.3)
                st.plotly_chart(fig_pred, use_container_width=True)
            with c2:
                fig_act = px.pie(df, names='actual_label', title='7일 뒤 실제 결과 라벨 분포', hole=0.3)
                st.plotly_chart(fig_act, use_container_width=True)
                
        elif chart_type == "환율 및 RSI 추이":
            available_metrics = [m for m in ['FX_Rate', 'RSI'] if m in df.columns]
            if available_metrics:
                fig_macro = px.line(df, x='date', y=available_metrics, 
                                    title="프롬프트 내 지표 변화 추이",
                                    markers=True)
                st.plotly_chart(fig_macro, use_container_width=True)
            else:
                st.warning("프롬프트에서 수치 데이터를 추출하지 못했습니다.")
        
        # 데이터 테이블
        st.subheader("📋 전체 결과 데이터 테이블")
        show_cols = [c for c in ['date', 'predicted_label', 'actual_label', 'actual_return_pct'] if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True)

    with col_right:
        st.subheader("🔍 일자별 LLM 상세 분석 내용 (Output)")
        
        if 'date' in df.columns:
            date_strings = df['date'].dt.strftime('%Y-%m-%d').unique()
            selected_date = st.selectbox("상세 추론 과정을 확인할 날짜 선택", date_strings)
            
            selected_row = df[df['date'].dt.strftime('%Y-%m-%d') == selected_date].iloc[0]
            
            st.info(f"📅 **기준 날짜:** {selected_date}")
            c_label1, c_label2 = st.columns(2)
            if 'predicted_label' in df.columns:
                c_label1.metric("🤖 LLM 예측", selected_row['predicted_label'])
            if 'actual_label' in df.columns:
                c_label2.metric("🎯 실제 결과", selected_row['actual_label'])
            
            tab1, tab2 = st.tabs(["📝 LLM Chain-of-Thought 분석 결과", "📥 입력된 원본 프롬프트"])
            
            with tab1:
                if 'output' in selected_row and selected_row['output']:
                    st.markdown(selected_row['output'])
                else:
                    st.warning("이 데이터엔 모델의 output(CoT 추론) 내용이 비어있습니다.")
                    
            with tab2:
                if 'prompt' in selected_row:
                    st.text_area(label="전체 금융 데이터 프롬프트 구조", value=selected_row['prompt'], height=450, disabled=True)