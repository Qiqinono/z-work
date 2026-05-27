# bond_risk_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 设置页面配置
st.set_page_config(
    page_title="债券违约预测可视化系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
custom_css = """
<style>
.main {
    background-color: #f0f4f8;
    min-height: 100vh;
}
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 50%, #415a77 100%);
    color: white;
    padding: 1rem;
}
.nav-button {
    background: rgba(255,255,255,0.1);
    color: white;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px;
    width: 100%;
    text-align: left;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 14px;
    font-weight: 500;
}
.nav-button:hover {
    background: rgba(255,255,255,0.2);
}
.stMetric {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    border-top: 4px solid #415a77;
}
.page-header {
    background: linear-gradient(135deg, #1b263b 0%, #415a77 100%);
    padding: 20px 30px;
    border-radius: 12px;
    margin-bottom: 24px;
}
.page-header h1 {
    color: white;
    font-size: 28px;
    margin: 0;
}
.page-header p {
    color: rgba(255,255,255,0.8);
    font-size: 14px;
    margin: 8px 0 0 0;
}
.chart-container {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 24px;
}
.chart-title {
    font-size: 18px;
    font-weight: 600;
    color: #1b263b;
    margin-bottom: 16px;
}
.chart-summary {
    background: linear-gradient(90deg, #e8f5f3 0%, #f0f7ff 100%);
    border-left: 4px solid #415a77;
    padding: 16px 20px;
    margin-top: 16px;
    border-radius: 0 8px 8px 0;
}
.chart-summary p {
    color: #4a5568;
    font-size: 14px;
    line-height: 1.6;
    margin: 0;
}
.logo {
    font-size: 24px;
    font-weight: 700;
    color: #72efdd;
    margin-bottom: 8px;
}
.logo-subtitle {
    font-size: 12px;
    color: rgba(255,255,255,0.6);
    margin-bottom: 24px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 加载数据
# ==========================================
@st.cache_data
def load_predictions():
    df = pd.read_csv('output_expanding/predictions_20250630.csv')
    # 如果没有 risk_level 列，根据 y12m_cum_prob 计算
    if 'risk_level' not in df.columns:
        df['risk_level'] = df['y12m_cum_prob'].apply(
            lambda x: '高风险' if x > 0.5 else ('中风险' if x >= 0.2 else '低风险')
        )
    return df

predictions = load_predictions()

# 计算统计数据
stats = {
    'total_bonds': len(predictions),
    'high_risk': (predictions['risk_level'] == '高风险').sum(),
    'medium_risk': (predictions['risk_level'] == '中风险').sum(),
    'low_risk': (predictions['risk_level'] == '低风险').sum(),
}

# 历史表现指标（从 fold1~fold3 测试集提取）
metrics_data = [
    ('Fold1', '6m', 0.9895, 0.6918, 0.7912, 0.3750),
    ('Fold1', '12m', 0.9647, 0.7174, 0.9121, 0.3374),
    ('Fold1', '18m', 0.9517, 0.6842, 0.9121, 0.2804),
    ('Fold1', '24m', 0.9423, 0.6868, 0.8837, 0.2331),
    ('Fold2', '6m', 0.9709, 0.5838, 0.7937, 0.3802),
    ('Fold2', '12m', 0.9774, 0.6059, 0.8103, 0.3219),
    ('Fold3', '6m', 0.9829, 0.6901, 0.7397, 0.3195),
    ('Fold3', '12m', 0.9848, 0.7247, 0.7808, 0.3024),
    ('Fold3', '18m', 0.9859, 0.7561, 0.8356, 0.3073),
    ('Fold3', '24m', 0.9866, 0.7388, 0.7808, 0.2774),
]
df_metrics = pd.DataFrame(metrics_data, columns=['fold', 'horizon', 'auc', 'prauc', 'top1_precision', 'top1_recall'])

# 初始化 session state
if 'selected_module' not in st.session_state:
    st.session_state['selected_module'] = '首页总览'

# ==========================================
# 首页总览
# ==========================================
def home_overview():
    st.markdown("""
    <div class="page-header">
        <h1>🏠 首页总览</h1>
        <p>快速了解当前债券市场的整体风险状况和核心指标</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("债券总数", f"{stats['total_bonds']:,}")
    with col2:
        st.metric("高风险债券", f"{stats['high_risk']:,}", delta=f"{stats['high_risk']/stats['total_bonds']*100:.1f}%")
    with col3:
        st.metric("中风险债券", f"{stats['medium_risk']:,}", delta=f"{stats['medium_risk']/stats['total_bonds']*100:.1f}%")
    with col4:
        st.metric("低风险债券", f"{stats['low_risk']:,}", delta=f"{stats['low_risk']/stats['total_bonds']*100:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container"><div class="chart-title">📈 12个月累积违约概率分布</div>', unsafe_allow_html=True)
        fig = px.histogram(predictions, x='y12m_cum_prob', nbins=50, color_discrete_sequence=['#415a77'], template='plotly_white')
        fig.update_layout(xaxis_title='违约概率', yaxis_title='债券数量')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="chart-summary"><p><strong>分析结论：</strong>大部分债券的12个月违约概率集中在0-30%区间，市场整体风险可控。</p></div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container"><div class="chart-title">📊 最可能违约时间段分布</div>', unsafe_allow_html=True)
        if 'most_likely_period' in predictions.columns:
            period_counts = predictions['most_likely_period'].value_counts().reset_index()
            period_counts.columns = ['违约时间段', '债券数量']
            fig = px.bar(period_counts, x='违约时间段', y='债券数量', color_discrete_sequence=['#72efdd'], template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无数据")
        st.markdown('<div class="chart-summary"><p><strong>分析结论：</strong>违约风险在时间维度上分布较为均匀。</p></div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container"><div class="chart-title">🥧 风险等级分布</div>', unsafe_allow_html=True)
    risk_data = pd.DataFrame({'风险等级': ['高风险', '中风险', '低风险'], '债券数量': [stats['high_risk'], stats['medium_risk'], stats['low_risk']]})
    fig = px.pie(risk_data, values='债券数量', names='风险等级',
                 color_discrete_map={'高风险': '#e53e3e', '中风险': '#ed8936', '低风险': '#38a169'},
                 template='plotly_white', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="chart-summary"><p><strong>分析结论：</strong>低风险债券占比最高，整体风险结构健康。</p></div></div>', unsafe_allow_html=True)

# ==========================================
# 模型历史表现
# ==========================================
def model_performance():
    st.markdown("""
    <div class="page-header">
        <h1>📈 模型历史表现</h1>
        <p>查看模型的架构设计和性能指标</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("模型架构")
    st.info("本系统采用XGBoost算法构建债券违约预测模型，使用Expanding Window方法进行时间序列交叉验证。")
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("模型类型", "XGBoost")
    with col2: st.metric("训练样本", "44,979")
    with col3: st.metric("验证样本", "14,693")
    
    st.markdown("---")
    st.subheader("性能指标")
    
    # 汇总各期限平均指标
    summary = df_metrics.groupby('horizon')[['auc', 'prauc', 'top1_precision', 'top1_recall']].mean().reset_index()
    summary.columns = ['预测期限', 'ROC-AUC', 'PR-AUC', 'Top1%精准率', 'Top1%召回率']
    
    fig = px.bar(summary.melt(id_vars=['预测期限'], var_name='指标', value_name='分数'),
                 x='预测期限', y='分数', color='指标', barmode='group',
                 template='plotly_white', color_discrete_sequence=['#415a77', '#72efdd', '#1b263b', '#0d1b2a'])
    fig.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="chart-summary"><p><strong>分析结论：</strong>模型在不同预测期限上均表现良好，ROC-AUC均超过0.94，具有较强的区分能力。</p></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 查询功能
# ==========================================
def query_function():
    st.markdown("""
    <div class="page-header">
        <h1>🔍 查询功能</h1>
        <p>搜索和查询债券信息，获取风险预测结果</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔎 债券代码搜索")
    
    bond_list = predictions['Liscd'].astype(str).tolist()
    selected_bond = st.selectbox("选择债券代码", bond_list)
    
    if selected_bond:
        bond = predictions[predictions['Liscd'].astype(str) == selected_bond].iloc[0]
        
        risk_level = bond['risk_level']
        prob = bond['y12m_cum_prob']
        risk_color = {"高风险": "#e53e3e", "中风险": "#ed8936", "低风险": "#38a169"}.get(risk_level, "#4a5568")
        
        top_1_threshold = np.percentile(predictions['y12m_cum_prob'], 99)
        is_top_1 = prob >= top_1_threshold
        most_likely = bond.get('most_likely_period', '未知')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div style="background:#fed7d7; border-radius:12px; padding:20px; text-align:center;"><div style="font-size:14px; color:#666;">风险等级</div><div style="font-size:28px; font-weight:700; color:{risk_color};">{risk_level}</div><div style="font-size:12px;">12个月概率: {prob:.1%}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div style="background:#ebf8ff; border-radius:12px; padding:20px; text-align:center;"><div style="font-size:14px; color:#666;">最可能违约时段</div><div style="font-size:28px; font-weight:700; color:#2b6cb0;">{most_likely}</div></div>', unsafe_allow_html=True)
        with col3:
            top_text = "⚠️ 是" if is_top_1 else "✅ 否"
            top_color = "#e53e3e" if is_top_1 else "#38a169"
            st.markdown(f'<div style="background:#fff5f5; border-radius:12px; padding:20px; text-align:center;"><div style="font-size:14px; color:#666;">Top 1%高风险</div><div style="font-size:28px; font-weight:700; color:{top_color};">{top_text}</div></div>', unsafe_allow_html=True)
        
        st.subheader("📊 各时间段违约概率")
        prob_df = pd.DataFrame({'时间周期': ['6个月', '12个月', '18个月', '24个月'],
                                '违约概率': [bond['y6m_cum_prob'], bond['y12m_cum_prob'], bond['y18m_cum_prob'], bond['y24m_cum_prob']]})
        fig = px.bar(prob_df, x='时间周期', y='违约概率', color='违约概率', color_continuous_scale='RdYlGn_r', template='plotly_white')
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📝 风险评估总结")
        st.write(f"**债券 {bond['Liscd']}** 的12个月违约概率为 **{prob:.1%}**，属于**{risk_level}**等级。{' ⚠️ 该债券属于市场Top 1%高风险债券！' if is_top_1 else ''}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 风险分析
# ==========================================
def risk_analysis():
    st.markdown("""
    <div class="page-header">
        <h1>📉 风险分析</h1>
        <p>分析债券违约风险的分布特征和变化趋势</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container"><div class="chart-title">📊 累积违约概率分布对比</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=2, subplot_titles=('6个月', '12个月', '18个月', '24个月'))
    fig.add_trace(go.Histogram(x=predictions['y6m_cum_prob'], marker_color='#0d1b2a'), row=1, col=1)
    fig.add_trace(go.Histogram(x=predictions['y12m_cum_prob'], marker_color='#1b263b'), row=1, col=2)
    fig.add_trace(go.Histogram(x=predictions['y18m_cum_prob'], marker_color='#415a77'), row=2, col=1)
    fig.add_trace(go.Histogram(x=predictions['y24m_cum_prob'], marker_color='#72efdd'), row=2, col=2)
    fig.update_layout(height=600, showlegend=False, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="chart-summary"><p><strong>分析结论：</strong>随着时间周期延长，违约概率分布逐渐右移，长期风险有所累积。</p></div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container"><div class="chart-title">📈 概率分布箱线图</div>', unsafe_allow_html=True)
    prob_data = pd.melt(predictions, id_vars=['Liscd'], value_vars=['y6m_cum_prob', 'y12m_cum_prob', 'y18m_cum_prob', 'y24m_cum_prob'], var_name='时间周期', value_name='违约概率')
    prob_data['时间周期'] = prob_data['时间周期'].map({'y6m_cum_prob': '6个月', 'y12m_cum_prob': '12个月', 'y18m_cum_prob': '18个月', 'y24m_cum_prob': '24个月'})
    fig = px.box(prob_data, x='时间周期', y='违约概率', color='时间周期', color_discrete_sequence=['#0d1b2a', '#1b263b', '#415a77', '#72efdd'], template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="chart-summary"><p><strong>分析结论：</strong>随着时间推移，违约概率的离散程度增加，长期预测不确定性更大。</p></div></div>', unsafe_allow_html=True)

# ==========================================
# 数据详情
# ==========================================
def data_details():
    st.markdown("""
    <div class="page-header">
        <h1>📋 数据详情</h1>
        <p>查看预测数据和统计摘要</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container"><div class="chart-title">📊 预测数据</div>', unsafe_allow_html=True)
    display_df = predictions[['Liscd', 'y6m_cum_prob', 'y12m_cum_prob', 'y18m_cum_prob', 'y24m_cum_prob', 'risk_level']].copy()
    display_df.columns = ['债券代码', '6个月', '12个月', '18个月', '24个月', '风险等级']
    st.dataframe(display_df.style.format({'6个月': '{:.2%}', '12个月': '{:.2%}', '18个月': '{:.2%}', '24个月': '{:.2%}'}), height=500, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 主应用
# ==========================================
def main_app():
    with st.sidebar:
        st.markdown('<div class="logo">📊 BondRisk</div>', unsafe_allow_html=True)
        st.markdown('<div class="logo-subtitle">债券违约预测系统</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        modules = ["首页总览", "模型历史表现", "查询功能", "风险分析", "数据详情"]
        for module in modules:
            if st.button(module, key=f"nav_{module}", use_container_width=True):
                st.session_state['selected_module'] = module
        
        st.markdown("---")
        st.info("数据更新: 2025-06-30")
        st.info("模型版本: XGBoost")
    
    module_map = {
        "首页总览": home_overview,
        "模型历史表现": model_performance,
        "查询功能": query_function,
        "风险分析": risk_analysis,
        "数据详情": data_details,
    }
    module_map[st.session_state['selected_module']]()

# ==========================================
# 入口
# ==========================================
if __name__ == "__main__":
    main_app()
