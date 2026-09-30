"""Batch-distillation sensor review dashboard (research decision support only)."""
import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="배치 증류 공정 모니터링", page_icon="⚗️", layout="wide")

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
NAVY, TEAL, AMBER, RED, GREEN = "#8AC9E8", "#4CC8C2", "#E4B76A", "#EF7A72", "#69C79F"
INK, MUTED, BORDER, BG = "#EDF7FA", "#9BB0BC", "rgba(151,204,218,.16)", "#071722"
bg_svg = ROOT / "assets" / "distillation_background.svg"
BG_URI = base64.b64encode(bg_svg.read_bytes()).decode("ascii") if bg_svg.exists() else ""

st.markdown(f"""
<style>
:root{{--ink:{INK};--muted:{MUTED};--line:{BORDER};}}
.stApp{{color:{INK};background-color:{BG};background-image:linear-gradient(180deg,rgba(4,14,22,.08),rgba(4,14,22,.58)),url('data:image/svg+xml;base64,{BG_URI}');background-size:cover;background-position:center top;background-attachment:fixed}}
.main .block-container{{max-width:1460px;padding:1.2rem 2rem 3.2rem}}
h1,h2,h3,p,label{{color:{INK}}}h1,h2,h3{{letter-spacing:-.03em}}h1{{font-size:2.15rem!important;margin:.05rem 0 .15rem!important}}
[data-testid="stHeader"]{{background:rgba(4,14,22,.55)}}
[data-testid="stSidebar"]{{background:rgba(7,18,29,.94);border-right:1px solid rgba(126,190,205,.16);backdrop-filter:blur(14px)}}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{{color:{INK}}}
[data-testid="stMetric"]{{background:linear-gradient(145deg,rgba(20,46,61,.91),rgba(11,29,41,.91));border:1px solid {BORDER};border-radius:14px;padding:.78rem .9rem;min-height:98px;box-shadow:0 12px 35px rgba(0,0,0,.18)}}
[data-testid="stMetricLabel"]{{color:{MUTED}}}[data-testid="stMetricValue"]{{color:{INK};font-size:1.4rem}}[data-testid="stMetricDelta"]{{font-size:.72rem}}
[data-testid="stTabs"] [data-baseweb="tab-list"]{{gap:1.2rem;border-bottom:1px solid rgba(140,202,217,.16)}}
[data-testid="stTabs"] [data-baseweb="tab"]{{padding:.9rem .12rem;font-weight:750;color:{MUTED}}}[data-testid="stTabs"] [aria-selected="true"]{{color:{TEAL}}}
[data-testid="stRadio"] label,[data-testid="stCaptionContainer"],.stCaption{{color:{MUTED}!important}}
[data-testid="stFileUploaderDropzone"]{{background:rgba(14,34,47,.86);border:1px dashed rgba(110,201,202,.38);border-radius:12px}}
[data-testid="stExpander"]{{background:rgba(10,27,39,.72);border:1px solid {BORDER};border-radius:12px}}
.eyebrow{{color:{TEAL};font-weight:850;font-size:.7rem;letter-spacing:.14em}}.subtitle{{color:#B5C6CF;font-size:.92rem;margin-bottom:.9rem}}
.badge{{display:inline-block;color:#B9ECE8;background:rgba(76,200,194,.12);border:1px solid rgba(76,200,194,.28);border-radius:999px;padding:.27rem .64rem;font-size:.7rem;font-weight:750}}
.hero{{background:linear-gradient(110deg,rgba(12,38,54,.94),rgba(11,30,43,.68));border:1px solid rgba(123,203,209,.18);border-radius:18px;padding:1rem 1.2rem;display:flex;align-items:center;justify-content:space-between;gap:1rem;margin:.65rem 0 1rem;box-shadow:0 18px 45px rgba(0,0,0,.22);backdrop-filter:blur(10px)}}
.hero .k{{color:#8CCFCB;font-size:.68rem;font-weight:800;letter-spacing:.1em}}.hero .v{{font-size:1.16rem;font-weight:850;margin-top:.18rem}}.hero .d{{color:#ACC0CA;font-size:.78rem;text-align:right;max-width:480px;line-height:1.5}}
.process{{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin:.55rem 0 1.05rem}}.unit{{position:relative;background:linear-gradient(145deg,rgba(17,42,57,.88),rgba(9,27,39,.88));border:1px solid {BORDER};border-radius:13px;padding:.72rem .78rem;box-shadow:0 10px 30px rgba(0,0,0,.14)}}
.unit:not(:last-child):after{{content:'›';position:absolute;right:-8px;top:20px;color:{TEAL};font-size:1.2rem;font-weight:850;z-index:2}}.unit.on{{border-top:3px solid {TEAL};padding-top:.6rem}}
.unit .n{{color:#7395A5;font-size:.61rem;font-weight:800;letter-spacing:.06em}}.unit .t{{color:{INK};font-size:.84rem;font-weight:850;margin-top:.1rem}}.unit .s{{color:{MUTED};font-size:.65rem;margin-top:.06rem}}
.status{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:.65rem 0 1rem}}.status>div{{background:linear-gradient(145deg,rgba(19,45,60,.9),rgba(10,28,40,.9));border:1px solid {BORDER};border-left:3px solid {TEAL};border-radius:13px;padding:.78rem .86rem;box-shadow:0 12px 32px rgba(0,0,0,.16)}}
.status .warn{{border-left-color:{AMBER}}}.status .bad{{border-left-color:{RED}}}.status .good{{border-left-color:{GREEN}}}.status .k{{color:{MUTED};font-size:.65rem;font-weight:750}}.status .v{{color:{INK};font-size:1.08rem;font-weight:850;margin:.15rem 0}}.status .d{{color:#819CAA;font-size:.67rem}}
.head{{display:flex;justify-content:space-between;align-items:end;margin:1.05rem 0 .52rem}}.head .t{{font-size:1.12rem;font-weight:850}}.head .d{{color:{MUTED};font-size:.73rem}}
.guide,.split{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:.3rem 0 1rem}}.guide>div,.split>div{{background:rgba(15,38,52,.86);border:1px solid {BORDER};border-radius:12px;padding:.72rem .78rem}}
.guide b,.split b{{display:block;color:{TEAL};font-size:.67rem;margin-bottom:.16rem;letter-spacing:.05em}}.guide span,.split span{{color:#C8D7DD;font-size:.74rem;line-height:1.42}}
.panel{{background:linear-gradient(145deg,rgba(18,43,57,.94),rgba(10,29,42,.94));border:1px solid {BORDER};border-radius:14px;padding:.95rem 1rem;margin-bottom:.75rem;box-shadow:0 14px 36px rgba(0,0,0,.16)}}
.ok{{border-left:4px solid {GREEN}}}.alert{{border-left:4px solid {RED}}}.panel .title{{font-size:1.04rem;font-weight:850}}.panel .copy{{color:#A9BDC7;font-size:.79rem;line-height:1.55;margin-top:.25rem}}
.note{{border-left:3px solid {TEAL};background:rgba(12,43,54,.74);color:#B9CCD4;padding:.7rem .82rem;font-size:.75rem;line-height:1.58;border-radius:0 9px 9px 0}}
.claim{{min-height:228px}}.claim h4{{margin:0 0 .45rem;color:{NAVY}}}.claim li{{margin:.38rem 0;color:#B8C9D1;font-size:.8rem}}
.domain-rail{{display:grid;grid-template-columns:1.2fr 1fr 1fr 1fr;gap:9px;margin:.2rem 0 1rem}}.domain-rail>div{{background:rgba(9,29,42,.78);border:1px solid {BORDER};border-radius:12px;padding:.75rem}}
.domain-rail b{{display:block;color:{NAVY};font-size:.76rem;margin-bottom:.2rem}}.domain-rail span{{color:{MUTED};font-size:.69rem;line-height:1.42}}
.footer{{color:#7792A0;font-size:.68rem;border-top:1px solid rgba(138,196,210,.14);padding-top:.75rem;margin-top:1.2rem}}
div[data-testid="stDataFrame"]{{border:1px solid {BORDER};border-radius:12px;overflow:hidden;background:rgba(8,23,34,.86)}}
div[data-testid="stDataFrame"] iframe{{background:transparent!important}}
@media(max-width:900px){{.process,.status,.guide,.split,.domain-rail{{grid-template-columns:1fr 1fr}}.hero{{display:block}}.hero .d{{text-align:left;margin-top:.5rem}}}}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="검증 모델 로딩")
def load_model():
    path = ART / "model_v3.joblib"
    if not path.exists():
        return None, "artifacts/model_v3.joblib 미확인"
    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, f"모델 로딩 실패: {type(exc).__name__}: {exc}"


@st.cache_data
def load_csv(name):
    path = ART / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_sample(kind):
    path = ART / f"sample_{kind}.csv.gz"
    return pd.read_csv(path) if path.exists() else None


MODEL, error = load_model()
if MODEL is None:
    st.error(error)
    st.info("배포 조건: Python 3.12 · 저장소 requirements.txt 적용")
    st.stop()

M, FEATS = MODEL["metrics"], MODEL["feature_names"]
SUMMARY = pd.DataFrame(MODEL["model_summary"])
ROC, SENS = load_csv("roc_curve.csv"), load_csv("sensitivity.csv")
DET = SUMMARY[SUMMARY.task == "y_detect"].set_index("model")
FC = SUMMARY[SUMMARY.task == "y_forecast"].set_index("model")


def spe_score(x):
    pca = MODEL["pca"]
    residual = x - pca.inverse_transform(pca.transform(x))
    return (residual ** 2).sum(axis=1), residual


def recall_at_fpr(target):
    if ROC is None:
        return np.nan, np.nan
    valid = ROC[ROC.fpr <= target]
    if valid.empty:
        return np.nan, np.nan
    row = valid.loc[valid.tpr.idxmax()]
    return float(row.tpr), float(row.thr)


def tag_context(tag):
    for prefix, text in [("PDI", "차압 계통"), ("FT", "유량 계통"), ("FYI", "유량 계통"),
                         ("LS", "액위 스위치"), ("AV", "밸브·조작 계통"), ("TV", "밸브·조작 계통")]:
        if tag.startswith(prefix):
            return text
    if tag.startswith("T"):
        return "온도 계통"
    if tag.startswith("P"):
        return "압력·구동 계통"
    if tag.startswith("H"):
        return "조작·가열 계통"
    return "공정 센서"


def section(title, desc=""):
    st.markdown(f'<div class="head"><div class="t">{title}</div><div class="d">{desc}</div></div>', unsafe_allow_html=True)


def style_plot(fig, height=350):
    fig.update_layout(
        height=height,
        margin=dict(t=42, b=48, l=38, r=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,26,38,.82)",
        font=dict(family="Arial, Noto Sans KR, sans-serif", color="#D9E8ED", size=12),
        title_font=dict(color=INK, size=14),
        legend=dict(orientation="h", y=1.15, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#102D3E", bordercolor="#4CC8C2", font_color=INK),
        hovermode="x unified",
        bargap=.34,
    )
    fig.update_xaxes(gridcolor="rgba(134,190,204,.10)", linecolor="rgba(134,190,204,.20)",
                     tickcolor="rgba(134,190,204,.25)", zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="rgba(134,190,204,.10)", linecolor="rgba(134,190,204,.20)",
                     tickcolor="rgba(134,190,204,.25)", zeroline=False, automargin=True)
    return fig


# Sidebar: only inputs the user can actually act on.
with st.sidebar:
    st.markdown('<div class="eyebrow">BATCH REVIEW</div>', unsafe_allow_html=True)
    st.header("분석 Run 선택")
    source = st.radio("데이터 소스", ["정상 예시 Run", "이상 예시 Run", "CSV 업로드"],
                      help="예시 Run: 화면 및 산출 로직 확인용")
    uploaded = st.file_uploader("윈도 특성 파일", type=["csv", "gz"]) if source == "CSV 업로드" else None
    if source == "CSV 업로드":
        st.caption("필수 스키마: 학습 파이프라인 186개 특성 + t_end")
    st.divider()
    st.markdown("**판독 순서**")
    st.caption("Run 선별 → 이상 시점 → 점검 변수 → 담당자 조사")
    with st.expander("연구자용 임계값 설정"):
        fpr_target = st.select_slider("Calibration 목표 오탐률", [0.01, 0.03, 0.05, 0.10, 0.20], value=0.05,
                                      format_func=lambda x: f"{x:.0%}")
        st.caption("5%: 모델 비교용 운영점 · 현장 관리한계: QA 조사 용량과 누락 위해도 기반 별도 승인")
    st.divider()
    st.markdown("**검증 데이터**")
    st.caption(f"{M['n_runs']} Run · {M['n_windows']:,} windows\n\n120초 window · 5초 stride · {M['n_splits']}개 OP 분할")
    st.markdown('<span class="badge">연구 검증용 · Deviation 판정 아님</span>', unsafe_allow_html=True)

if source == "정상 예시 Run":
    df = load_sample("normal")
elif source == "이상 예시 Run":
    df = load_sample("anomalous")
elif uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"CSV 판독 오류: {exc}")
        df = None
else:
    df = None

_, probability_threshold = recall_at_fpr(fpr_target)
spe_limit = MODEL.get("run_spe_limits", MODEL["spe_limits"]).get(f"{fpr_target:.2f}")

# Header: domain, project verdict, process context.
st.markdown('<div class="eyebrow">BATCH DISTILLATION · PROCESS ANALYTICS</div>', unsafe_allow_html=True)
st.title("배치 증류 이상신호 검토")
st.markdown('<div class="subtitle">공정 센서 기반 Run 선별 · 이탈 시점 추적 · QA 원인조사 후보 도출</div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div><div class="k">PROJECT STATUS</div><div class="v">현재 이상상태 분리 가능 · 60초 사전예측 근거 미확보</div></div>'
            '<div class="d">적용 범위: 관리한계 초과 Run 선별 및 조사 우선순위 제안<br>품질 일탈·출하 판정은 승인 시험법과 품질시스템 절차의 적용 대상</div></div>', unsafe_allow_html=True)
st.markdown('<div class="process">'
            '<div class="unit"><div class="n">01 · CHARGE</div><div class="t">원료 투입</div><div class="s">조성·투입 조건</div></div>'
            '<div class="unit"><div class="n">02 · HEAT</div><div class="t">가열·기화</div><div class="s">가열·압력 신호</div></div>'
            '<div class="unit on"><div class="n">03 · SEPARATE</div><div class="t">증류탑 분리</div><div class="s">온도·차압 프로파일</div></div>'
            '<div class="unit on"><div class="n">04 · CONDENSE</div><div class="t">응축·유량</div><div class="s">유량·밸브 신호</div></div>'
            '<div class="unit"><div class="n">05 · REVIEW</div><div class="t">품질 검토</div><div class="s">GC·배치기록 확인</div></div></div>', unsafe_allow_html=True)
st.markdown('<div class="domain-rail">'
            '<div><b>탑 온도 프로파일 · T701–T712</b><span>분리 진행과 조성 변화를 반영하는 축방향 온도 계열</span></div>'
            '<div><b>차압 · PDI701–PDI702</b><span>탑 내부 유동·부하 변화 점검</span></div>'
            '<div><b>유량 · FT703/704 · FYI702</b><span>응축·환류·제품 유출 계통 확인</span></div>'
            '<div><b>조작부 · H/AV/TV</b><span>가열·밸브 명령과 센서 반응 대조</span></div>'
            '</div>', unsafe_allow_html=True)

rf_det, rf_fc, time_fc = DET.loc["RF", "auroc"], FC.loc["RF", "auroc"], FC.loc["TimeOnly", "auroc"]
st.markdown(f'<div class="status"><div class="good"><div class="k">현재 이상상태 탐지</div><div class="v">검증됨</div><div class="d">RF AUROC {rf_det:.3f}</div></div>'
            f'<div class="bad"><div class="k">60초 사전 예측</div><div class="v">미입증</div><div class="d">RF {rf_fc:.3f} · 시간대조군 {time_fc:.3f}</div></div>'
            f'<div class="warn"><div class="k">정상구간 오경보</div><div class="v">{M["fa_per_hour"]:.2f}회/h</div><div class="d">사전 기준 1.0회/h 미충족</div></div>'
            f'<div><div class="k">검증 설계</div><div class="v">{M["n_splits"]}-fold</div><div class="d">System+OP 기준 외부 분할</div></div></div>', unsafe_allow_html=True)

tab_run, tab_valid, tab_early, tab_scope = st.tabs(["01  Run 검토", "02  모델 검증", "03  사전예측 검토", "04  적용 조건"])

with tab_run:
    section("Run 이상신호 검토", "Run 대표점수 → 변화 구간 → 기여 태그 → 품질조사")
    st.markdown('<div class="guide"><div><b>01 · SCREEN</b><span>q95 SPE 기반 Run 선별</span></div><div><b>02 · LOCATE</b><span>시계열 변화 및 최대 이탈 확인</span></div>'
                '<div><b>03 · TRACE</b><span>온도·차압·유량 기여 태그 추적</span></div><div><b>04 · REVIEW</b><span>배치기록·P&amp;ID·GC 결과 대조</span></div></div>', unsafe_allow_html=True)
    if df is None:
        st.info("분석 대기 · 사이드바에서 예시 Run 선택 또는 윈도 특성 CSV 업로드")
    else:
        missing = [c for c in FEATS if c not in df.columns]
        if missing:
            st.error(f"입력 스키마 불일치 · 필수 특성 {len(missing)}개 누락 · Run 산출 중단")
        else:
            x = MODEL["scaler"].transform(df[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0))
            spe, residual = spe_score(x)
            probability = MODEL["rf"].predict_proba(x)[:, 1]
            time_axis = df["t_end"].to_numpy() if "t_end" in df else np.arange(len(df)) * MODEL["stride_sec"]
            run_score, peak_idx = float(np.quantile(spe, .95)), int(np.argmax(spe))
            ratio, alarm = run_score / spe_limit, run_score >= spe_limit
            peak_time = float(time_axis[peak_idx])
            run_name = str(df["run_id"].dropna().iloc[0]).split("|")[-1] if "run_id" in df else "업로드 Run"
            left, right = st.columns([1.25, 2.4])
            with left:
                title = "추가 조사 대상" if alarm else "연구 관리한계 이내"
                copy = ("q95 SPE의 calibration 관리한계 초과. 최대 이탈 시점과 상위 기여 태그를 중심으로 배치기록·설비 상태 대조 필요."
                        if alarm else "q95 SPE 기준의 지속 이탈 미확인. 배치 적합 판정과는 별개이며 기존 시험·기록 검토 절차 유지.")
                st.markdown(f'<div class="panel {"alert" if alarm else "ok"}"><div style="color:{MUTED};font-size:.72rem">{run_name}</div>'
                            f'<div class="title">{title}</div><div class="copy">{copy}</div></div>', unsafe_allow_html=True)
                a, b = st.columns(2); a.metric("Run q95 점수", f"{run_score:,.1f}"); b.metric("관리한계", f"{spe_limit:,.1f}")
                c, d = st.columns(2); c.metric("한계 대비", f"{ratio:.2f}배"); d.metric("최대 이탈", f"{peak_time:,.0f}초")
                st.markdown('<div class="note"><b>Run 대표지표 선정</b><br>단일 스파이크의 영향 완화를 위해 시점별 SPE의 95백분위수 채택. 관리한계는 test와 분리된 calibration Run에서 산출.</div>', unsafe_allow_html=True)
            with right:
                smooth_window = 9
                spe_smooth = pd.Series(spe).rolling(smooth_window, center=True, min_periods=1).median().to_numpy()
                prob_smooth = pd.Series(probability).rolling(smooth_window, center=True, min_periods=1).median().to_numpy()
                spe_pad = max((np.nanmax(spe_smooth) - np.nanmin(spe_smooth)) * .14, .08)
                prob_pad = max((np.nanmax(prob_smooth) - np.nanmin(prob_smooth)) * .14, .015)
                spe_range = [max(0, np.nanmin(spe_smooth) - spe_pad), np.nanmax(spe_smooth) + spe_pad]
                prob_range = [max(0, np.nanmin(prob_smooth) - prob_pad), min(1, np.nanmax(prob_smooth) + prob_pad)]
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.13, row_heights=[.62, .38],
                                    subplot_titles=("PCA-SPE 변화 프로파일 · 자동 축", "RF 이상확률 변화 프로파일 · 자동 축"))
                fig.add_trace(go.Scatter(x=time_axis, y=spe, name="시점 SPE", line=dict(color="rgba(122,184,201,.24)", width=1),
                                         hovertemplate="%{x:.0f}초 · SPE %{y:.2f}<extra></extra>"), row=1, col=1)
                fig.add_trace(go.Scatter(x=time_axis, y=spe_smooth, name="9-point median", line=dict(color=TEAL, width=2.8, shape="spline", smoothing=.7),
                                         fill="tozeroy", fillcolor="rgba(76,200,194,.09)", hovertemplate="%{x:.0f}초 · 평활 SPE %{y:.2f}<extra></extra>"), row=1, col=1)
                fig.add_trace(go.Scatter(x=[peak_time], y=[spe[peak_idx]], name="최대 이탈", mode="markers",
                                         marker=dict(color=AMBER, size=10, line=dict(color="#FFF0C8", width=1.5))), row=1, col=1)
                fig.add_trace(go.Scatter(x=time_axis, y=probability, name="시점 확률", line=dict(color="rgba(138,201,232,.24)", width=1),
                                         hovertemplate="%{x:.0f}초 · %{y:.1%}<extra></extra>"), row=2, col=1)
                fig.add_trace(go.Scatter(x=time_axis, y=prob_smooth, name="확률 추세", line=dict(color=NAVY, width=2.5, shape="spline", smoothing=.7),
                                         fill="tozeroy", fillcolor="rgba(138,201,232,.07)", hovertemplate="%{x:.0f}초 · 평활 %{y:.1%}<extra></extra>"), row=2, col=1)
                fig.update_yaxes(title_text="SPE", range=spe_range, row=1, col=1)
                fig.update_yaxes(title_text="이상확률", tickformat=".0%", range=prob_range, row=2, col=1)
                fig.update_xaxes(title_text="Run 경과 시간 (초)", row=2, col=1)
                style_plot(fig, 470); st.plotly_chart(fig, width="stretch")
                st.caption(f"시계열 축: Run 내부 변화 확인용 자동 범위 · 판정 기준: q95 SPE {run_score:,.1f} / 관리한계 {spe_limit:,.1f}")

            section("공정 태그 기여도", "최대 이탈 시점의 PCA 잔차 분해 · 원인 확정값 아님")
            contribution = pd.Series(residual[peak_idx] ** 2, index=FEATS)
            tags = contribution.groupby(contribution.index.str.split("|").str[0]).sum().sort_values(ascending=False).head(7)
            share = tags / max(contribution.sum(), 1e-12)
            chart_col, note_col = st.columns([1.65, 1])
            with chart_col:
                bar = go.Figure(go.Bar(x=share.values[::-1], y=share.index[::-1], orientation="h", marker_color=TEAL,
                                       text=[f"{v:.1%}" for v in share.values[::-1]], textposition="outside"))
                bar.update_xaxes(title="최대 이탈 시점의 SPE 기여 비중", tickformat=".0%"); style_plot(bar, 320)
                st.plotly_chart(bar, width="stretch")
            with note_col:
                st.dataframe(pd.DataFrame({"태그": share.index, "확인 영역": [tag_context(x) for x in share.index],
                                           "기여": [f"{v:.1%}" for v in share.values]}), hide_index=True, width="stretch", height=270)
                st.markdown(f'<div class="note"><b>해석 범위</b><br>{peak_time:,.0f}초의 PCA 잔차 기여도. 인과 추론이 아닌 조사 순위이며, 태그 정의·설비 위치·조치 기준은 P&amp;ID와 SOP 대조 대상.</div>', unsafe_allow_html=True)

with tab_valid:
    section("검증 구조", "System·Operating Point 독립성 유지 및 test 정보 차단")
    st.markdown('<div class="split"><div><b>01 · FIT</b><span>전처리와 모델 학습</span></div><div><b>02 · EARLY STOP</b><span>학습 중단 시점 선택</span></div>'
                '<div><b>03 · CALIBRATION</b><span>관리한계 결정</span></div><div><b>04 · TEST</b><span>최종 평가만 수행</span></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="note"><b>분할 설계 근거</b><br>동일 Run의 인접 시점 간 강한 자기상관을 고려해 시점 무작위 분할 제외. System+Operating Point 기준 15개 외부 분할로 미관측 운전조건 성능 평가.</div>', unsafe_allow_html=True)
    section("현재 이상상태 분류 성능", "양성비율 15.9% · AUROC와 AUPRC 병행 평가")
    dp = DET.reset_index().sort_values("auprc")
    colors = ["#7D919D" if x == "TimeOnly" else (GREEN if x == "ConvAE" else TEAL) for x in dp.model]
    fig = go.Figure()
    for _, row in dp.iterrows():
        fig.add_trace(go.Scatter(x=[M["detect_prevalence"], row.auprc], y=[row.model, row.model], mode="lines",
                                 line=dict(color="rgba(122,177,190,.25)", width=5), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dp.auprc, y=dp.model, mode="markers+text", marker=dict(size=17, color=colors,
                             line=dict(color="rgba(237,247,250,.65)", width=1.2)),
                             error_x=dict(type="data", array=dp.auprc_sd, color="#91A8B4", thickness=1.2),
                             text=[f"  {v:.3f}" for v in dp.auprc], textposition="middle right", showlegend=False,
                             hovertemplate="%{y}<br>AUPRC %{x:.3f}<extra></extra>"))
    fig.add_vline(x=M["detect_prevalence"], line=dict(color=RED, dash="dash"), annotation_text=f"무작위 기준 {M['detect_prevalence']:.3f}")
    fig.update_xaxes(title="AUPRC · 양성 구간 선별 성능", range=[0, .8]); style_plot(fig, 320); st.plotly_chart(fig, width="stretch")
    cols = st.columns(4)
    for col, model in zip(cols, ["RF", "XGBoost", "ConvAE", "TimeOnly"]):
        col.metric(model, f"{DET.loc[model,'auroc']:.3f} AUROC", f"AUPRC {DET.loc[model,'auprc']:.3f}")
    st.markdown('<div class="note"><b>모델 비교 해석</b><br>정상 Run만으로 학습한 ConvAE의 성능 범위가 RF·XGBoost와 중첩. fold 간 표준편차를 고려할 때 모델 우열보다 센서 공간에서의 현재 이상상태 분리 가능성에 의미.</div>', unsafe_allow_html=True)
    if SENS is not None:
        section("평가조건 민감도 검토", "라벨 경계 제외폭과 Recovery 포함 여부에 따른 성능 변동")
        l, r = st.columns(2); boundary = SENS[SENS.analysis == "label_boundary"]
        with l:
            f = go.Figure(go.Scatter(x=boundary.setting, y=boundary.auprc, mode="lines+markers+text", line=dict(color=TEAL, width=2.5, shape="spline", smoothing=.7),
                                     text=[f"{v:.3f}" for v in boundary.auprc], textposition="top center"))
            f.update_yaxes(title="AUPRC", range=[.45, .53]); f.update_xaxes(title="라벨 경계 제외"); style_plot(f, 270); st.plotly_chart(f, width="stretch")
            st.caption("라벨 경계 ±0~30초 제외 시 AUPRC 변동 0.008 이내 · 경계 오차 영향 제한적")
        with r:
            recovery = SENS[SENS.analysis == "recovery"]
            f = go.Figure(go.Bar(x=recovery.setting, y=recovery.auprc, marker_color=[TEAL, AMBER], text=[f"{v:.3f}" for v in recovery.auprc], textposition="outside"))
            f.update_yaxes(title="AUPRC", range=[0, .75]); f.update_xaxes(title="Recovery 구간"); style_plot(f, 270); st.plotly_chart(f, width="stretch")
            st.caption("Recovery 포함 시 양성 구간 동시 확대 · 성능 상승과 모델 개선을 구분해 해석")

with tab_early:
    section("사전예측 타당성 검토", "현재 상태 탐지와 센서 관찰 이전 예측의 근거 분리")
    st.markdown('<div class="panel alert"><div class="title">판정: 60초 사전예측 근거 미확보</div>'
                '<div class="copy">RF AUROC 0.643, 경과시간 기준선 0.668. 센서 모델의 기준선 우위가 없어 해당 성능을 선행 신호로 해석하기 어려운 상태.</div></div>', unsafe_allow_html=True)
    det_best, det_time = DET.drop(index=["TimeOnly"]).auroc.max(), DET.loc["TimeOnly", "auroc"]
    fc_best, fc_time = FC.drop(index=["TimeOnly", "Dummy"]).auroc.max(), FC.loc["TimeOnly", "auroc"]
    tasks = ["현재 이상상태 탐지", "60초 사전 예측"]
    sensor_values, baseline_values = [det_best, fc_best], [det_time, fc_time]
    fig = go.Figure()
    for task, sensor, baseline in zip(tasks, sensor_values, baseline_values):
        fig.add_trace(go.Scatter(x=[baseline, sensor], y=[task, task], mode="lines",
                                 line=dict(color="rgba(132,186,199,.30)", width=7), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sensor_values, y=tasks, mode="markers+text", name="센서 모델",
                             marker=dict(size=18, color=TEAL, line=dict(color="#DDF8F5", width=1.2)),
                             text=[f"{v:.3f}" for v in sensor_values], textposition="top center"))
    fig.add_trace(go.Scatter(x=baseline_values, y=tasks, mode="markers+text", name="경과시간 기준선",
                             marker=dict(size=18, color=AMBER, symbol="diamond", line=dict(color="#FFF0C8", width=1.2)),
                             text=[f"{v:.3f}" for v in baseline_values], textposition="bottom center"))
    fig.add_vline(x=.5, line=dict(color=RED, dash="dot"), annotation_text="무작위 0.5")
    fig.update_xaxes(title="AUROC", range=[.4, .9]); style_plot(fig, 320); st.plotly_chart(fig, width="stretch")
    a, b, c, d = st.columns(4)
    a.metric("관찰 이전 탐지", f"{M['early_hits']} / {M['early_total']}", f"{M['early_rate']:.1%}")
    b.metric("120초 이내 탐지", f"{M['within120']} / {M['early_total']}", f"{M['within120_rate']:.1%}")
    c.metric("정상구간 오경보", f"{M['fa_per_hour']:.2f}회/h", "기준 초과", delta_color="inverse")
    d.metric("Gate 판단", "미입증", "5개 중 2개 미충족", delta_color="inverse")
    gate = pd.DataFrame([
        ["정상구간 오경보", "≤ 1.0회/h", f"{M['fa_per_hour']:.2f}회/h", "미충족"], ["관찰 이전 탐지", "1건 이상", f"{M['early_hits']}건", "통과"],
        ["리드타임", "30초 이상", f"탐지 건 중앙값 {M['lead_median']:.0f}초", "통과"], ["현재 탐지의 시간대조군 우위", "> 0", f"{det_best-det_time:+.3f}", "통과"],
        ["사전 예측의 시간대조군 우위", "> 0", f"{fc_best-fc_time:+.3f}", "미충족"]], columns=["검증 항목", "사전 기준", "실측값", "판정"])
    st.dataframe(gate, hide_index=True, width="stretch")
    st.markdown('<div class="note"><b>리드타임 해석 유의사항</b><br>리드타임 중앙값은 탐지 성공 사건만을 대상으로 산출. 전체 234건 중 관찰 이전 탐지 19건, 오경보 기준과 경과시간 기준선 비교 미충족. 성공 사례의 긴 리드타임만으로 전체 사전예측 성립 판단 불가.</div>', unsafe_allow_html=True)

with tab_scope:
    section("적용 조건 및 해석 범위", "연구 결과와 현장 품질판정의 경계")
    l, r = st.columns(2)
    with l:
        st.markdown('<div class="panel claim"><h4>근거가 확보된 범위</h4><ul><li>현재 이상상태와 정상상태의 센서 공간 분리</li>'
                    '<li>정상 Run 기반 ConvAE와 지도학습 후보군의 유사 성능 범위</li><li>Run/OP 그룹 분리 및 독립 calibration 기반 누수 통제</li>'
                    '<li>PCA-SPE 잔차 기반 공정 태그 조사 순위 산출</li><li>연구용 Run 선별·원인조사 지원 도구</li></ul></div>', unsafe_allow_html=True)
    with r:
        st.markdown('<div class="panel claim"><h4>추가 검증이 필요한 범위</h4><ul><li>센서 관찰 60초 이전의 안정적 사전예측</li>'
                    '<li>배치 적합·부적합 및 품질 일탈 자동판정</li><li>상위 기여 태그와 고장 원인의 인과관계</li>'
                    '<li>GMP 생산 규모에서의 외부 타당성</li><li>원료·설비·운전조건 변경 후 관리한계 유지 가능성</li></ul></div>', unsafe_allow_html=True)
    section("현장 전환 선결조건", "데이터 정의 · 관리한계 · 설비 해석 · 외부 검증")
    actions = pd.DataFrame([
        ["이벤트 기준", "explicit/inferred onset과 신뢰도 기록", "조기예측 라벨 불확실성 축소"],
        ["정상 대조군", "동일 chemical system·OP·진행률 Run 확대", "운전조건 차이를 이상으로 오인하지 않기"],
        ["관리한계", "QA 조사 가능량·누락 위해도로 승인", "연구용 5% 운영점과 현장 기준 분리"],
        ["설비 해석", "P&ID·태그 사전과 조치 SOP 연결", "기여 후보를 실제 점검 항목으로 전환"],
        ["외부 검증", "새 Run·새 시기에서 잠금 모델 재평가", "재현성과 데이터 드리프트 확인"]], columns=["보완 영역", "필요 작업", "이유"])
    st.dataframe(actions, hide_index=True, width="stretch")
    with st.expander("이 화면을 이 순서로 배치한 이유"):
        st.markdown("""
1. **Run 선택의 사이드바 분리** — 입력 조건과 결과 영역의 시각적 분리
2. **프로젝트 판단의 상단 고정** — 현재 이상탐지와 사전예측 결론의 혼동 방지
3. **배치 증류 공정 흐름 선행 배치** — 센서·설비·품질검토 맥락 제시
4. **시계열 이후 기여 태그 배치** — 기여도와 원인의 오인 방지
5. **검증·한계의 별도 탭 구성** — Run 검토 화면의 집중도와 연구 근거의 투명성 확보

기존 **연간 조사 부하** 계산 제외. 연간 배치 수와 실제 이상 유병률 근거가 없는 조건에서의
임의 환산값은 검증되지 않은 운영 KPI로 오인될 가능성 존재.
        """)
    with st.expander("모델·데이터 상세"):
        st.dataframe(pd.DataFrame({"항목": ["Run", "윈도", "특성", "외부 분할", "윈도 길이", "이동 간격", "초기 제외"],
                                   "값": [f"{M['n_runs']}개", f"{M['n_windows']:,}개", f"{M['n_features']}개", f"{M['n_splits']}개",
                                          f"{MODEL['window_sec']}초", f"{MODEL['stride_sec']}초", f"{MODEL['warmup_sec']}초"]}),
                     hide_index=True, width="stretch")
        st.caption("Run 검토: PCA-SPE 1차 관리지표 · Random Forest 확률 보조지표 · ConvAE 비교 검증 전용")

st.markdown('<div class="footer">RESEARCH VALIDATION ONLY · 일탈 및 출하 판정은 승인 시험법·배치기록·품질시스템 절차의 적용 대상</div>', unsafe_allow_html=True)
