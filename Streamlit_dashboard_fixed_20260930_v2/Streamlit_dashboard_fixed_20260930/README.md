# 배치 증류 이상신호 검토 — Domain UI v2

목적: 현재 이상상태 Run 선별, 변화 시점 추적, 공정 태그 조사 순위 산출.
해석 흐름: `Run 검토 → 모델 검증 → 사전예측 검토 → 적용 조건`.

## v2 UI 변경

- 배치 증류탑·응축기·리보일러 라인아트 기반 전체 배경
- 온도·차압·유량·조작부 센서 계통 표시
- 흰 Plotly 캔버스 제거 및 반투명 공정 모니터링 테마 적용
- 정상 Run에서도 변동을 확인할 수 있는 자동 축 + 9-point median 곡선
- 모델 비교 막대그래프를 dot/lollipop 방식으로 변경
- 설명형 종결어미를 줄이고 검증 보고서형 명사 문장으로 통일
- 연간 배치 수와 이상 유병률 근거가 없는 `연간 조사 부하` 계산 제외

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

기본 주소는 http://localhost:8501 입니다.

## Streamlit Community Cloud 배포

1. GitHub의 기존 `Streamlit_dashboard_fixed_20260930` 폴더를 이 폴더 내용으로 교체
2. 새 앱을 배포할 때 **Python 3.12**를 선택합니다.
3. Main file path: `Streamlit_dashboard_fixed_20260930/app.py`
4. 기존 앱이 다른 Python 버전으로 만들어졌다면 앱을 삭제한 뒤 Python 3.12로 다시 배포합니다.

`model_v3.joblib`은 scikit-learn 1.8.0에서 생성됐으므로 `requirements.txt`의 버전을 임의로 올리지 않습니다.
이 앱은 외부 시스템 패키지가 없어 `packages.txt`가 필요하지 않습니다.

## 폴더 구성

```
app/
├── app.py                      Streamlit 앱
├── assets/
│   └── distillation_background.svg  배치 증류 공정 배경
├── requirements.txt
└── artifacts/
    ├── model_v3.joblib         RF · PCA · 스케일러 · 관리한계 · 성능 지표
    ├── roc_curve.csv           운영점 환산용 ROC
    ├── feature_importance.csv  공정변수별 기여도
    ├── sensitivity.csv         민감도 분석 결과
    ├── model_summary.csv       모델 성능 요약
    ├── metrics.json            전체 지표
    ├── sample_normal.csv.gz    정상 Run 샘플
    └── sample_anomalous.csv.gz 이상 Run 샘플
```

저장소에 다른 `uv.lock`, `Pipfile`, `environment.yml`, `pyproject.toml`이 함께 있으면
Community Cloud가 `requirements.txt`보다 먼저 인식할 수 있습니다. 이 대시보드 저장소에는
Python 의존성 파일을 `requirements.txt` 하나만 두는 것을 권장합니다.

## 배포 오류 확인

- `Error installing requirements`: Python 3.12 선택 여부와 `requirements.txt` 위치를 확인합니다.
- 모델 로딩 오류: `artifacts/model_v3.joblib` 업로드 여부와 scikit-learn 1.8.0 설치 여부를 확인합니다.
- 수정 사항이 반영되지 않음: GitHub의 파일 내용을 확인한 뒤 앱을 Reboot합니다.

## 화면 구성과 배치 이유

| 탭 | 내용 |
|---|---|
| **Run 검토** | q95 관리한계 → 시계열 변화 → 공정 태그 기여도 |
| **모델 검증** | 외부 분할 구조 · AUPRC 비교 · 평가조건 민감도 |
| **사전예측 검토** | 경과시간 기준선 비교 · 사전 게이트 판정 |
| **적용 조건** | 근거 확보 범위 · 추가 검증 범위 · 현장 전환 조건 |

기존 **연간 조사 부하** 계산 제외. 연간 배치 수와 실제 이상 유병률 근거가 없는 조건에서의
임의 환산값은 운영 KPI로 사용할 수 없음. 관리한계 설정은 `연구자용 임계값 설정`에 한정.

## 판정 방식

배치 단위 판정 지표는 **시점별 SPE 의 상위 5% 분위수**다.
시점 단위 관리한계를 배치에 그대로 적용하면 정상 배치도 다수 초과하므로,
Run 단위로 요약한 뒤 정상 Run 분위수 기준의 관리한계와 비교한다.

| 지표 | Run 단위 AUROC |
|---|---|
| max | 0.870 |
| **q95 (채택)** | **0.854** |
| mean | 0.833 |
| median | 0.707 |

max 가 근소하게 높으나 단일 시점에 좌우되어 불안정하므로 q95 를 채택했다.

## 입력 데이터 형식

분석 노트북 PART 3 에서 생성한 윈도 특성 CSV 를 사용한다.
필수 컬럼은 `t_end` 와 186개 특성입니다. 필수 특성이 누락되면 값을 임의로 채우지 않고
판정을 중단합니다. `run_id`가 있으면 화면에 Run 이름도 표시됩니다.

## 적용 범위

- 본 시스템은 일탈(Deviation)을 **판정하지 않는다**
- 기여도는 인과가 아니라 **조사 우선순위 후보**다
- 실험실 규모 연구 데이터 기반이며 GMP 생산 기록이 아니다
- 원료 로트·설비 변경 시 기준선 재평가가 필요하다 (변경관리 대상)
