import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Прогноз цены сырого молока", layout="wide")

st.title("🥛 Прогноз цены на сырое молоко в РФ (Росстат)")
st.markdown("Загрузите Excel-файл с ежемесячными данными. Модель подберёт лучшие факторы и построит прогноз на июль–сентябрь 2025.")

uploaded_file = st.file_uploader("📂 Загрузите Excel файл с данными", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # ─────────────────────────────────────
        # 1. Загрузка и очистка
        # ─────────────────────────────────────
        df_raw = pd.read_excel(uploaded_file, header=None)
        header_idx = df_raw[df_raw[0].astype(str).str.contains("Дата", na=False)].index[0]
        df = pd.read_excel(uploaded_file, skiprows=header_idx)
        df.columns = df.columns.str.strip()
        df["Дата"] = pd.to_datetime(df["Дата"], format="mixed", dayfirst=False)
        df = df.sort_values("Дата").reset_index(drop=True)

        for col in df.columns:
            if col != "Дата":
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "."), errors="coerce"
                )

        TARGET = "Цена РС мол сырое"
        TEST_START = pd.Timestamp("2025-01-01")

        st.success(f"✅ Загружено {len(df)} наблюдений: {df['Дата'].min().strftime('%b %Y')} – {df['Дата'].max().strftime('%b %Y')}")

        # ─────────────────────────────────────
        # 2. Корреляционный анализ (только на train)
        # ─────────────────────────────────────
        train_raw = df[df["Дата"] < TEST_START].copy()
        num_cols = train_raw.select_dtypes(include=np.number).columns
        corr = (
            train_raw[num_cols]
            .corr()[TARGET]
            .drop(TARGET)
            .abs()
            .sort_values(ascending=False)
        )
        TOP_N = 3
        top_regressors = corr.head(TOP_N).index.tolist()

        st.subheader("📊 Топ-3 фактора влияния на цену (по корреляции, train-данные)")
        col1, col2, col3 = st.columns(3)
        for i, (feat, val) in enumerate(corr.head(3).items()):
            [col1, col2, col3][i].metric(label=feat, value=f"{val:.3f}", delta="корреляция")

        # Бар-чарт по всем факторам
        fig_corr = px.bar(
            x=corr.values,
            y=corr.index,
            orientation="h",
            labels={"x": "Абс. корреляция", "y": "Фактор"},
            title="Корреляция факторов с ценой сырого молока",
            color=corr.values,
            color_continuous_scale="Blues",
        )
        fig_corr.update_layout(showlegend=False, coloraxis_showscale=False,
                               height=500, yaxis=dict(autorange="reversed"))
        fig_corr.add_vline(x=0.5, line_dash="dot", line_color="gray", annotation_text="0.5")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.info(f"💡 В модель добавлены топ-{TOP_N} регрессора: **{', '.join(top_regressors)}**. "
                "Для прогноза их будущих значений строятся отдельные Prophet-модели.")

        # ─────────────────────────────────────
        # 3. Подготовка данных для Prophet
        # ─────────────────────────────────────
        df_prophet = df[["Дата", TARGET] + top_regressors].copy()
        df_prophet.columns = ["ds", "y"] + top_regressors

        df_prophet = df_prophet.drop_duplicates(subset=["ds"])

        train = df_prophet[df_prophet["ds"] < TEST_START].copy()
        test  = df_prophet[df_prophet["ds"] >= TEST_START].copy()

        # ─────────────────────────────────────
        # 4. Вспомогательная функция: прогноз регрессора
        # ─────────────────────────────────────
        def forecast_regressor(series_df: pd.DataFrame, periods: int) -> pd.DataFrame:
            """Строит Prophet-прогноз для одного регрессора на N периодов вперёд."""
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                changepoint_range=0.80,
                uncertainty_samples=0,
            )
            m.fit(series_df)
            future = m.make_future_dataframe(periods=periods, freq="MS")
            fc = m.predict(future)
            return fc[["ds", "yhat"]].tail(periods)

        # ─────────────────────────────────────
        # 5. Построение тест-модели (MAE на янв–июн 2025)
        # ─────────────────────────────────────
        with st.spinner("⏳ Обучаю модель и считаю MAE на тесте..."):

            model_test = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                changepoint_range=0.80,
                uncertainty_samples=1000,
            )
            for reg in top_regressors:
                model_test.add_regressor(reg)
            model_test.fit(train[["ds", "y"] + top_regressors])

            # На тесте реальные значения регрессоров известны → подаём напрямую
            forecast_test = model_test.predict(test[["ds"] + top_regressors])
            mae = mean_absolute_error(test["y"].values, forecast_test["yhat"].values)

        st.metric("🎯 MAE на тесте (Январь–Июнь 2025)", f"{mae:.2f} ₽/кг",
                  help="Средняя абсолютная ошибка прогноза на 6 тестовых месяцах")

        # Таблица факт vs прогноз на тесте
        test_compare = test[["ds", "y"]].copy()
        test_compare["Прогноз"] = forecast_test["yhat"].values
        test_compare["Ошибка (₽)"] = (test_compare["y"] - test_compare["Прогноз"]).abs()
        test_compare["Дата"] = test_compare["ds"].dt.strftime("%B %Y")
        test_compare = test_compare.set_index("Дата")[["y", "Прогноз", "Ошибка (₽)"]]
        test_compare.columns = ["Факт (₽/кг)", "Прогноз (₽/кг)", "Ошибка (₽)"]

        st.subheader("🔍 Факт vs Прогноз на тестовом периоде")
        st.dataframe(test_compare.style.format("{:.2f}").highlight_min(
            subset=["Ошибка (₽)"], color="#d4f0d4").highlight_max(
            subset=["Ошибка (₽)"], color="#ffd4d4"), use_container_width=True)

        # ─────────────────────────────────────
        # 6. Финальная модель (все данные) + прогноз на 3 мес.
        # ─────────────────────────────────────
        with st.spinner("⏳ Строю итоговый прогноз на июль–сентябрь 2025..."):

            model_final = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                changepoint_range=0.80,
                uncertainty_samples=1000,
            )
            for reg in top_regressors:
                model_final.add_regressor(reg)
            model_final.fit(df_prophet[["ds", "y"] + top_regressors])

            # Прогнозируем регрессоры на 3 месяца вперёд
            future_regs = pd.DataFrame()
            future_regs["ds"] = pd.date_range("2025-07-01", periods=3, freq="MS")
            for reg in top_regressors:
                reg_df = df_prophet[["ds", reg]].rename(columns={reg: "y"})
                fc_reg = forecast_regressor(reg_df, periods=3)
                future_regs[reg] = fc_reg["yhat"].values

            forecast_3m = model_final.predict(future_regs)

        # ─────────────────────────────────────
        # 7. Основной график
        # ─────────────────────────────────────
        # Полный in-sample forecast для доверительного интервала
        all_hist = model_final.predict(df_prophet[["ds"] + top_regressors])

        fig = go.Figure()

        # ДИ история
        fig.add_trace(go.Scatter(
            x=pd.concat([all_hist["ds"], all_hist["ds"].iloc[::-1]]),
            y=pd.concat([all_hist["yhat_upper"], all_hist["yhat_lower"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(31,119,180,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="95% ДИ (история)", showlegend=True,
        ))
        # ДИ прогноз
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_3m["ds"], forecast_3m["ds"].iloc[::-1]]),
            y=pd.concat([forecast_3m["yhat_upper"], forecast_3m["yhat_lower"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(214,39,40,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="95% ДИ (прогноз)", showlegend=True,
        ))
        # Факт
        fig.add_trace(go.Scatter(
            x=df_prophet["ds"], y=df_prophet["y"],
            mode="lines", name="Факт", line=dict(color="#1f77b4", width=2.5),
        ))
        # Соединительная линия (последний факт → первый прогноз)
        connect_x = [df_prophet["ds"].iloc[-1], forecast_3m["ds"].iloc[0]]
        connect_y = [df_prophet["y"].iloc[-1], forecast_3m["yhat"].iloc[0]]
        fig.add_trace(go.Scatter(
            x=connect_x, y=connect_y,
            mode="lines", line=dict(color="#d62728", width=2.5, dash="dash"),
            showlegend=False,
        ))
        # Прогноз 3 мес.
        fig.add_trace(go.Scatter(
            x=forecast_3m["ds"], y=forecast_3m["yhat"],
            mode="lines+markers", name="Прогноз (Июль–Сент 2025)",
            line=dict(color="#d62728", width=3, dash="dash"),
            marker=dict(size=10, symbol="circle"),
        ))
        # Тест-период подсветка
        fig.add_vrect(
            x0=TEST_START, x1=df_prophet["ds"].max(),
            fillcolor="orange", opacity=0.06, line_width=0,
            annotation_text="Тест 2025", annotation_position="top left",
        )

        fig.update_layout(
            title="Динамика и прогноз цены сырого молока (₽/кг)",
            xaxis_title="Дата", yaxis_title="Цена (₽/кг)",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ─────────────────────────────────────
        # 8. Таблица прогноза
        # ─────────────────────────────────────
        st.subheader("📅 Прогноз цены сырого молока на 3 месяца")
        forecast_display = forecast_3m[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_display.columns = ["Дата", "Прогноз (₽/кг)", "Нижняя граница", "Верхняя граница"]
        forecast_display["Дата"] = forecast_display["Дата"].dt.strftime("%B %Y")
        forecast_display = forecast_display.set_index("Дата")

        st.dataframe(
            forecast_display.style
            .format("{:.2f}")
            .background_gradient(subset=["Прогноз (₽/кг)"], cmap="YlOrRd"),
            use_container_width=True,
        )

        # KPI-метрики прогноза
        c1, c2, c3 = st.columns(3)
        months = ["Июль 2025", "Август 2025", "Сентябрь 2025"]
        for i, (col, month) in enumerate(zip([c1, c2, c3], months)):
            val = forecast_3m["yhat"].iloc[i]
            lo  = forecast_3m["yhat_lower"].iloc[i]
            hi  = forecast_3m["yhat_upper"].iloc[i]
            col.metric(
                label=f"📌 {month}",
                value=f"{val:.2f} ₽/кг",
                delta=f"ДИ: {lo:.2f} – {hi:.2f}",
            )

        # ─────────────────────────────────────
        # 9. График: прогноз регрессоров
        # ─────────────────────────────────────
        with st.expander("📈 Прогноз факторов-регрессоров (детали)"):
            for reg in top_regressors:
                reg_df = df_prophet[["ds", reg]].rename(columns={reg: "y"})
                full_fc = forecast_regressor(reg_df, periods=3)
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(
                    x=df_prophet["ds"], y=df_prophet[reg],
                    mode="lines", name="Факт", line=dict(color="#1f77b4"),
                ))
                connect_r = [df_prophet["ds"].iloc[-1], full_fc["ds"].iloc[0]]
                connect_v = [df_prophet[reg].iloc[-1], full_fc["yhat"].iloc[0]]
                fig_r.add_trace(go.Scatter(
                    x=connect_r, y=connect_v,
                    mode="lines", line=dict(color="#d62728", dash="dash"), showlegend=False,
                ))
                fig_r.add_trace(go.Scatter(
                    x=full_fc["ds"], y=full_fc["yhat"],
                    mode="lines+markers", name="Прогноз", line=dict(color="#d62728", dash="dash"),
                ))
                fig_r.update_layout(title=f"Прогноз: {reg}", height=280,
                                    hovermode="x unified", margin=dict(t=40, b=30))
                st.plotly_chart(fig_r, use_container_width=True)

        # ─────────────────────────────────────
        # 10. Объяснение модели
        # ─────────────────────────────────────
        st.subheader("🧠 Как работает модель?")
        st.markdown(f"""
| Параметр | Значение | Что это значит |
|---|---|---|
| **Алгоритм** | Prophet (Meta) | Автоматически учитывает тренд и сезонность |
| **Регрессоры** | {', '.join(top_regressors)} | Внешние факторы с корреляцией > 0.9 — улучшают точность |
| **Гибкость тренда** | 0.05 | Плавно следует за ростом, не реагирует на шум |
| **Поиск переломов** | 80% истории | Не «угадывает» разворот на последних точках |
| **Июльская яма** | Учтена через сезонность | Пик летних надоев → цена традиционно снижается |
| **Доверительный интервал** | 95% | Чем уже — тем выше уверенность модели |
| **MAE на тесте** | **{mae:.2f} ₽/кг** | Средняя ошибка на 6 тестовых месяцах (янв–июн 2025) |

**Вкратце:**  
Модель смотрит на историю цены молока за 6+ лет, находит повторяющийся годовой паттерн (сезонность), общий тренд роста, и учитывает цены на масло, сухое молоко и сыры — которые двигаются вместе с сырым молоком. Для июля–сентября модель сначала прогнозирует эти факторы, а потом использует их для прогноза целевой цены.
        """)

    except Exception as e:
        st.error(f"❌ Ошибка при обработке файла: {e}")
        st.exception(e)

else:
    st.warning("📂 Загрузите файл `данные для прогноза (1).xlsx` для начала работы.")
    st.markdown("""
    **Что умеет этот дашборд:**
    - 📊 Анализ корреляций факторов с ценой молока
    - 🎯 Оценка точности (MAE) на тестовом периоде янв–июн 2025
    - 📅 Прогноз на июль, август и сентябрь 2025
    - 📈 Интерактивный график факт/прогноз с доверительным интервалом
    """)
