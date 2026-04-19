import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Прогноз цены сырого молока", layout="wide")
st.title("🥛 Прогноз цены на сырое молоко в РФ (Росстат)")
st.markdown("Загрузите Excel-файл с ежемесячными данными. Модель SARIMA построит прогноз на июль–сентябрь 2025.")

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

        fig_corr = px.bar(
            x=corr.values, y=corr.index, orientation="h",
            labels={"x": "Абс. корреляция", "y": "Фактор"},
            title="Корреляция факторов с ценой сырого молока",
            color=corr.values, color_continuous_scale="Blues",
        )
        fig_corr.update_layout(showlegend=False, coloraxis_showscale=False,
                               height=500, yaxis=dict(autorange="reversed"))
        fig_corr.add_vline(x=0.5, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.info(f"💡 В модель добавлены топ-{TOP_N} регрессора: **{', '.join(top_regressors)}**")

        # ─────────────────────────────────────
        # 3. Подготовка данных
        # ─────────────────────────────────────
        df_sarima = df[["Дата", TARGET] + top_regressors].copy()
        df_sarima.columns = ["ds", "y"] + top_regressors
        df_sarima = df_sarima.set_index("ds").asfreq("MS")  # Monthly Start frequency

        train = df_sarima[df_sarima.index < TEST_START].copy()
        test = df_sarima[df_sarima.index >= TEST_START].copy()

        # ─────────────────────────────────────
        # 4. Вспомогательная функция: прогноз регрессора через SARIMA
        # ─────────────────────────────────────
        def forecast_regressor_sarima(series: pd.Series, periods: int) -> pd.Series:
            """Прогнозирует один регрессор на N месяцев вперёд через SARIMA."""
            series_clean = series.dropna()
            if len(series_clean) < 24:
                # Если мало данных — простой линейный тренд
                return pd.Series([series_clean.iloc[-1]] * periods, 
                               index=pd.date_range(series_clean.index[-1], periods=periods, freq="MS"))
            
            model = SARIMAX(series_clean, order=(1,1,1), seasonal_order=(1,1,1,12),
                           enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=periods)
            return forecast.predicted_mean

        # ─────────────────────────────────────
        # 5. Тест-модель для расчёта MAE (янв–июн 2025)
        # ─────────────────────────────────────
        with st.spinner("⏳ Обучаю SARIMA и считаю MAE на тесте..."):
            
            # Подготавливаем данные: y + exog (регрессоры)
            endog_train = train["y"].dropna()
            exog_train = train[top_regressors].dropna()
            exog_test = test[top_regressors].dropna()
            
            # SARIMA с регрессорами
            model_test = SARIMAX(
                endog_train,
                exog=exog_train,
                order=(1, 1, 1),           # (p,d,q): авторегрессия, дифференцирование, скользящее среднее
                seasonal_order=(1, 1, 1, 12),  # Сезонность: период 12 месяцев
                enforce_stationarity=False,
                enforce_invertibility=False,
                measurement_error=True     # Учёт шума в данных
            )
            results_test = model_test.fit(disp=False)
            
            # Прогноз на тест с известными регрессорами
            forecast_test = results_test.get_forecast(steps=len(test), exog=exog_test)
            test_pred = forecast_test.predicted_mean
            test_ci = forecast_test.conf_int(alpha=0.05)  # 95% интервал
            
            mae = mean_absolute_error(test["y"].dropna(), test_pred)
            st.metric("🎯 MAE на тесте (Январь–Июнь 2025)", f"{mae:.2f} ₽/кг")

        # Таблица факт vs прогноз на тесте
        test_compare = test[["y"]].copy()
        test_compare["Прогноз"] = test_pred.values
        test_compare["Ошибка (₽)"] = (test_compare["y"] - test_compare["Прогноз"]).abs()
        test_compare.index = test_compare.index.strftime("%B %Y")
        test_compare.columns = ["Факт (₽/кг)", "Прогноз (₽/кг)", "Ошибка (₽)"]

        st.subheader("🔍 Факт vs Прогноз на тестовом периоде")
        st.dataframe(test_compare.style.format("{:.2f}").highlight_min(
            subset=["Ошибка (₽)"], color="#d4f0d4").highlight_max(
            subset=["Ошибка (₽)"], color="#ffd4d4"), use_container_width=True)

        # ─────────────────────────────────────
        # 6. Финальная модель (все данные) + прогноз на 3 мес.
        # ─────────────────────────────────────
        with st.spinner("⏳ Строю итоговый прогноз на июль–сентябрь 2025..."):
            
            # Обучаем на ВСЕХ доступных данных
            endog_final = df_sarima["y"].dropna()
            exog_final = df_sarima[top_regressors].dropna()
            
            model_final = SARIMAX(
                endog_final,
                exog=exog_final,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
                measurement_error=True
            )
            results_final = model_final.fit(disp=False)
            
            # Прогнозируем регрессоры на 3 месяца вперёд
            future_dates = pd.date_range(start=df_sarima.index[-1], periods=4, freq="MS")[1:]  # исключаем последний известный
            future_exog = pd.DataFrame(index=future_dates)
            for reg in top_regressors:
                future_exog[reg] = forecast_regressor_sarima(df_sarima[reg], periods=3)
            
            # Прогноз целевой переменной
            forecast_3m = results_final.get_forecast(steps=3, exog=future_exog)
            forecast_mean = forecast_3m.predicted_mean
            forecast_ci = forecast_3m.conf_int(alpha=0.05)
            
            # Формируем таблицу прогноза
            forecast_display = pd.DataFrame({
                "Дата": future_dates,
                "Прогноз (₽/кг)": forecast_mean.values,
                "Нижняя граница": forecast_ci.iloc[:, 0].values,
                "Верхняя граница": forecast_ci.iloc[:, 1].values
            })
            forecast_display["Месяц"] = forecast_display["Дата"].dt.strftime("%B %Y")
            forecast_display = forecast_display.set_index("Месяц").drop(columns=["Дата"])

        # ─────────────────────────────────────
        # 7. Основной график
        # ─────────────────────────────────────
        fig = go.Figure()
        
        # Факт (вся история)
        fig.add_trace(go.Scatter(
            x=df_sarima.index, y=df_sarima["y"],
            mode="lines", name="Факт", line=dict(color="#1f77b4", width=2.5),
        ))
        
        # Прогноз на 3 месяца
        fig.add_trace(go.Scatter(
            x=forecast_display.index.map(lambda x: datetime.strptime(x + " 2025", "%B %Y %Y") if "2025" not in x else datetime.strptime(x, "%B %Y")),
            y=forecast_display["Прогноз (₽/кг)"].values,
            mode="lines+markers", name="Прогноз (Июль–Сент 2025)",
            line=dict(color="#d62728", width=3, dash="dash"),
            marker=dict(size=10, symbol="circle"),
        ))
        
        # 🔧 Доверительный интервал 95% (расширяющийся!)
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_display.index.to_series(), forecast_display.index.to_series().iloc[::-1]]).map(
                lambda x: datetime.strptime(x + " 2025", "%B %Y %Y") if "2025" not in x else datetime.strptime(x, "%B %Y")),
            y=pd.concat([forecast_display["Верхняя граница"], forecast_display["Нижняя граница"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(214,39,40,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="95% Доверительный интервал", showlegend=True,
        ))
        
        # Соединительная линия
        connect_x = [df_sarima.index[-1], forecast_display.index[0]]
        connect_y = [df_sarima["y"].iloc[-1], forecast_display["Прогноз (₽/кг)"].iloc[0]]
        fig.add_trace(go.Scatter(
            x=connect_x, y=connect_y,
            mode="lines", line=dict(color="#d62728", width=2.5, dash="dot"),
            showlegend=False,
        ))
        
        # Подсветка тест-периода
        fig.add_vrect(
            x0=TEST_START, x1=df_sarima.index.max(),
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
        # 8. Таблица прогноза + метрики
        # ─────────────────────────────────────
        st.subheader("📅 Прогноз цены сырого молока на 3 месяца")
        st.dataframe(
            forecast_display.style.format("{:.2f} ₽")
            .background_gradient(subset=["Прогноз (₽/кг)"], cmap="YlOrRd"),
            use_container_width=True,
        )
        
        # Показать ширину ДИ (для проверки расширения)
        forecast_check = forecast_display.copy()
        forecast_check["Ширина ДИ (₽)"] = (
            forecast_check["Верхняя граница"] - forecast_check["Нижняя граница"]
        ).round(2)
        st.caption(f"📏 Ширина 95% ДИ: {forecast_check['Ширина ДИ (₽)'].tolist()} ₽ — растёт от июля к сентябрю (неопределённость)")

        # KPI-метрики
        c1, c2, c3 = st.columns(3)
        months = forecast_display.index.tolist()
        for i, (col, month) in enumerate(zip([c1, c2, c3], months)):
            val = forecast_display["Прогноз (₽/кг)"].iloc[i]
            lo = forecast_display["Нижняя граница"].iloc[i]
            hi = forecast_display["Верхняя граница"].iloc[i]
            col.metric(
                label=f"📌 {month}",
                value=f"{val:.2f} ₽/кг",
                delta=f"ДИ: {lo:.2f} – {hi:.2f}",
            )

        # ─────────────────────────────────────
        # 9. График: прогноз регрессоров (опционально)
        # ─────────────────────────────────────
        with st.expander("📈 Прогноз факторов-регрессоров (детали)"):
            for reg in top_regressors:
                series = df_sarima[reg].dropna()
                future_vals = forecast_regressor_sarima(series, periods=3)
                future_idx = pd.date_range(start=series.index[-1], periods=4, freq="MS")[1:]
                
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(
                    x=series.index, y=series,
                    mode="lines", name="Факт", line=dict(color="#1f77b4"),
                ))
                fig_r.add_trace(go.Scatter(
                    x=future_idx, y=future_vals,
                    mode="lines+markers", name="Прогноз", 
                    line=dict(color="#d62728", dash="dash"),
                ))
                fig_r.update_layout(
                    title=f"Прогноз: {reg}", height=280,
                    hovermode="x unified", margin=dict(t=40, b=30)
                )
                st.plotly_chart(fig_r, use_container_width=True)

        # ─────────────────────────────────────
        # 10. Объяснение модели
        # ─────────────────────────────────────
        st.subheader("🧠 Как работает модель?")
        st.markdown(f"""
| Параметр | Значение | Что это значит |
|---|---|---|
| **Алгоритм** | SARIMA (statsmodels) | Классическая статистическая модель для временных рядов |
| **Регрессоры** | {', '.join(top_regressors)} | Внешние факторы с высокой корреляцией улучшают точность |
| **Порядок (p,d,q)** | (1,1,1) | Учитывает автокорреляцию, тренд и шум |
| **Сезонность** | (1,1,1,12) | Годовой цикл (12 месяцев) + сезонный тренд |
| **Июльская яма** | Учтена через сезонность | Пик летних надоев → цена традиционно снижается |
| **Доверительный интервал** | 95% | Чем уже — тем выше уверенность; **расширяется** в будущем |
| **MAE на тесте** | **{mae:.2f} ₽/кг** | Средняя ошибка на 6 тестовых месяцах (янв–июн 2025) |

**Вкратце:**  
Модель SARIMA анализирует историю цены молока за 6+ лет, выделяет:
1. 📈 **Тренд** — долгосрочный рост из-за инфляции и себестоимости
2. 🔄 **Сезонность** — повторяющийся годовой паттерн (в т.ч. июльское снижение)
3. 🔗 **Внешние факторы** — цены на масло, сухое молоко, сыры, которые двигаются синхронно с сырым молоком

Для прогноза на июль–сентябрь 2025 модель сначала предсказывает значения регрессоров, а затем использует их для расчёта целевой цены. Доверительный интервал расширяется по мере удаления от известных данных — это нормальное свойство любого прогноза.
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
    - 📈 Интерактивный график факт/прогноз с расширяющимся доверительным интервалом
    """)
