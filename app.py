import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(page_title="BIST Teknik Analiz", page_icon="📈", layout="wide")

st.title("📈 BIST Fiyat, Hacim ve Teknik Göstergeler")
st.warning(
    "Veriler yfinance üzerinden gelir; gecikmeli veya eksik olabilir. "
    "Bu uygulama canlı emir defteri/derinlik verisi göstermez ve yatırım tavsiyesi değildir."
)


@st.cache_data(ttl=60)
def get_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.Ticker(symbol).history(
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=False,
    )

    if data.empty:
        return data

    # Bazı veri kaynaklarında sütun adları küçük harfli olabilir.
    data.columns = [str(column).title() for column in data.columns]

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(data.columns):
        return pd.DataFrame()

    return data.dropna(subset=["Close"])


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # Üssel hareketli ortalamalar
    data["EMA20"] = close.ewm(span=20, adjust=False).mean()
    data["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # RSI (14)
    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI14"] = 100 - (100 / (1 + rs))
    data.loc[avg_loss == 0, "RSI14"] = 100
    data.loc[(avg_gain == 0) & (avg_loss == 0), "RSI14"] = 50

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACDSignal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACDHist"] = data["MACD"] - data["MACDSignal"]

    # Bollinger Bantları (20 dönem, 2 standart sapma)
    middle = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    data["BBMiddle"] = middle
    data["BBUpper"] = middle + (2 * std)
    data["BBLower"] = middle - (2 * std)

    # Göreli hacim: son hacim / son 20 barın ortalama hacmi
    data["VolumeAvg20"] = volume.rolling(window=20).mean()
    data["RelativeVolume"] = volume / data["VolumeAvg20"].replace(0, np.nan)

    # ATR (14): ortalama gerçek fiyat aralığı
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["ATR14"] = true_range.rolling(window=14).mean()

    return data


def number_or_dash(value, decimals=2) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}"


def make_chart(data: pd.DataFrame, symbol: str):
    chart_data = data.tail(250)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="Fiyat",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["EMA20"],
            name="EMA 20",
            line={"width": 1.5, "color": "#f0a202"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["EMA50"],
            name="EMA 50",
            line={"width": 1.5, "color": "#3a86ff"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["BBUpper"],
            name="Bollinger üst",
            line={"width": 1, "color": "#999", "dash": "dot"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["BBLower"],
            name="Bollinger alt",
            line={"width": 1, "color": "#999", "dash": "dot"},
        ),
        row=1,
        col=1,
    )

    colors = [
        "#2ca02c" if close >= open_ else "#d62728"
        for open_, close in zip(chart_data["Open"], chart_data["Close"])
    ]

    fig.add_trace(
        go.Bar(
            x=chart_data.index,
            y=chart_data["Volume"],
            name="Hacim",
            marker_color=colors,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{symbol} — fiyat ve hacim",
        height=650,
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        legend_y=1.02,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
    )
    fig.update_yaxes(title_text="Fiyat", row=1, col=1)
    fig.update_yaxes(title_text="Hacim", row=2, col=1)

    return fig


symbol = st.text_input("BIST sembolü", value="THYAO.IS").strip().upper()

timeframe = st.selectbox(
    "Veri aralığı",
    [
        "Günlük — 6 ay",
        "15 dakika — 5 gün",
        "5 dakika — 5 gün",
    ],
)

settings = {
    "Günlük — 6 ay": ("6mo", "1d"),
    "15 dakika — 5 gün": ("5d", "15m"),
    "5 dakika — 5 gün": ("5d", "5m"),
}

below = st.number_input(
    "İsteğe bağlı alt alarm seviyesi",
    min_value=0.0,
    value=0.0,
    help="0 girersen alt seviye kontrolü yapılmaz.",
)
above = st.number_input(
    "İsteğe bağlı üst alarm seviyesi",
    min_value=0.0,
    value=0.0,
    help="0 girersen üst seviye kontrolü yapılmaz.",
)

if st.button("Veriyi yenile ve analiz et", type="primary"):
    if not symbol:
        st.error("Bir sembol gir. Örnek: THYAO.IS")
    else:
        period, interval = settings[timeframe]

        try:
            with st.spinner("Fiyat ve hacim verisi alınıyor..."):
                get_history.clear()
                raw_data = get_history(symbol, period, interval)

            if raw_data.empty:
                st.error(
                    "Veri alınamadı. Sembolü veya zaman aralığını kontrol et. "
                    "Yahoo Finance bu sembol için intraday veri sunmuyor olabilir."
                )
            else:
                data = add_indicators(raw_data)
                latest = data.iloc[-1]

                st.caption(f"Son veri zamanı: {data.index[-1]}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Son fiyat", number_or_dash(latest["Close"]))
                m2.metric("RSI (14)", number_or_dash(latest["RSI14"]))
                m3.metric("MACD histogramı", number_or_dash(latest["MACDHist"]))
                m4.metric("Göreli hacim", f"{number_or_dash(latest['RelativeVolume'])}×")

                st.plotly_chart(make_chart(data, symbol), use_container_width=True)

                st.subheader("Teknik gözlemler")

                observations = []

                if pd.notna(latest["EMA20"]) and pd.notna(latest["EMA50"]):
                    if latest["EMA20"] > latest["EMA50"]:
                        observations.append("EMA 20, EMA 50'nin üzerinde.")
                    else:
                        observations.append("EMA 20, EMA 50'nin altında.")

                if pd.notna(latest["RSI14"]):
                    if latest["RSI14"] >= 70:
                        observations.append("RSI 70 veya üzerinde: yüksek momentum/ aşırı alım bölgesi olabilir.")
                    elif latest["RSI14"] <= 30:
                        observations.append("RSI 30 veya altında: düşük momentum/ aşırı satım bölgesi olabilir.")
                    else:
                        observations.append("RSI 30–70 aralığında.")

                if pd.notna(latest["MACDHist"]):
                    if latest["MACDHist"] > 0:
                        observations.append("MACD histogramı sıfırın üzerinde.")
                    else:
                        observations.append("MACD histogramı sıfırın altında.")

                if pd.notna(latest["RelativeVolume"]):
                    if latest["RelativeVolume"] >= 1.5:
                        observations.append("Son hacim, 20 bar ortalamasının belirgin üzerinde.")
                    else:
                        observations.append("Son hacim, 20 bar ortalamasının 1.5 katına ulaşmadı.")

                for observation in observations:
                    st.write(f"- {observation}")

                if below > 0 and latest["Close"] <= below:
                    st.warning(f"Fiyat belirlediğin alt seviye olan {below:.2f} veya altında.")
                if above > 0 and latest["Close"] >= above:
                    st.success(f"Fiyat belirlediğin üst seviye olan {above:.2f} veya üzerinde.")

                with st.expander("Son veriler ve göstergeler"):
                    columns = [
                        "Close", "Volume", "EMA20", "EMA50", "RSI14",
                        "MACD", "MACDSignal", "MACDHist",
                        "BBUpper", "BBLower", "ATR14", "RelativeVolume",
                    ]
                    st.dataframe(data[columns].tail(20), use_container_width=True)

        except Exception as error:
            st.error(f"Analiz sırasında hata oluştu: {error}")
