import requests
import streamlit as st

st.title("🍷 葡萄酒分类预测系统 (AUTO MATES)")

st.sidebar.header("输入葡萄酒化学指标")
alcohol = st.sidebar.slider("酒精浓度 (Alcohol)", 0.0, 20.0, 13.0)
malic_acid = st.sidebar.slider("苹果酸 (Malic Acid)", 0.0, 10.0, 2.0)
ash = st.sidebar.slider("灰分 (Ash)", 0.0, 5.0, 2.3)
alcalinity_of_ash = st.sidebar.slider("灰分碱度", 0.0, 30.0, 15.0)
magnesium = st.sidebar.slider("镁 (Magnesium)", 0.0, 200.0, 100.0)
total_phenols = st.sidebar.slider("总酚", 0.0, 5.0, 2.5)
flavanoids = st.sidebar.slider("黄酮类", 0.0, 5.0, 2.0)
nonflavanoid_phenols = st.sidebar.slider("非黄酮酚", 0.0, 1.0, 0.3)
proanthocyanins = st.sidebar.slider("原花青素", 0.0, 5.0, 1.5)
color_intensity = st.sidebar.slider("色彩强度", 0.0, 15.0, 5.0)
hue = st.sidebar.slider("色调", 0.0, 2.0, 1.0)
od280_od315_of_diluted_wines = st.sidebar.slider("OD280/OD315", 0.0, 5.0, 2.5)
proline = st.sidebar.slider("脯氨酸", 0.0, 2000.0, 750.0)

if st.button("开始预测"):
    payload = {
        "alcohol": alcohol, "malic_acid": malic_acid, "ash": ash,
        "alcalinity_of_ash": alcalinity_of_ash, "magnesium": magnesium,
        "total_phenols": total_phenols, "flavanoids": flavanoids,
        "nonflavanoid_phenols": nonflavanoid_phenols, "proanthocyanins": proanthocyanins,
        "color_intensity": color_intensity, "hue": hue,
        "od280_od315_of_diluted_wines": od280_od315_of_diluted_wines, "proline": proline
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            pred = response.json().get("prediction")
            st.success(f"🎉 预测成功！该葡萄酒属于类别：{pred}")
        else:
            st.error(f"⚠️ 预测失败：{response.text}")
    except requests.exceptions.RequestException as e:
     st.error(f"✖ 无法连接到后端 API: {e}")
