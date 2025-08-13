import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load model and encoders
model = tf.keras.models.load_model("surface_ann_model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("onehot_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

num_cols = ['HNSPDI', 'WNSPDI', 'RMEXTG', 'SLFUTI', 'LSP_Body', 'Entry_Body','FT_HEAD', 'CT_HEAD','FTGM', 'HDFBTH']
cat_cols = ['QUASTR', 'OPCCO', 'LCBXON', 'Product', 'ENDUSE', 'PASSNR']

st.title("Surface Defect Prediction (ANN)")

input_mode = st.radio("เลือกวิธีการกรอกข้อมูล", ["กรอกข้อมูลเอง", "อัพโหลดไฟล์ Excel"])

if input_mode == "กรอกข้อมูลเอง":
    st.subheader("กรอกข้อมูลแต่ละฟีเจอร์")
    input_data = {}
    # for col in num_cols:
    #     input_data[col] = st.number_input(col, value=0.0)
    # for col in cat_cols:
    #     input_data[col] = st.text_input(col, value="")
    input_data = {
                    "HNSPDI": st.number_input("HNSPDI", value=4.0),
                    "WNSPDI": st.number_input("WNSPDI", value=1219.0),
                    "RMEXTG": st.number_input("RMEXTG", value=38.0),
                    "SLFUTI": st.number_input("SLFUTI", value=3.5),
                    "LSP_Body": st.number_input("LSP_Body", value=1110.0),
                    "Entry_Body": st.number_input("Entry_Body", value=1040.0),
                    "FT_HEAD": st.number_input("FT_HEAD", value=860.0),
                    "CT_HEAD": st.number_input("CT_HEAD", value=540.0),
                    "FTGM": st.number_input("FTGM", value=9000),
                    "HDFBTH": st.number_input("HDFBTH", value=18.0),
                    "QUASTR": st.selectbox("QUASTR", options=["C032", "C032RBB", "CG145", "CS0810", "CN1410", "CR1512"]),
                    "OPCCO": st.selectbox("OPCCO", options=["0", "10", "21", "31", "41", "51", "66"]),
                    "LCBXON": st.selectbox("LCBXON", options=["USED CB", "BYPASS CB"]),
                    "Product": st.selectbox("Product", options=["ColdRoll", "CutSheet", "Other", "PO/POx", "Stock"]),
                    "ENDUSE": st.selectbox("ENDUSE", options=["PNX", "SDX", "FXX", "DGX", "ADO", "ADH", "K1I", "GXX", "RST"]
                    "PASSNR": st.selectbox("PASSNR", options=["5", "7", "9"]),
                }

    df_input = pd.DataFrame([input_data])
    predict_btn = st.button("ทำนายผล")
    if predict_btn:
        X_num = df_input[num_cols]
        X_cat = df_input[cat_cols]
        X_scaled = scaler.transform(X_num)
        X_cat_encoded = encoder.transform(X_cat).toarray()
        X_all = np.hstack((X_scaled, X_cat_encoded))
        pred = model.predict(X_all)
        pred_label = label_encoder.inverse_transform([np.argmax(pred)])
        st.success(f"ผลการทำนาย: {pred_label[0]}")
else:
    st.subheader("อัพโหลดไฟล์ Excel")
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("Preview ข้อมูลที่อัพโหลด", df.head())
        if st.button("ทำนายผลทั้งไฟล์"):
            X_num = df[num_cols]
            X_cat = df[cat_cols]
            X_scaled = scaler.transform(X_num)
            X_cat_encoded = encoder.transform(X_cat).toarray()
            X_all = np.hstack((X_scaled, X_cat_encoded))
            preds = model.predict(X_all)
            pred_labels = label_encoder.inverse_transform(np.argmax(preds, axis=1))
            df["Predicted_Defect"] = pred_labels
            st.write("ผลการทำนาย", df[["Predicted_Defect"]])
            # Download result
            result_file = "prediction_result.xlsx"
            df.to_excel(result_file, index=False)
            with open(result_file, "rb") as f:
                st.download_button(
                    "Download ผลการทำนาย", 
                    f, 
                    file_name="prediction_result.xlsx"
                    )
        else:
            st.warning("ไม่มีข้อมูลที่ครบถ้วนสำหรับการทำนาย")
    else:
        st.info("กรุณาอัพโหลดไฟล์ Excel ที่มีข้อมูลฟีเจอร์ที่ต้องการทำนาย")

