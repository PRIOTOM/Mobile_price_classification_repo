import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open("Mobile_Price_Classification.pkl", "rb") as f:
    model = pickle.load(f)


def predict_range(battery_power, blue, clock_speed, dual_sim, fc, four_g,
                  int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
                  px_width, ram, sc_h, sc_w, talk_time, three_g,
                  touch_screen, wifi):
    input_df = pd.DataFrame([[
        battery_power, blue, clock_speed, dual_sim, fc, four_g,
        int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
        px_width, ram, sc_h, sc_w, talk_time, three_g,
        touch_screen, wifi
    ]],
        columns=[
        'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
        'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
        'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
        'touch_screen', 'wifi'
    ])
    prediction = model.predict(input_df)[0]
    return f"Predicted Price Range: {np.clip(prediction, 0, 3):.0f}"


inputs = [
    gr.Number(label="Battery Power"),
    gr.Radio(["Yes", "No"], label="Blue"),
    gr.Number(label="Clock Speed"),
    gr.Radio(["Yes", "No"], label="Dual Sim"),
    gr.Number(label="Front Camera"),
    gr.Radio(["Yes", "No"], label="Four G"),
    gr.Number(label="Internal Memory"),
    gr.Number(label="Mobile Depth"),
    gr.Number(label="Mobile Weight"),
    gr.Number(label="Number of Cores"),
    gr.Number(label="Primary Camera"),
    gr.Number(label="Pixel Height"),
    gr.Number(label="Pixel Width"),
    gr.Number(label="RAM"),
    gr.Number(label="Screen Height"),
    gr.Number(label="Screen Width"),
    gr.Number(label="Talk Time"),
    gr.Radio(["Yes", "No"], label="Three G"),
    gr.Radio(["Yes", "No"], label="Touch Screen"),
    gr.Radio(["Yes", "No"], label="WiFi")
]

app = gr.Interface(
    fn=predict_range,
    inputs=inputs,
    outputs="text",
    title="Mobile Price Range"
)

app.launch(share=True)
