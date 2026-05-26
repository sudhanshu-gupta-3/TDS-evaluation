import pandas as pd


def render_detection_table(detections: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(detections)
    if "confidence" in df.columns:
        df = df.sort_values("confidence", ascending=False)
    return df.reset_index(drop=True)
