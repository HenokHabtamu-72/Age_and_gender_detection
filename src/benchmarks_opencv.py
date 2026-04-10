import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import AGE_BUCKET_LABELS, AGE_BUCKET_MIDPOINTS, PLOTS_DIR, RESULTS_DIR
from download_opencv_models import MODEL_URLS
from plots import plot_benchmark_bars, plot_bucket_confusion, plot_confusion, plot_group_metrics
from utils import (
    age_to_bucket_index,
    age_to_group,
    build_group_metrics,
    classification_metrics,
    compute_age_bucket_accuracy,
    ensure_dir,
    flatten_metrics_for_csv,
    regression_metrics,
    save_dataframe,
    save_json,
)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LABELS = [0, 1]  # 0=male, 1=female


def download_missing_models(model_dir: Path):
    import urllib.request

    ensure_dir(model_dir)
    for filename, url in MODEL_URLS.items():
        dst = model_dir / filename
        if not dst.exists():
            print(f"Downloading missing file: {filename}")
            urllib.request.urlretrieve(url, dst)


def load_nets(model_dir: Path):
    face_net = cv2.dnn.readNet(str(model_dir / "opencv_face_detector_uint8.pb"), str(model_dir / "opencv_face_detector.pbtxt"))
    age_net = cv2.dnn.readNetFromCaffe(str(model_dir / "age_deploy.prototxt"), str(model_dir / "age_net.caffemodel"))
    gender_net = cv2.dnn.readNetFromCaffe(str(model_dir / "gender_deploy.prototxt"), str(model_dir / "gender_net.caffemodel"))
    return face_net, age_net, gender_net


def detect_face(face_net, image_bgr, conf_threshold=0.7):
    h, w = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(image_bgr, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    best_box = None
    best_conf = -1.0
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < conf_threshold:
            continue
        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w - 1, int(detections[0, 0, i, 5] * w))
        y2 = min(h - 1, int(detections[0, 0, i, 6] * h))
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue
        if conf > best_conf:
            best_conf = conf
            best_box = (x1, y1, x2, y2, conf)

    if best_box is None:
        return None
    return best_box


def predict_opencv(face_bgr, age_net, gender_net):
    blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    gender_net.setInput(blob)
    gender_probs = gender_net.forward()[0]
    pred_gender = int(np.argmax(gender_probs))
    pred_gender_prob = float(gender_probs[pred_gender])

    age_net.setInput(blob)
    age_probs = age_net.forward()[0]
    pred_age_bucket = int(np.argmax(age_probs))
    pred_age = float(AGE_BUCKET_MIDPOINTS[pred_age_bucket])
    pred_age_prob = float(age_probs[pred_age_bucket])

    return pred_age, pred_age_bucket, pred_age_prob, pred_gender, pred_gender_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="models/opencv")
    parser.add_argument("--experiment_name", type=str, default="opencv_benchmark")
    parser.add_argument("--download_missing", action="store_true")
    args = parser.parse_args()

    split_df = pd.read_csv(args.split_csv)
    test_df = split_df[split_df["split"] == "test"].copy().reset_index(drop=True)
    model_dir = Path(args.model_dir)

    if args.download_missing:
        download_missing_models(model_dir)

    face_net, age_net, gender_net = load_nets(model_dir)

    preds = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running OpenCV benchmark"):
        image_path = row["image_path"]
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            continue

        det = detect_face(face_net, image_bgr)
        if det is None:
            face_bgr = image_bgr.copy()
            face_source = "full_image_fallback"
            face_conf = np.nan
        else:
            x1, y1, x2, y2, face_conf = det
            margin = 10
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image_bgr.shape[1] - 1, x2 + margin)
            y2 = min(image_bgr.shape[0] - 1, y2 + margin)
            face_bgr = image_bgr[y1:y2, x1:x2]
            if face_bgr.size == 0:
                face_bgr = image_bgr.copy()
                face_source = "full_image_fallback_empty_crop"
            else:
                face_source = "detected_face"

        pred_age, pred_age_bucket, pred_age_prob, pred_gender, pred_gender_prob = predict_opencv(face_bgr, age_net, gender_net)

        preds.append(
            {
                "image_path": image_path,
                "filename": row["filename"],
                "true_age": float(row["age"]),
                "pred_age": pred_age,
                "true_gender": int(row["gender"]),
                "pred_gender": pred_gender,
                "pred_gender_prob": pred_gender_prob,
                "true_age_bucket": age_to_bucket_index(float(row["age"])),
                "pred_age_bucket": pred_age_bucket,
                "pred_age_bucket_label": AGE_BUCKET_LABELS[pred_age_bucket],
                "pred_age_bucket_prob": pred_age_prob,
                "face_source": face_source,
                "face_confidence": face_conf,
                "age_group": age_to_group(float(row["age"])),
            }
        )

    pred_df = pd.DataFrame(preds)
    y_true_age = pred_df["true_age"].to_numpy()
    y_pred_age = pred_df["pred_age"].to_numpy()
    y_true_gender = pred_df["true_gender"].to_numpy()
    y_pred_gender = pred_df["pred_gender"].to_numpy()
    y_prob_gender = pred_df["pred_gender_prob"].to_numpy()

    metrics = {}
    metrics.update(regression_metrics(y_true_age, y_pred_age))
    metrics.update(classification_metrics(y_true_gender, y_pred_gender, y_prob_gender))
    metrics["age_bucket_accuracy"] = float((pred_df["true_age_bucket"].to_numpy() == pred_df["pred_age_bucket"].to_numpy()).mean())
    metrics["num_test_samples"] = int(len(pred_df))
    metrics["face_detection_fallback_rate"] = float((pred_df["face_source"] != "detected_face").mean())

    result_dir = RESULTS_DIR / args.experiment_name
    plot_dir = PLOTS_DIR / args.experiment_name
    ensure_dir(result_dir)
    ensure_dir(plot_dir)

    save_dataframe(pred_df, result_dir / "test_predictions.csv")
    save_json(metrics, result_dir / "metrics.json")
    flatten_metrics_for_csv(metrics, "opencv_adience_bucket_model").to_csv(result_dir / "metrics_summary.csv", index=False)

    group_df = build_group_metrics(pred_df)
    save_dataframe(group_df, result_dir / "group_metrics.csv")

    plot_confusion(y_true_gender, y_pred_gender, plot_dir / "gender_confusion_matrix.png", labels=[0, 1], title="OpenCV gender confusion matrix")
    plot_bucket_confusion(
        pd.crosstab(pred_df["true_age_bucket"], pred_df["pred_age_bucket"]).reindex(index=range(8), columns=range(8), fill_value=0).to_numpy(),
        plot_dir / "age_bucket_confusion_matrix.png",
    )
    plot_group_metrics(group_df, plot_dir / "group_age_mae.png")

    benchmark_path = RESULTS_DIR / "benchmark_comparison.csv"
    rows = []
    if benchmark_path.exists():
        rows.append(pd.read_csv(benchmark_path))
    rows.append(flatten_metrics_for_csv(metrics, "opencv_benchmark"))
    benchmark_df = pd.concat(rows, ignore_index=True)
    benchmark_df = benchmark_df.drop_duplicates(subset=["method"], keep="last")
    benchmark_df.to_csv(benchmark_path, index=False)
    plot_benchmark_bars(benchmark_df, PLOTS_DIR / "benchmark_comparison")

    print(pd.DataFrame([metrics]).T)


if __name__ == "__main__":
    main()
