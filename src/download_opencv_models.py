import argparse
import urllib.request
from pathlib import Path

from utils import ensure_dir

MODEL_URLS = {
    "opencv_face_detector.pbtxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb",
    "age_deploy.prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt",
    "age_net.caffemodel": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_net.caffemodel",
    "gender_deploy.prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt",
    "gender_net.caffemodel": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_net.caffemodel",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/opencv")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    ensure_dir(model_dir)

    for filename, url in MODEL_URLS.items():
        destination = model_dir / filename
        if destination.exists():
            print(f"Exists, skipping: {destination}")
            continue
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, destination)
        print(f"Saved to {destination}")


if __name__ == "__main__":
    main()
