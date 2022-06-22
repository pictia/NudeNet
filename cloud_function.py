# Pictia (https://www.pictia.io) Copyright (c) 2020 Pictia SAS

from nudenet import NudeClassifier, NudeDetector
from google.cloud import storage
import os

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

def download_models():
    files = ["detector_v2_default_checkpoint.onnx", "classifier_model.onnx"]
    for file in files:
        if not os.path.exists(f"/tmp/{file}"):
            download_blob('pictia-dev', f'nsfw-models/{file}', f'/tmp/{file}')

def process_analyse(paths, images):
    classifier = NudeClassifier(model_path="/tmp/classifier_model.onnx")
    detector = NudeDetector(model_path="/tmp/detector_v2_default_checkpoint.onnx")
    results = {}

    classifier_predictions = classifier.classify(images, paths)

    detector_predictions = {paths[i]: detector.detect(images[i]) for i in range(len(images))}
    for path, pred in classifier_predictions.items():
        results[path] = {"sfw": True if pred["sfw"] >= 0.5 else False, "probs": pred, "detector": detector_predictions[path]}

    return results

def check_nsfw(request):
    if request_json := request.get_json():
        try:
            images = request_json["images"]
            paths = request_json["paths"]
            return {"results": process_analyse(paths, images)}
        except Exception as e:
            return {"error": str(e)}
    return {"error": []}
