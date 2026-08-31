import json


def export_predictions(predictions, output_path):
    with open(output_path, "w") as handle: json.dump(predictions, handle)


def evaluate_nuscenes(result_path, output_dir=None):
    try:
        from nuscenes.eval.detection.evaluate import NuScenesEval
    except ImportError as exc:
        raise ImportError("install nuscenes-devkit for official evaluation") from exc
    return NuScenesEval
