import json


def export_predictions(predictions, output_path):
    with open(output_path, "w") as handle: json.dump(predictions, handle)


def evaluate_nuscenes(result_path, output_dir=None):
    try:
        from nuscenes.eval.detection.evaluate import NuScenesEval
    except ImportError as exc:
        raise ImportError("install nuscenes-devkit for official evaluation") from exc
    from nuscenes import NuScenes
    from pathlib import Path
    output_dir = output_dir or str(Path(result_path).parent / "nuscenes_eval")
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(Path(result_path).parents[1]), verbose=False)
    evaluator = NuScenesEval(nusc, config="detection_cvpr_2019", result_path=result_path, eval_set="val", output_dir=output_dir, verbose=False)
    evaluator.main(render_curves=False)
    return evaluator.summary
