import numpy as np
from typing import List

from jdhr.engine import EVALUATORS, cfg
from jdhr.utils.console_utils import *
from jdhr.utils.base_utils import dotdict
from jdhr.utils.json_utils import serialize
from jdhr.utils.data_utils import Visualization
from jdhr.utils.metric_utils import psnr, ssim, lpips, Metrics
from jdhr.runners.visualizers.volumetric_video_visualizer import VolumetricVideoVisualizer


@EVALUATORS.register_module()
class VolumetricVideoEvaluator(VolumetricVideoVisualizer):
    def __init__(self,
                 skip_time_in_summary: int = 0,  # skip first 5 image in summary
                 result_dir: str = cfg.runner_cfg.visualizer_cfg.result_dir,  # MARK: GLOBAL
                 save_tag: str = cfg.runner_cfg.visualizer_cfg.save_tag,  # MARK: GLOBAL
                 metrics_file: str = 'metrics.json',
                 compute_metrics: List[str] = ['PSNR', 'SSIM', 'LPIPS'],
                 **kwargs,
                 ) -> None:
        super().__init__(verbose=False, result_dir=result_dir, save_tag=save_tag, **kwargs)
        self.skip_time_in_summary = skip_time_in_summary
        self.metrics = []
        self.metrics_file = metrics_file
        self.compute_metrics = [getattr(Metrics, m) for m in compute_metrics]

    def evaluate(self, output: dotdict, batch: dotdict):
        # TODO: This is a bit wasteful since the images are already generated by the visualizer
        img, img_gt, _ = super().generate_type(output, batch, Visualization.RENDER)
        metrics = dotdict()
        # Read rendering time from output
        if 'time' in output.keys() and output.time != 0:
            metrics.time = output.time

        if img_gt is not None:
            # Computing metrics
            img, img_gt = img[..., :3], img_gt[..., :3]  # image loss are compute in 3 channels (last are only for saving)
            metrics.psnr = psnr(img, img_gt)  # actual computation of the metrics
            metrics.ssim = ssim(img, img_gt)
            metrics.lpips = lpips(img, img_gt)

        if len(metrics):
            self.metrics.append(metrics)

            # For recording
            c = batch.meta.camera_index.item()
            f = batch.meta.frame_index.item()
            log(f'camera: {c}', f'frame: {f}', metrics)
            metrics.camera = c
            metrics.frame = f
        scalar_stats = dotdict({f'{k}_frame{f:04d}_cam{c:04d}': v for k, v in metrics.items()})
        return scalar_stats

    def summarize(self):
        summary = dotdict()
        if len(self.metrics):
            for key in self.metrics[0].keys():
                values = [m[key] for m in self.metrics]
                if key == 'time':
                    if np.sum(values) == 0: continue  # timer has not been enabled
                    values = values[self.skip_time_in_summary:]
                    summary[f'{key}{self.skip_time_in_summary:}+_mean'] = np.mean(values).astype(float).item()
                    summary[f'{key}{self.skip_time_in_summary:}+_std'] = np.std(values).astype(float).item()
                elif key == 'camera':
                    pass
                elif key == 'frame':
                    pass
                else:
                    summary[f'{key}_mean'] = np.mean(values).astype(float).item()
                    summary[f'{key}_std'] = np.std(values).astype(float).item()

        if len(summary):
            log(summary)

        if len(self.metrics):
            metric = dotdict()
            metric.summary = summary
            metric.metrics = self.metrics
            metric_path = join(self.result_dir, self.metrics_file)
            try:
                with open(metric_path, 'w') as f:
                    # TODO: After finding out the offending object, we can remove the try-except block and serialize call
                    json.dump(serialize(metric), f, indent=4)
                log(yellow(f'Evaluation metrics saved to {blue(metric_path)}'))
            except Exception as e:
                log(red(f'Error in dumping evaluation metrics to {blue(metric_path)}: {e}'))

            self.metrics.clear()  # clear mean after extracting summary
        return summary


@EVALUATORS.register_module()
class NoopEvaluator(VolumetricVideoVisualizer):
    def __init__(self) -> None:
        pass