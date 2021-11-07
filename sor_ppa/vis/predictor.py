import torch
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from visualizer import SORVisualizer
import numpy as np


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):

        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        predictions = self.predictor(image)
        
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        # image = image[:, :, ::-1]
        
        visualizer = SORVisualizer(np.zeros_like(image, dtype=np.uint8), self.metadata, instance_mode=self.instance_mode)
        
        instances = predictions["instances"].to(self.cpu_device)
        vis_output = visualizer.draw_instance_sor_predictions(predictions=instances)

        return predictions, vis_output
