from detectron2.utils.visualizer import Visualizer, GenericMask, _create_text_labels, ColorMode, random_color, _SMALL_OBJECT_AREA_THRESH
import numpy as np


class SORVisualizer(Visualizer):
    def __init__(self, image, metadata, instance_mode=ColorMode.IMAGE):
        super(SORVisualizer, self).__init__(
            image, metadata, instance_mode=instance_mode)

    def draw_instance_sor_predictions(self, predictions):
        masks = np.asarray(predictions.pred_masks)
        masks = [GenericMask(x, self.output.height, self.output.width)
                 for x in masks]
        ranks = predictions.pred_ranks if len(masks) > 0 else None

        self.overlay_sor_instances(
            masks=masks,
            ranks=ranks
        )
        return self.output

    def overlay_sor_instances(
        self,
        *,
        masks=None,
        ranks=None
    ):
        
        num_instances = len(masks)
        if num_instances == 0:
            return self.output

        filtered_masks = []
        filtered_ranks = []
        for rank, mask in zip(ranks, masks):
            if rank >=0:
                filtered_masks.append(mask)
                filtered_ranks.append(rank)
        masks = filtered_masks
        ranks = filtered_ranks

        num_instances = len(masks)
        if num_instances == 0:
            return self.output

        

        masks = self._convert_masks(masks)

        assigned_colors = []
        for r in ranks:
            val = (r + 6.0) / 10.0
            assigned_colors.append(np.array([val] * 3))
            # assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            masks = [masks[idx]
                     for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

        for i in range(num_instances):
            color = assigned_colors[i]

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=1.0)
        return self.output
