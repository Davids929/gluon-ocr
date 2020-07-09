 
import numpy as np
from mxnet import gluon

class CLRSAnchorGenerator(gluon.HybridBlock):
    def __init__(self, index, im_size, sizes, ratios, step, alloc_size=(128, 128),
                offsets=(0.5, 0.5), clip=False, **kwargs):
        super(CLRSAnchorGenerator, self).__init__(**kwargs)
        assert len(im_size) == 2
        self._im_size = im_size
        self._clip = clip
        self._sizes = sizes
        self._ratios = ratios
        anchors = self._generate_anchors(self._sizes, self._ratios, step, alloc_size, offsets)
        self.anchors = self.params.get_constant('anchor_%d'%(index), anchors)

    def _generate_anchors(self, sizes, ratios, step, alloc_size, offsets):
        anchors = []
        for i in range(alloc_size[0]):
            for j in range(alloc_size[1]):
                cy = (i + offsets[0]) * step
                cx = (j + offsets[1]) * step
                # ratio = ratios[0], size = size_min or sqrt(size_min * size_max)
                for s in sizes:
                    anchors.append([cx, cy, s, s])
                    # size = sizes[0], ratio = ...
                    for r in ratios[1:]:
                        sr = np.sqrt(r)
                        w = sizes[0] * sr
                        h = sizes[0] / sr
                        anchors.append([cx, cy, w, h])
        return np.array(anchors).reshape(1, 1, alloc_size[0], alloc_size[1], -1)

    @property
    def num_depth(self):
        """Number of anchors at each pixel."""
        return len(self._sizes) + len(self._ratios) - 1

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, anchors):
        a = F.slice_like(anchors, x * 0, axes=(2, 3))
        a = a.reshape((1, -1, 4))
        if self._clip:
            cx, cy, cw, ch = a.split(axis=-1, num_outputs=4)
            H, W = self._im_size
            a = F.concat(*[cx.clip(0, W), cy.clip(0, H), cw.clip(0, W), ch.clip(0, H)], dim=-1)
        return a.reshape((1, -1, 4))
