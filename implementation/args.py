class rcan_args:
    def __init__(self, n_resgroups=5, n_resblocks=10, scale=[1], n_feats=64,
                 reduction=16, rgb_range=255, n_colors=3, res_scale=1):
        # n_resgroups  n_resblocks    n_feats  reduction  scale   rgb_range   n_colors   res_scale

        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.reduction = reduction
        self.scale = scale
        self.rgb_range = rgb_range
        self.n_colors = n_colors
        self.res_scale = res_scale