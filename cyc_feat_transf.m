function [feat_sin, feat_cos] = cyc_feat_transf(data, period)
    feat_sin = sin(2*pi*data/period);
    feat_cos = cos(2*pi*data/period);

end