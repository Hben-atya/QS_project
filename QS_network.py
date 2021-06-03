from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class encoder_block(nn.Module):
    def __init__(self, input_feature, output_feature, use_dropout):
        super(encoder_block, self).__init__()

        self.use_dropout = use_dropout
        self.input_feature = input_feature
        self.output_feature = output_feature

    def _encoder_layers(self, input_feature, output_feature):

        encoder_layers = []
        # block 1
        encoder_layers.append(nn.Conv3d(input_feature, output_feature, 3, 1, 1, 1))
        encoder_layers.append(nn.PReLU())
        if self.use_dropout:
            encoder_layers.append(nn.Dropout(0.2))

        # block 2
        encoder_layers.append(nn.Conv3d(output_feature, output_feature, 3, 1, 1, 1))
        encoder_layers.append(nn.PReLU())
        if self.use_dropout:
            encoder_layers.append(nn.Dropout(0.2))

        # block 3
        encoder_layers.append(nn.Conv3d(output_feature, output_feature, 3, 1, 1, 1))
        encoder_layers.append(nn.PReLU())
        if self.use_dropout:
            encoder_layers.append(nn.Dropout(0.2))

        # block 4
        encoder_layers.append(nn.Conv3d(output_feature, output_feature, 2, 2, 1, 1))
        encoder_layers.append(nn.PReLU())
        encoder_seq = nn.Sequential(encoder_layers)
        return encoder_seq

    def forward(self, x):
        input_feature = self.input_feature
        output_feature = self.output_feature

        encoder_block_1 = self._encoder_layers(input_feature, output_feature)
        out = encoder_block_1(x)

        encoder_block_2 = self._encoder_layers(output_feature, output_feature*2)
        out = encoder_block_2(out)

        return out


class decoder_block(nn.Module):
    def __init__(self, input_feature, output_feature, pooling_filter1, pooling_filter2, use_dropout):
        super(decoder_block, self).__init__()

        self.use_dropout = use_dropout
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.pooling_filter1 = pooling_filter1
        self.pooling_filter2 = pooling_filter2

    def _decoder_layers(self, input_feature, output_feature, pooling_filter):

        decoder_layers = []
        # block 1
        decoder_layers.append(nn.ConvTranspose3d(input_feature, input_feature, pooling_filter, 2, 1))
        decoder_layers.append(nn.PReLU())

        # block 2
        decoder_layers.append(nn.Conv3d(input_feature, input_feature, 3, 1, 1, 1))
        decoder_layers.append(nn.PReLU())
        if self.use_dropout:
            decoder_layers.append(nn.Dropout(0.2))

        # block 3
        decoder_layers.append(nn.Conv3d(input_feature, input_feature, 3, 1, 1, 1))
        decoder_layers.append(nn.PReLU())
        if self.use_dropout:
            decoder_layers.append(nn.Dropout(0.2))

        # block 4
        decoder_layers.append(nn.Conv3d(input_feature, output_feature, 3, 1, 1, 1))
        decoder_layers.append(nn.PReLU())
        if self.output_feature != 1 and self.use_dropout:
            decoder_layers.append(nn.Dropout(0.2))
        decoder_seq = nn.Sequential(decoder_layers)
        return decoder_seq

    def forward(self, x):

        input_feature = self.input_feature
        output_feature = self.output_feature
        pooling_filter1 = self.pooling_filter1  # 2
        pooling_filter2 = self.pooling_filter2  # 3

        decoder_block_1 = self._decoder_layers(input_feature, output_feature, pooling_filter1)
        out = decoder_block_1(x)

        decoder_block_2 = self._decoder_layers(output_feature, 1, pooling_filter2)
        out = decoder_block_2(out)
        return out


class net(nn.Module):
    def __init__(self, feature_num, use_dropout=False):
        super(net, self).__init__()
        self.encoder_m = encoder_block(input_feature=1, output_feature=feature_num, use_dropout=use_dropout)
        self.encoder_t = encoder_block(input_feature=1, output_feature=feature_num, use_dropout=use_dropout)

        self.decoder_x = decoder_block(input_feature=feature_num * 4, output_feature=feature_num * 2,
                                       pooling_filter1=2, pooling_filter2=3, use_dropout=use_dropout)
        self.decoder_y = decoder_block(input_feature=feature_num * 4, output_feature=feature_num * 2,
                                       pooling_filter1=2, pooling_filter2=3, use_dropout=use_dropout)
        self.decoder_z = decoder_block(input_feature=feature_num * 4, output_feature=feature_num * 2,
                                       pooling_filter1=2, pooling_filter2=3, use_dropout=use_dropout)

    def forward(self, x):
        [moving, target] = torch.split(x, 1, 1)

        moving_encoder_output = self.encoder_m(moving)
        target_encoder_output = self.encoder_t(target)

        combine_encoder_output = torch.cat((moving_encoder_output, target_encoder_output), 1)
        predict_result_x = self.decoder_x(combine_encoder_output)
        predict_result_y = self.decoder_y(combine_encoder_output)
        predict_result_z = self.decoder_z(combine_encoder_output)

        return torch.cat((predict_result_x, predict_result_y, predict_result_z), 1)
