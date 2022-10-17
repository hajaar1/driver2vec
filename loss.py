# Full architecture

# The following class implements the full architecture of Driver2Vec


class Driver2Vec(nn.Module):
    def __init__(
            self,
            input_size,
            input_length,
            num_channels,
            output_size,
            kernel_size,
            dropout,
            do_wavelet=True,
            fc_output_size=15):
        super(Driver2Vec, self).__init__()

        self.tcn = TemporalConvNet(input_size,
                                   num_channels,
                                   kernel_size=kernel_size,
                                   dropout=dropout)
        self.wavelet = do_wavelet
        if self.wavelet:
            self.haar = WaveletPart(
                input_length, input_size*input_length//2, fc_output_size)

            linear_size = num_channels[-1] + fc_output_size*2
        else:
            linear_size = num_channels[-1]
        self.input_length = input_length

        self.input_bn = nn.BatchNorm1d(linear_size)
        self.linear = nn.Linear(linear_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, inputs, print_temp=False):
        """Inputs have to have dimension (N, C_in, L_in*2)
        the base time series, and the two wavelet transform channel are concatenated along the third dim"""

        # split the inputs, in the last dim, first is the unchanged data, then
        # the wavelet transformed data
        input_tcn, input_haar = torch.split(inputs, self.input_length, 2)

        # feed each one to their corresponding network
        y1 = self.tcn(input_tcn)
        # for the TCN, only the last output element interests us
        y1 = y1[:, :, -1]

        if self.wavelet:
            y2 = self.haar(input_haar)

            out = torch.cat((y1, y2), 1)
        else:
            out = y1
        # bsize = out.shape[0]

        # if bsize > 1:  # issue when the batch size is 1, can't batch normalize it
        #     out = self.input_bn(out)
        # else:
        #     out = out
        # out = self.linear(out)
        out = self.activation(out)

        # if print_temp:
        #     print(out)

        return out
