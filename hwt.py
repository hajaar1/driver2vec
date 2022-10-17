# Haar Wavelet Transorm

# The following two blocks implement the second part of the Driver2Vec architecture, related to the Haar wavelet transorm. Its aim is to capture spectral components of the inputs.


def reference_transform(tensor):
    """
    Apply the Haar wavelet transform to a tensor

    input:
        tensor: a tensor with dimension (N, C, L), with N batch size, C number of channels and L the input length

    output:
        a tensor with dimensions (N, C, L) where the the two output channels of the transform are concatenated along the L dimension
    """
    array = tensor.numpy()
    out1, out2 = pywt.dwt(array, "haar")
    out1 = torch.from_numpy(out1)
    out2 = torch.from_numpy(out2)

    # concatenate each channel to be able to concatenate it to the untransformed data
    # everything will then be split when fed to the network
    return torch.cat((out1, out2), -1)


class WaveletPart(nn.Module):
    """
    Module to map the (N, C, L) output of the Haar transform to a (N, 2*O) tensor
    """

    def __init__(self, input_length, input_size, output_size):
        """
        inputs:
            input_length: length of the initial sequence fed to the network
            input_size: size of the inputs of the FC layer 
            output_size: output size of the FC layer
        """
        super(WaveletPart, self).__init__()

        # used two different layers here as in the paper but in the github code, they are the same
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)

        self.input_size = input_size
        self.input_length = input_length

        self.haar = reference_transform

    def init_weight(self):
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.normal_(0, 0.01)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.normal_(0, 0.01)

    def forward(self, x):
        # split the wavelet transformed data along third dim
        # the data should have  dimension (N,C,L*2) where N is the batch size,
        # C is the number of channels and L the in input_length (*2 because of wavelet transforma concatenation)
        x1, x2 = torch.split(x, self.input_length//2, 2)

        # reshape everything to feed to the linear layer
        bsize = x.size()[0]
        x1 = self.fc1(x1.reshape((bsize, -1, 1)).squeeze())
        x2 = self.fc2(x2.reshape((bsize, -1, 1)).squeeze())
        x1 = x1.reshape(bsize, -1)
        x2 = x2.reshape(bsize, -1)
        return torch.cat((x1, x2), -1)
