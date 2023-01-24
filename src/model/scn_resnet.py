import torch
import sparseconvnet as scn

class SCN_ResNet(torch.nn.Module):
    def __init__(
          self,
          dimension,
          outputSpatialSize: torch.LongTensor,
          device
      ):

        super(SCN_ResNet, self).__init__()

        self.model = scn.Sequential(
            scn.SubmanifoldConvolution(dimension=dimension, nIn=1, nOut=8, filter_size=3, bias=False),
            scn.MaxPooling(dimension=dimension, pool_size=3, pool_stride=2),
            scn.SparseResNet(dimension=dimension, nInputPlanes=8, layers=[
                          ['b', 8, 2, 1], # block_type, n, reps, stride
                          ['b', 16, 2, 2],
                          ['b', 24, 2, 2],
                          ['b', 32, 2, 2]]),
            scn.Convolution(dimension, 32, 64, 5, 1, False),
            scn.BatchNormReLU(64),
            scn.SparseToDense(dimension, 64)).to(device)
        self.loss_fn = torch.nn.MSELoss()
        self.inputSpatialSize = self.model.input_spatial_size(outputSpatialSize)
        self.input_layer = scn.InputLayer(dimension, self.inputSpatialSize)

        self.in_features = 64 * torch.prod(outputSpatialSize)
        self.linear = torch.nn.Linear(self.in_features, 100, bias=False)

    def forward(self, batch):
        input = self.input_layer([batch['locations'], batch['features']])
        preds = self.model(input)
        preds = preds.flatten(start_dim=1)
        preds = self.linear(preds)
        preds = preds.view(preds.shape[0], -1, 1)
        return preds

    def _get_loss(self, dl):
        cum_loss = 0
        cum_batches = 0
        for batch in dl:
            preds = self.forward(batch)
            cum_loss += self.loss_fn(
                torch.flatten(preds), torch.flatten(batch["trajectories_to_predict"])
            )
            cum_batches += 1
        mse = cum_loss / cum_batches
        return mse

    def training_step(self, batch):
        preds = self.forward(batch)
        return self.loss_fn(
            torch.flatten(preds), torch.flatten(batch["trajectories_to_predict"])
        ) 

    def validation_step(self, dlval):
        mse = self._get_loss(dlval)
        return mse, mse

    def test_step(self, dltest):
        mse = self._get_loss(dltest)
        return mse, mse