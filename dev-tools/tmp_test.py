import sys
from typing import cast

import torch
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.store import LocalStore, Store
from pyspark.ml import Pipeline, Model
from pyspark.ml.feature import VectorAssembler
from torch import optim

from lightautoml.spark.ml_algo.linear_pyspark import SparkCatRegression

import pandas as pd
import horovod.spark.torch as hvd

from lightautoml.spark.utils import spark_session

if __name__ == "__main__":
    model = SparkCatRegression(numeric_size=2, embed_sizes=[2, 2, 3])

    # optimizer = optim.LBFGS(
    #     model.parameters(),
    #     lr=0.1,
    #     max_iter=100,
    #     tolerance_change=1e-5,
    #     tolerance_grad=1e-5,
    #     line_search_fn="strong_wolfe",
    # )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    c = 1.0
    mse_loss = torch.nn.MSELoss(reduction='sum')

    def loss_criterion(y_pred, y_true):
        # loss = self.loss(y_true, y_pred, sample_weight=weights)
        lss = mse_loss(y_true, y_pred)

        n = y_true.shape[0]
        # if weights is not None:
        #     n = weights.sum()

        all_params = torch.cat([y.view(-1) for (x, y) in model.named_parameters() if x != "bias"])

        penalty = torch.norm(all_params, 2).pow(2) / 2 / n

        return lss + 0.5 * penalty / c

    def dummy_loss(y_pred, y_true):
        return 5.0

    # def closure():
    #     optimizer.zero_grad()
    #
    #     output = model(x)
    #     loss = loss_criterion(y, output)
    #     if loss.requires_grad:
    #         loss.backward()
    #     return loss

    N = 25
    x = torch.cat([
        torch.rand(size=(N, 2)),
        torch.randint(0, 2, size=(N, 2)),
        torch.randint(0, 3, size=(25, 1))
    ], dim=1)

    # y = torch.rand(size=(N, 1))
    y = torch.ones([25, 1], dtype=torch.float32) * 2.5

    # model.train()
    # for t in range(200):
    #     # Forward pass: Compute predicted y by passing x to the model
    #     y_pred = model(x)
    #
    #     # Compute and print loss
    #     loss = loss_criterion(y_pred, y)
    #     if t % 100 == 99:
    #         print(t, loss.item())
    #
    #     # Zero gradients, perform a backward pass, and update the weights.
    #     optimizer.zero_grad()
    #     loss.backward()
    #
    #     def closure():
    #         return loss
    #
    #     # optimizer.step(closure)
    #     optimizer.step()
    #
    # print(f'Result: {model}')

    # =============================================

    with spark_session('local[4]') as spark:
        train_data = torch.cat([x, y], dim=1)
        td_dict = {f"x_{c}": x[:, c] for c in range(x.shape[1])}
        td_dict['y'] = y[:, 0]
        train_data_df = pd.DataFrame(td_dict)

        sdf = spark.createDataFrame(train_data_df).cache()

        sdf.write.mode('overwrite').format("noop").save()

        store = Store.create('/tmp/horovod')

        assembler = VectorAssembler(
            inputCols=[c for c in sdf.columns if c.startswith("x")],
            outputCol="torch_based_linear_estimator_vassembler_features"
        )

        preproc_sdf = assembler.transform(sdf).cache()

        preproc_sdf.write.mode('overwrite').format("noop").save()

        dummy_loss = torch.nn.NLLLoss()
        dummy_model = torch.nn.Linear(in_features=5, out_features=1)
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=1e-6)

        backend = SparkBackend(num_proc=1,
                               stdout=sys.stdout, stderr=sys.stderr,
                               prefix_output_with_timestamp=True)

        torch_estimator = hvd.TorchEstimator(
            backend=backend,
            store=store,
            model=dummy_model,
            optimizer=optimizer,
            loss=dummy_loss,
            input_shapes=[[-1, len(assembler.getInputCols())]],
            feature_cols=[assembler.getOutputCol()],
            label_cols=['y'],
            epochs=2,
            verbose=1)

        trained_model = cast(Model, torch_estimator.fit(preproc_sdf))

        torch_model = trained_model.getOrDefault('model')

        # pipeline = Pipeline(stages=[assembler, torch_estimator])

        # trained_model = pipeline.fit(sdf)

        k = 0
        pass