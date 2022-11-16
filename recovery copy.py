import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import lightning as L
from torch.utils.data import TensorDataset, DataLoader


class BasicLightning(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.learning_rate = 0.01  # The value will change over the time by the Lightning and 0.01 is just a placeholder!

    def forward(self, input):
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = (
            scaled_bottom_relu_output + scaled_top_relu_output + self.final_bias
        )

        output = F.relu(input_to_final_relu)

        return output

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch  # Unpack the batch wich is (input, label)
        output_i = self.forward(input_i)  # Forward pass
        loss = (output_i - label_i) ** 2  # Compute the loss (MSE)
        return loss


input_doses = torch.linspace(start=0, end=1, steps=11)
inputs = torch.tensor([0.0, 0.5, 1.0])
labels = torch.tensor([0.0, 1.0, 0.0])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

model = BasicLightning()

trainer = L.Trainer(max_epochs=34)

lr_find_result = trainer.tuner.lr_find(
    model,
    train_dataloaders= dataloader,
    min_lr=0.001,
    max_lr=1.0,
    early_stop_threshold=None,
) # This will run the LR finder to find the best learning rate

new_lr = lr_find_result.suggestion() # This will return the best learning rate 

print(f"The best learning rate is:  {new_lr:.5f}") # Print the best learning rate :.5f means 5 digits after the decimal point

model.learning_rate = new_lr # Set the learning rate of the model to the best learning rate

# Train the model with the best learning rate
trainer.fit(model, train_dataloaders=dataloader)
print(model.final_bias.data)


output_values = model(input_doses)

sns.set(style="whitegrid")

sns.lineplot(
    x=input_doses,
    y=output_values.detach(),
    color="green",
    linewidth=2.5,
)

plt.ylabel("Efc.")
plt.xlabel("Dose")
plt.show()
