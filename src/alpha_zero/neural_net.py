import torch
import torch.nn as nn
import torch.optim as optim

from src.boards.bitboard import ConnectGameBitboard as Game
from src.utils import Config

configuration = Config()


class ResidualBlock(nn.Module):
    """
    Residual block used in the neural network. Multiple copies of this block are stacked together to
    form the ResNet.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Forward pass of the residual block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class NeuralNetwork(nn.Module):
    """AlphaZero neural network.

    Consists of a convolutional layer, residual blocks, and two heads:
    policy (action probabilities) and value (state evaluation).
    """

    def __init__(self, game: Game):
        super().__init__()
        self.row = game.h
        self.column = game.w
        self.action_size = game.w

        self.conv1 = nn.Conv2d(1, 256, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256, 256) for _ in range(configuration.resnet_blocks)]
        )

        # Policy head
        self.conv4 = nn.Conv2d(256, 2, kernel_size=1, padding=0)
        self.bn4 = nn.BatchNorm2d(2)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc_pi = nn.Linear(self.row * self.column * 2, self.action_size)

        # Value Head
        self.conv5 = nn.Conv2d(256, 1, kernel_size=1, padding=0)
        self.bn5 = nn.BatchNorm2d(1)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc_v = nn.Sequential(
            nn.Flatten(), nn.Linear(6 * 7, 256), nn.ReLU(inplace=True), nn.Linear(256, 1), nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x: Input tensor.

        Returns:
            pi: A tensor containing the policy probabilities.
            v: A tensor containing the value of the current state.
        """
        x = x.view(-1, 1, self.row, self.column)  # Reshape input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        resnet_out = self.residual_blocks(x)

        # Policy head
        pi = self.conv4(resnet_out)
        pi = self.bn4(pi)
        pi = self.relu4(pi)
        pi = pi.view(-1, self.row * self.column * 2)
        pi = self.fc_pi(pi)
        pi = torch.nn.functional.softmax(pi, dim=1)

        # Value Head
        v = self.conv5(resnet_out)
        v = self.bn5(v)
        v = self.relu5(v)
        v = self.fc_v(v)

        return pi, v


class NNWrapper:
    """
    Wrapper for the neural network. This class is used to train the network and to make predictions.

    Attributes:
        game: An object containing the game state.
        net: a NeuralNetwork object containing the neural network to wrap
        optimizer: An optimizer object used to train the network.
    """

    def __init__(self, game):
        self.game = game
        self.net = NeuralNetwork(self.game)
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=configuration.learning_rate, momentum=configuration.momentum
        )

    def predict(self, state):
        """
        Predict the policy probabilities and value of the current state.
        Args:
            state: A numpy array containing the state canonical representation.

        Returns:
            pi: A numpy array containing the policy probabilities.
            v: A float representing the value of the current state.
        """
        state = torch.FloatTensor(state).unsqueeze(0)

        pi, v = self.net(state)
        pi, v = pi.detach().numpy()[0], v.item()

        return pi, v

    def train(self, training_data):
        """
        Train the neural network using the training data.

        Args:
            training_data: A list containing the self play states, pis and vs.
        """
        print("\nTraining the network.\n")

        for epoch in range(configuration.epochs):
            print("Epoch", epoch + 1)

            examples_num = len(training_data)

            # Divide epoch into batches.
            for i in range(0, examples_num, configuration.batch_size):
                states, pis, vs = zip(
                    *training_data[i : i + configuration.batch_size], strict=False
                )

                states = torch.FloatTensor(states)
                pis = torch.FloatTensor(pis)
                vs = torch.FloatTensor(vs)

                self.optimizer.zero_grad()

                pi_pred, v_pred = self.net(states)
                loss_pi = torch.nn.functional.cross_entropy(pi_pred, torch.argmax(pis, dim=1))
                loss_v = nn.MSELoss()(v_pred.view(-1), vs)
                total_loss = loss_pi + loss_v

                total_loss.backward()
                self.optimizer.step()

                # Record pi and v loss to a file.
                if configuration.record_loss:
                    file_path = f"{configuration.model_dir_path}/loss.txt"
                    with open(file_path, "a") as loss_file:
                        loss_file.write(f"{loss_pi.item():f}|{loss_v.item():f}\n")

        print("\n")

    def save_model(self, filename="current_model"):
        """
        Save the neural network model in self.net.

        Args:
            filename: A string representing the name of the file to save the model to.
        """
        file_path = f"{configuration.model_dir_path}/{filename}.pt"
        print("Saving model:", filename, "at", configuration.model_dir_path)
        torch.save(self.net.state_dict(), file_path)

    def load_model(self, filename="current_model"):
        """
        Load the neural network model to self.net.

        Args:
            filename: A string representing the name of the file to load the model from.
        """
        file_path = f"{configuration.model_dir_path}/{filename}.pt"
        print("Loading model:", filename, "from", configuration.model_dir_path)
        self.net.load_state_dict(torch.load(file_path))
