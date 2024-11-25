import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # Learning rate

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        # Select activation function
        if activation == 'tanh':
            self.activation_fn = self.tanh
            self.d_activation_fn = self.d_tanh
        elif activation == 'relu':
            self.activation_fn = self.relu
            self.d_activation_fn = self.d_relu
        elif activation == 'sigmoid':
            self.activation_fn = self.sigmoid
            self.d_activation_fn = self.d_sigmoid
        else:
            raise ValueError("Unsupported activation function. Choose 'tanh', 'relu', or 'sigmoid'.")

    # Activation functions and their derivatives
    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, x):
        return 1 - np.tanh(x)**2

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    # Forward pass
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1  # Input to hidden layer
        self.A1 = self.activation_fn(self.Z1)  # Apply activation
        self.Z2 = self.A1 @ self.W2 + self.b2  # Hidden to output layer
        self.A2 = np.tanh(self.Z2)  # Output activation (e.g., tanh for regression)
        return self.A2

    # Backward pass
    def backward(self, X, y):
        m = X.shape[0]

        # Output layer gradients
        dZ2 = (self.A2 - y) * (1 - np.tanh(self.Z2)**2)  # Derivative of tanh
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.d_activation_fn(self.Z1)  # Apply derivative of selected activation
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        self.grad_W1 = dW1
        self.grad_b1 = db1
        self.grad_W2 = dW2
        self.grad_b2 = db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    # Clear axes for fresh updates
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform multiple training steps
    for _ in range(10):  # Adjust this number for smoother or faster updates
        mlp.forward(X)
        mlp.backward(X, y)

    # 1. Plot Hidden Features
    hidden_features = mlp.A1  # Activations of the hidden layer
    ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
        c=y.ravel(), cmap='bwr', alpha=0.7
    )


    # 2. Hyperplane Visualization in the Hidden Space
    x = np.linspace(-1, 1, 10)
    y_grid = np.linspace(-1, 1, 10)
    X_grid, Y_grid = np.meshgrid(x, y_grid)
    Z_grid = -(mlp.W2[0, 0] * X_grid + mlp.W2[1, 0] * Y_grid + mlp.b2[0, 0]) / mlp.W2[2, 0]
    ax_hidden.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, color='blue')

    # 3. Distorted Input Space Transformed by the Hidden Layer
    transformed_X = mlp.A1  # Hidden layer activations
    ax_input.scatter(
        transformed_X[:, 0], transformed_X[:, 1],
        c=y.ravel(), cmap='bwr', alpha=0.7, edgecolor='k'
    )
    ax_input.set_title("Transformed Input Space")
    ax_input.set_xlabel("Transformed Feature 1")
    ax_input.set_ylabel("Transformed Feature 2")

    # 4. Input Layer Decision Boundary
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid)  # Pass grid through the network
    predictions = predictions.reshape(xx.shape)
    ax_input.contourf(xx, yy, predictions, levels=50, cmap='bwr', alpha=0.7)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title("Input Layer Decision Boundary")
    ax_input.set_xlabel("Input Feature 1")
    ax_input.set_ylabel("Input Feature 2")

    # 5. Visualize Features and Gradients as Circles and Edges

    input_nodes = [(0, y) for y in np.linspace(0, 1, mlp.W1.shape[0])]

    # Hidden nodes at x = 1
    hidden_nodes = [(1, y) for y in np.linspace(0, 1, mlp.W1.shape[1])]

    # Output nodes at x = 2
    output_nodes = [(2, y) for y in np.linspace(0.4, 0.6, mlp.W2.shape[1])]

    for x, y in input_nodes:
        ax_gradient.scatter(x, y, color='red', s=100, zorder=2)
    for x, y in hidden_nodes:
        ax_gradient.scatter(x, y, color='blue', s=100, zorder=2)
    for x, y in output_nodes:
        ax_gradient.scatter(x, y, color='green', s=100, zorder=2)

    # Draw edges between input and hidden layers
    for i, (x1, y1) in enumerate(input_nodes):
        for j, (x2, y2) in enumerate(hidden_nodes):
            gradient_magnitude = np.abs(mlp.W1[i, j])  # Use weight or gradient magnitude
            ax_gradient.plot(
                [x1, x2], [y1, y2],
                linewidth=gradient_magnitude * 50, color='gray', alpha=0.8, zorder=1
            )

    # Draw edges between hidden and output layers
    for i, (x1, y1) in enumerate(hidden_nodes):
        for j, (x2, y2) in enumerate(output_nodes):
            gradient_magnitude = np.abs(mlp.W2[i, j])  # Use weight or gradient magnitude
            ax_gradient.plot(
                [x1, x2], [y1, y2],
                linewidth=gradient_magnitude * 5, color='gray', alpha=0.8, zorder=1
            )


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)