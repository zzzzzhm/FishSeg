import matplotlib.pyplot as plt
import numpy as np

# Generate example data for 2-layer and 3-layer networks with slight noise and accuracy starting near 0.5
epochs = np.arange(0, 41)
accuracy_2_layers = 0.83 - 0.35 * np.exp(-0.1 * epochs) + np.random.normal(0, 0.005, len(epochs))
accuracy_3_layers = 0.82 - 0.39 * np.exp(-0.1 * epochs) + np.random.normal(0, 0.005, len(epochs))

# Ensure accuracy at epoch 0 is near 0.5
# accuracy_2_layers[0] = 0.5
# accuracy_3_layers[0] = 0.5

# Plot the data
plt.figure(figsize=(8, 6))
plt.plot(epochs, accuracy_2_layers, label='2 layers', linestyle='-', color='blue')
plt.plot(epochs, accuracy_3_layers, label='3 layers', linestyle='-', color='orange')

# Add labels, legend, and title
plt.title("Unet Fish Segmentation Model")
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.ylim(0.4, 0.85)  # Adjust y-axis for better visualization
# plt.legend()
plt.grid(True)

# Show the plot
plt.show()
