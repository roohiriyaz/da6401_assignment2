Part A - Training CNN from scratch

objective:
- Build a 5-layer CNN from scratch
- Incorporate configurable hyperparameters
- Train the model
- Perform hyperparameter tuning using wandb sweeps
- Analyze performance and visualize learned filters

model architecture:
- 5 x [Conv2D → Activation → MaxPooling]
- 1 x Fully Connected Layer
- 1 x Output Layer (10 neurons for 10 classes)

files:
- train.ipynb
- sweep_train.ipynb
- README.md
