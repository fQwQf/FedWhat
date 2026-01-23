# Experiment: Compare FAFI and AURORA with FedAvg Aggregator on CIFAR-10 (alpha=0.1)

# Ensure the config file exists
# We will use the standard CIFAR10 alpha=0.1 config, assuming it is at configs/CIFAR10_alpha0.1.yaml or similar
# If not, you may need to adjust the --cfp argument.

# 1. Run FAFI (OursV7) with FedAvg
echo "Running FAFI (OursV7) + FedAvg..."
python test.py --cfp configs/CIFAR10_alpha0.1.yaml --algo FAFIFedAvg

# 2. Run AURORA (OursV14) with FedAvg
echo "Running AURORA (OursV14) + FedAvg..."
python test.py --cfp configs/CIFAR10_alpha0.1.yaml --algo AURORAFedAvg
