import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfp', type=str, default='./configs/SVHN_alpha0.1.yaml', help='Name of the experiment configuration path')
    parser.add_argument('--algo', type=str, default='OursV4', help='Name of the algorithm', choices=['FedAvg', 'Ensemble', 'OTFusion', 'FedProto', 'FedETF', 'OursV1', 'OursV2', 'OursV3', 'OursV4', 'OursV5', 'OursV6', 'OursV7', 'OursV8', 'OursV4IFFI', 'OursV7IFFI','OursV6IFFI','OursV5IFFI','OursV4SIMPLE','OursV7SIMPLE','OursV9', 'OursV10','OursV11','OursV12','OursV13','OursV14', 'FAFIFedAvg', 'AURORAFedAvg'])
    parser.add_argument('--lambdaval', type=float, default=0, help='Alignment loss weight')
    parser.add_argument('--annealing_strategy', type=str, default='none', help='Select the annealing strategy for the alignment loss weight (lambda).')
    parser.add_argument('--gamma_reg', type=float, default=1e-5, help='Regularization factor for the alignment loss weight sigma_sq_align in V12.')
    parser.add_argument('--lambda_max', type=float, default=50.0, help='Maximum threshold for effective lambda in stability regularization (OursV13).')

    args = parser.parse_args()

    return args

