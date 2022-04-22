import argparse


def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # Mode and Exp. Settings.
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR100",
        help="[CIFAR10, CIFAR100]",
    )

    parser.add_argument(
        "--dir",
        type=str,
        default="./"
    )
    
    parser.add_argument(
        "--memory_size", type=int, default=2000, help="Episodic memory size"
    )

    

    # Model
    parser.add_argument(
        "--model_name", type=str, default="resnet18", help="[resnet18, resnet32]"  
    )

    # Train
    parser.add_argument("--opt_name", type=str, default="sgd", help="[adam, sgd]")
    parser.add_argument("--sched_name", type=str, default="cos", help="[cos, anneal]")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--n_epoch", type=int, default=250, help="Epoch")

    parser.add_argument("--n_worker", type=int, default=0, help="The number of workers")

    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--initial_annealing_period",
        type=int,
        default=20,
        help="Initial Period that does not anneal",
    )
    parser.add_argument(
        "--annealing_period",
        type=int,
        default=20,
        help="Period (Epochs) of annealing lr",
    )
    parser.add_argument(
        "--learning_anneal", type=float, default=10, help="Divisor for annealing"
    )

    # my arguments
    parser.add_argument('--epoch', default=250, type=int)
    parser.add_argument('--max_size', default=2000, type=int)
    parser.add_argument('--total_cls', default=100, type=int)

    parser.add_argument('--batch_num', default=5, type=int)
    parser.add_argument('--model_base_path', type=str, default="logs/CIFAR100/modelv2")


    args = parser.parse_args()
    return args
