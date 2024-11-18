# %%
        
from equitrain import get_args_parser_train
from equitrain import train_fabric

# %%

def main():

    r = 4.5

    args = get_args_parser_train().parse_args()

    args.train_file      = f'output_fix/train.h5'
    args.valid_file      = f'output_fix/valid.h5'
    args.statistics_file = f'output_fix/statistics.json'
    args.output_dir      = 'output_fix/result'
    args.model           = 'mace'

    args.epochs     = 50
    args.batch_size = 64
    args.lr         = 0.01

    train_fabric(args)

# %%
if __name__ == "__main__":
    main()
