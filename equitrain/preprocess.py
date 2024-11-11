# This file loads an xyz dataset and prepares
# new hdf5 file that is ready for training with on-the-fly dataloading

import logging
import ast
import numpy as np
import json
import random
import h5py
import torch_geometric
import os

from pathlib import Path

from equitrain.mace import tools
from equitrain.mace.data import HDF5Dataset, random_train_valid_split
from equitrain.mace.data.utils import save_configurations_as_HDF5, compute_statistics
from equitrain.mace.tools.scripts_utils import get_dataset_from_xyz, get_atomic_energies
from equitrain.mace.data.utils import load_from_xyz_in_chunks, process_atoms_list
from equitrain.mace.tools.scripts_utils import SubsetCollection

def split_array(a: np.ndarray, max_size: int):
    drop_last = False
    if len(a) % 2 == 1:
        a = np.append(a, a[-1])
        drop_last = True
    factors = get_prime_factors(len(a))
    max_factor = 1
    for i in range(1, len(factors) + 1):
        for j in range(0, len(factors) - i + 1):
            if np.prod(factors[j : j + i]) <= max_size:
                test = np.prod(factors[j : j + i])
                if test > max_factor:
                    max_factor = test
    return np.array_split(a, max_factor), drop_last


def get_prime_factors(n: int):
    factors = []
    for i in range(2, n + 1):
        while n % i == 0:
            factors.append(i)
            n = n / i
    return factors


def _preprocess(args):
    """
    This script loads an xyz dataset and prepares
    new hdf5 file that is ready for training with on-the-fly dataloading
    """

    # Setup
    tools.set_seeds(args.seed)
    random.seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    # Data preparation
    logging.info("Loading dataset in chunks")
    atomic_energies_dict = {}
    collections = SubsetCollection(train=[], valid=[], tests=[])

    if args.atomic_numbers is None:
        z_table = tools.get_atomic_number_table_from_zs(
            z
            for configs in (collections.train, collections.valid)
            for config in configs
            for z in config.atomic_numbers
        )
    else:
        logging.info("Using atomic numbers from command line argument")
        zs_list = ast.literal_eval(args.atomic_numbers)
        assert isinstance(zs_list, list)
        z_table = tools.get_atomic_number_table_from_zs(zs_list)

    # Load and process training data in chunks
    atomic_energies_dict, collections.train = load_from_xyz_in_chunks(
        file_path=args.train_file,
        config_type_weights=config_type_weights,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        extract_atomic_energies=True,
    )

    #if len(atomic_energies_dict) == 0:
    #    logging.info("Atomic energies not found in isolated atoms, using fallback calculation.")
    #    atomic_energies_dict = get_atomic_energies(args.E0s, collections.train, z_table)
    #logging.info(f"Fallback atomic energies: {atomic_energies_dict}")

    # If validation file is provided
    if args.valid_file:
        _, collections.valid = load_from_xyz_in_chunks(
            file_path=args.valid_file,
            config_type_weights=config_type_weights,
            energy_key=args.energy_key,
            forces_key=args.forces_key,
            stress_key=args.stress_key,
        )
    else:
        collections.train, collections.valid = random_train_valid_split(
            collections.train, args.valid_fraction, args.seed
        )

    # If test file is provided
    if args.test_file:
        for name, subset in collections.tests:
            _, collections.tests = load_from_xyz_in_chunks(
                file_path=args.test_file,
                config_type_weights=config_type_weights,
                energy_key=args.energy_key,
                forces_key=args.forces_key,
            )

    # Atomic number table
    # yapf: disable

    logging.info("Preparing training set")

    with h5py.File(os.path.join(args.output_dir, "train.h5"), "w") as f:
        # split collections.train into batches and save them to hdf5
        split_train, drop_last = split_array(collections.train, args.batch_size)
        f.attrs["drop_last"] = drop_last
        for i, batch in enumerate(split_train):
            save_configurations_as_HDF5(batch, i, f)
        

    if args.compute_statistics:
        # Compute statistics
        logging.info("Computing statistics")
        if len(atomic_energies_dict) == 0:
            atomic_energies_dict = get_atomic_energies(args.E0s, collections.train, z_table)
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        logging.info(f"Atomic energies array for computation: {atomic_energies.tolist()}")
        train_dataset = HDF5Dataset(os.path.join(args.output_dir, "train.h5"), z_table=z_table, r_max=args.r_max)
        train_loader = torch_geometric.loader.DataLoader(
            dataset=train_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            drop_last=False,
        )
        avg_num_neighbors, mean, std = compute_statistics(
            train_loader, atomic_energies, z_table
        )
        logging.info(f"Average number of neighbors: {avg_num_neighbors}")
        logging.info(f"Mean: {mean}")
        logging.info(f"Standard deviation: {std}")

        #sort atomic_energies_dict by atomic number (key) and convert to a single string
        atomic_energies_str = "{" + ", ".join(f"{int(k)}: {float(v)}" for k, v in sorted(atomic_energies_dict.items())) + "}"
            
        # save the statistics as a json
        statistics = {
            "atomic_energies": atomic_energies_str,
            "avg_num_neighbors": avg_num_neighbors,
            "mean": mean,
            "std": std,
            "atomic_numbers": str(z_table.zs),
            "r_max": args.r_max,
        }
        #logging.info(f"Final statistics to be saved: {statistics}")
        del train_dataset
        del train_loader
        with open(os.path.join(args.output_dir, "statistics.json"), "w") as f:
            json.dump(statistics, f)
    
    logging.info("Preparing validation set")

    with h5py.File(os.path.join(args.output_dir, "valid.h5"), "w") as f:    
        split_valid, drop_last = split_array(collections.valid, args.batch_size)
        f.attrs["drop_last"] = drop_last
        for i, batch in enumerate(split_valid):
            save_configurations_as_HDF5(batch, i, f)

    if args.test_file is not None:
        logging.info("Preparing test sets")
        for name, subset in collections.tests:
            with h5py.File(os.path.join(args.output_dir, name + "_test.h5"), "w") as f:
                split_test, drop_last = split_array(subset, args.batch_size)
                f.attrs["drop_last"] = drop_last
                for i, batch in enumerate(split_test):
                    save_configurations_as_HDF5(batch, i, f)

def preprocess(args):
    if args.train_file is None:
        raise ValueError("--train-file is a required argument")
    if args.valid_file is None:
        raise ValueError("--valid-file is a required argument")
    if args.output_dir is None:
        raise ValueError("--output-dir is a required argument")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    _preprocess(args)
