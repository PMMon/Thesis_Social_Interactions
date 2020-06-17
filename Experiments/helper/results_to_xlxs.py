import os
import pandas as pd

# ========================= Description ===========================
# Functions to write losses or other values to .xlsx-files for
# further analysis
# =================================================================

def note_best_val(filepath, args, best_val, best_val_FDE, best_val_epoch):
    """
    Writes best validation loss of model on dataset to .xlsx-file.
    :param filepath: Path to .xlsx-output-file
    :param args: Command-line arguments
    :param best_val: Best/lowest validation ADE of model on dataset
    :param best_val_FDE: Best/lowest validation FDE of model on dataset
    :param best_val_epoch: Number of epoch in which lowest ADE was measured
    :return:
    """
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        if ((df["lr"] == args.lr) & (df["wd"] == args.wd) & (df["dropout"] == args.dropout) & (df["batch_size"] == args.batch_size) & (df["enc_h_dim"] == args.encoder_h_dim) & (df["dec_h_dim"] == args.decoder_h_dim) & (df["emb_dim"] == args.emb_dim)).any():
            if df["ADE"][(df["lr"] == args.lr) & (df["wd"] == args.wd) & (df["dropout"] == args.dropout) & (
                    df["batch_size"] == args.batch_size) & (df["enc_h_dim"] == args.encoder_h_dim) & (df["dec_h_dim"] == args.decoder_h_dim) & (df["emb_dim"] == args.emb_dim)].any() > best_val:
                df.loc[(df["lr"] == args.lr) & (df["wd"] == args.wd) & (df["dropout"] == args.dropout) & (df["batch_size"] == args.batch_size) & (df["enc_h_dim"] == args.encoder_h_dim) & ( df["dec_h_dim"] == args.decoder_h_dim) & (df["emb_dim"] == args.emb_dim), "ADE"] = [best_val]
                df.loc[(df["lr"] == args.lr) & (df["wd"] == args.wd) & (df["dropout"] == args.dropout) & (df["batch_size"] == args.batch_size) & (df["enc_h_dim"] == args.encoder_h_dim) & (df["dec_h_dim"] == args.decoder_h_dim) & (df["emb_dim"] == args.emb_dim), "FDE"] = [best_val_FDE]
                df.loc[(df["lr"] == args.lr) & (df["wd"] == args.wd) & (df["dropout"] == args.dropout) & (df["batch_size"] == args.batch_size) & (df["enc_h_dim"] == args.encoder_h_dim) & (df["dec_h_dim"] == args.decoder_h_dim) & (df["emb_dim"] == args.emb_dim), "Epoch"] = [best_val_epoch]
        else:
            df_new = pd.DataFrame(
                {"ADE": best_val, "FDE": best_val_FDE, "Epoch": best_val_epoch, "lr": args.lr,
                 "wd": args.wd, "dropout": args.dropout, "batch_size": args.batch_size, "enc_h_dim": args.encoder_h_dim,
                 "dec_h_dim": args.decoder_h_dim, "emb_dim": args.emb_dim}, index=[0])
            df = df.append(df_new, ignore_index=True, sort=True)
    else:
        df = pd.DataFrame(
            {"ADE": best_val, "FDE": best_val_FDE, "Epoch": best_val_epoch, "lr": args.lr,
             "wd": args.wd, "dropout": args.dropout, "batch_size": args.batch_size, "enc_h_dim": args.encoder_h_dim,
             "dec_h_dim": args.decoder_h_dim, "emb_dim": args.emb_dim}, index=[0])

    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    writer = pd.ExcelWriter(filepath, engine="xlsxwriter")
    df.to_excel(writer)
    writer.save()


def note_test_results_for_socialforce(filepath, args, ADE_loss, FDE_loss):
    """
    Writes overall ADE and FDE of model on dataset to .xlsx-file
    :param filepath: Path to .xlsx-output-file
    :param args: Command-line arguments
    :param ADE_loss: Overall ADE of model
    :param FDE_loss: Overall FDE of model
    """

    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        if ((df["Phase"] == args.phase) & (df["V0"] == args.V0) & (df["sigma"] == args.sigma)).any():
            # always overwrite result
            df.loc[(df["Phase"] == args.phase) & (df["V0"] == args.V0) & (df["sigma"] == args.sigma), "ADE"] = [ADE_loss]
            df.loc[(df["Phase"] == args.phase) & (df["V0"] == args.V0) & (df["sigma"] == args.sigma), "FDE"] = [FDE_loss]
        else:
            df_new = pd.DataFrame({"Phase": args.phase, "V0": args.V0, "sigma": args.sigma,
                                   "ADE": ADE_loss,
                                   "FDE": FDE_loss},index=[0])
            df = df.append(df_new, ignore_index=True, sort=True)

    else:
        df = pd.DataFrame({"Phase": args.phase, "V0": args.V0, "sigma": args.sigma,
                           "ADE": ADE_loss,
                           "FDE": FDE_loss}, index=[0])

    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    writer2 = pd.ExcelWriter(filepath, engine="xlsxwriter")
    df.to_excel(writer2)
    writer2.save()


def note_nonlinear_loss(filepath, args, ADE_loss, ADE_nonlinear):
    """
    Write nonlinear ADE of model to .xlsx-file
    :param filepath: Path to .xlsx-output-file
    :param args: Command-line arguments
    :param ADE_loss: Overall ADE of model on dataset
    :param ADE_nonlinear: Nonlinear ADE of model on dataset for threshold
    """

    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        if ((df["model_type"] == args.model_type) & (df["threshold"] == args.threshold_nl) & (df["padding"] == args.padding) & (df["final_displ"] == args.final_position) & (df["V0"] == args.V0) & (df["sigma"] == args.sigma)).any():
            # always overwrite result
            df.loc[(df["model_type"] == args.model_type) & (df["threshold"] == args.threshold_nl) & (df["final_displ"] == args.final_position) & (df["padding"] == args.padding) & (df["V0"] == args.V0) & (df["sigma"] == args.sigma), "ADE_nonlinear"] = [ADE_nonlinear]
            df.loc[(df["model_type"] == args.model_type) & (df["threshold"] == args.threshold_nl) & (df["final_displ"] == args.final_position) & (df["padding"] == args.padding) & (df["V0"] == args.V0) & (df["sigma"] == args.sigma), "ADE"] = [ADE_loss]
        else:
            df_new = pd.DataFrame({"model_type": args.model_type, "threshold": args.threshold_nl, "padding": args.padding, "final_displ": args.final_position, "V0": args.V0, "sigma": args.sigma, "ADE_nonlinear": ADE_nonlinear,
                                   "ADE": ADE_loss}, index=[0])
            df = df.append(df_new, ignore_index=True, sort=True)

    else:
        df = pd.DataFrame({"model_type": args.model_type, "threshold": args.threshold_nl, "padding": args.padding, "final_displ": args.final_position, "V0": args.V0, "sigma": args.sigma, "ADE_nonlinear": ADE_nonlinear,
                                   "ADE": ADE_loss}, index=[0])

    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    writer2 = pd.ExcelWriter(filepath, engine="xlsxwriter")
    df.to_excel(writer2)
    writer2.save()


def loss_nonlinear_trajectories(path, args, loss_dict):
    file_name = "nl_traj_ADE_" + args.model_type
    if args.padding:
        file_name += "_padding"
    if args.final_position:
        file_name += "_final_dp"
    if args.lstm_pool:
        file_name += "_ns_" + str(int(args.neighborhood_size)) + "_gs_" + str(int(args.grid_size))

    file_name += "_" + args.dataset_name + ".xlsx"

    filepath = path+"//"+file_name

    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        for N, value_dict in loss_dict.items():
            for key, value in value_dict.items():
                if ((df["k"] == args.threshold_nl) & (df["N"] == N)).any():
                    df.loc[(df["k"] == args.threshold_nl) & (df["N"] == N), key] = [value]
                else:
                    df_new = pd.DataFrame({"k": args.threshold_nl, "N": N, key: value}, index=[0])
                    df = df.append(df_new, ignore_index=True, sort=True)
    else:
        df = pd.DataFrame({"k": args.threshold_nl, "N": 1, "ADE_nonlinear": loss_dict[1]["ADE_nonlinear"], "nr_traj": loss_dict[1]["nr_traj"]}, index=[0])
        for N, value_dict in loss_dict.items():
            if N == 1:
                continue
            df = df.append(pd.DataFrame({"k": args.threshold_nl, "N": N, "ADE_nonlinear": value_dict["ADE_nonlinear"], "nr_traj": value_dict["nr_traj"]}, index=[0]), ignore_index=True, sort=True)

    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    writer2 = pd.ExcelWriter(filepath, engine="xlsxwriter")
    df.to_excel(writer2)
    writer2.save()


def loss_on_traj_class(path, args, loss_dict):
    """
    Writes ADE and FDE of trajectory prediction models on classified trajectories into .xlsx-file. The following information is passed
    to the .xlsx-file:
    ADE on traj-class | FDE on traj-class | was information about destination of ped passed to model (True/False) | traj-class | gs (grid-size) | ns (neigh.-size) |
    model type| was data padded (True/False) | number of trajectories of class | total number of classified trajectories
    :param path: Path where .xlsx-file is stored
    :param args: Command-line arguments
    :param loss_dict: Dictionary with information about losses
    """
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = "loss_classified_traj_" + args.dataset_name + ".xlsx"
    filepath = os.path.join(path, file_name)

    # Write loss to file
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        for group, value_dict in loss_dict.items():
            for key, value in value_dict.items():
                if ((df["group"] == group) & (df["model_type"] == args.model_type) & (df["padding"] == args.padding) & (df["final_displ"] == args.final_position) & (df["ns"] == args.neighborhood_size) & (df["gs"] == args.grid_size)).any():
                    df.loc[(df["group"] == group) & (df["model_type"] == args.model_type) & (df["padding"] == args.padding) & (df["final_displ"] == args.final_position) & (df["ns"] == args.neighborhood_size) & (df["gs"] == args.grid_size), key] = [value]
                else:
                    df_new = pd.DataFrame({"group": group, "model_type": args.model_type, "padding": args.padding, "final_displ": args.final_position, "ns": args.neighborhood_size, "gs": args.grid_size, key: value}, index=[0])
                    df = df.append(df_new, ignore_index=True, sort=True)
    else:
        df = pd.DataFrame({"group": "strictly_linear", "model_type": args.model_type, "padding": args.padding, "final_displ": args.final_position, "ns": args.neighborhood_size, "gs": args.grid_size, "ADE_nonlinear": loss_dict["strictly_linear"]["ADE_nonlinear"], "nr_traj": loss_dict["strictly_linear"]["nr_traj"], "FDE_nonlinear": loss_dict["strictly_linear"]["FDE_nonlinear"], "total_nr_traj": loss_dict["strictly_linear"]["total_nr_traj"]}, index=[0])
        for group, value_dict in loss_dict.items():
            if group == "strictly_linear":
                continue
            df = df.append(pd.DataFrame({"group": group, "model_type": args.model_type, "padding": args.padding, "final_displ": args.final_position, "ns": args.neighborhood_size, "gs": args.grid_size, "ADE_nonlinear": value_dict["ADE_nonlinear"], "FDE_nonlinear": value_dict["FDE_nonlinear"], "nr_traj": value_dict["nr_traj"], "total_nr_traj": value_dict["total_nr_traj"]}, index=[0]), ignore_index=True, sort=True)

    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    writer2 = pd.ExcelWriter(filepath, engine="xlsxwriter")
    df.to_excel(writer2)
    writer2.save()