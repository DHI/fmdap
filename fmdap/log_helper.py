import io
import pandas as pd


def read_DA_log(logfile):
    lines = []
    with open(logfile, "r") as fl:
        for line in fl:
            lines.append(line)
    return lines


def extract_DA_statistics(logfile):
    lines = read_DA_log(logfile)

    search_str = "Measurement statistics - DA stations (forecast)"
    df_da_f = stats_section_to_df(lines, search_str)
    search_str = "Measurement statistics - DA stations (analysis)"
    df_da_a = stats_section_to_df(lines, search_str)
    search_str = "= Measurement statistics - Validation stations ="
    df_qa_f = stats_section_to_df(lines, search_str)
    search_str = "Measurement statistics - Validation stations (analysis)"
    df_qa_a = stats_section_to_df(lines, search_str)

    return df_da_f, df_da_a, df_qa_f, df_qa_a


# ============ Measurement statistics - Validation stations ============
#   ------------------------------------------------------------------
#   Measurement       n_pts   bias       RMSE      max(err)   St_dev
#   ------------------------------------------------------------------
#   11 K13aTG         1823 -0.274E-01  0.907E-01  0.466E+00  0.425E-01
#   53 Workington      317 -0.387E-01  0.150E+00  0.495E+00  0.519E-01
#   54 Goteborg       2000  0.281E-02  0.984E-01  0.483E+00  0.508E-01
#   ------------------------------------------------------------------
#      Total         46029 -0.296E-01  0.132E+00  0.107E+01  0.470E-01
#   ------------------------------------------------------------------
# ======================================================================


def combine_f_a0(df_f, df_a, namef="f", namea="a"):
    # cols_f = ['id','name','n_points_f','bias_f','rmse_f','max_err_f','stdev_f']
    # df_f.names = cols_f
    # cols_a = ['id','name','n_points_a','bias_a','rmse_a','max_err_a','stdev_a']
    # df_a.names = cols_a
    df = df_f[["name"]].copy()
    df["n_points_" + namef] = df_f["n_points"].values
    df["n_points_" + namea] = df_a["n_points"].values
    df["bias_" + namef] = df_f["bias"].values
    df["bias_" + namea] = df_a["bias"].values
    df["rmse_" + namef] = df_f["rmse"].values
    df["rmse_" + namea] = df_a["rmse"].values
    df["max_err_" + namef] = df_f["max_err"].values
    df["max_err_" + namea] = df_a["max_err"].values
    df["stdev_" + namef] = df_f["stdev"].values
    df["stdev_" + namea] = df_a["stdev"].values
    return df


def combine_f_a(df_f, df_a, namef="f", namea="a"):
    cols_f = [
        "name",
        "n_points_" + namef,
        "bias_" + namef,
        "rmse_" + namef,
        "max_err_" + namef,
        "stdev_" + namef,
    ]
    df_f.columns = cols_f
    cols_a = [
        "name_a",
        "n_points_" + namea,
        "bias_" + namea,
        "rmse_" + namea,
        "max_err_" + namea,
        "stdev_" + namea,
    ]
    df_a.columns = cols_a
    df = pd.concat([df_f, df_a], 1).dropna()  # .mean(axis=1, level=0)
    cols = [
        "name",
        "n_points_" + namef,
        "n_points_" + namea,
        "bias_" + namef,
        "bias_" + namea,
        "rmse_" + namef,
        "rmse_" + namea,
        "max_err_" + namef,
        "max_err_" + namea,
        "stdev_" + namef,
        "stdev_" + namea,
    ]
    return df[cols]


def process_line(line):
    parts = line.split()
    mid = int(parts[0])
    name = parts[1]
    num = int(parts[2])
    bias = float(parts[3])
    rmse = float(parts[4])
    maxerr = float(parts[5])
    stdev = float(parts[6])
    return mid, name, num, bias, rmse, maxerr, stdev


def stats_section_to_df(all_lines, search_str):
    ll = get_lines_in_stats_section(all_lines, search_str)
    if ll is None:
        return None
    lines_as_str = "".join(ll)
    cols = ["id", "name", "n_points", "bias", "rmse", "max_err", "stdev"]
    df = pd.read_csv(
        io.StringIO(lines_as_str), sep="\s+", header=None, names=cols, index_col=0
    )
    return df


def get_lines_in_stats_section(all_lines, search_str):
    idx = first_line_idx(all_lines, search_str)
    if idx is None:
        return None
    idx1 = idx + 4
    idx2 = idx1 + first_line_idx(
        all_lines[idx1:], "-------------------------------------"
    )
    return all_lines[idx1:idx2]


def first_line_idx(lines, search_str):
    n = len(lines)
    for i in range(n):
        if search_str in lines[i]:
            return i
    return None
