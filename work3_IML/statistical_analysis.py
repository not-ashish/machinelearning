import pandas as pd
import pingouin as pg
import sys


def read_command_line():
    algorithm = sys.argv[1]  # ibl, kbl, fkbl
    return algorithm


def ibl_statistical_analysis():
    data_algorithm_ib = [pd.DataFrame({'accuracy': [1.0, 1.0]}),
                         pd.DataFrame({'accuracy': [0.9473, 0.9371]}), pd.DataFrame({'accuracy': [0.8557, 0.8411]})]

    df = pd.concat(data_algorithm_ib, keys=['ib1', 'ib2', 'ib3']).reset_index()
    df.reset_index(inplace=True, drop=True)
    df = df.rename(columns={'level_0': 'algorithm'})
    aov = pg.anova(dv='accuracy', between='algorithm', data=df,
                   detailed=True)
    pg.print_table(aov)

    posthocs = pg.pairwise_ttests(dv='accuracy', between='algorithm', data=df)
    pg.print_table(posthocs)


def kibl_statistical_analysis():
    # Best value of K
        # Satimage
    """data_algorithm_k_sat = [
        pd.DataFrame({'accuracy': [0.9591, 0.9946, 1.0, 0.9602, 0.9902, 1.0, 0.9071, 0.9063, 0.9055]}),
        pd.DataFrame({'accuracy': [0.9458, 0.9786, 0.9629, 0.9458, 0.9722, 0.9653, 0.9111, 0.9082, 0.9128]}),
        pd.DataFrame({'accuracy': [0.9296, 0.9683, 0.9545, 0.9341, 0.9639, 0.9546, 0.9085, 0.9077, 0.9134]})]
    df = pd.concat(data_algorithm_k_sat, keys=['k3', 'k5', 'k7']).reset_index()
    df.reset_index(inplace=True, drop=True)
    df = df.rename(columns={'level_0': 'k-value'})
        # Hypothyroid
    data_algorithm_k_hypo = [
        pd.DataFrame({'accuracy': [0.9563, 0.9708, 1.0, 0.9229, 0.9229, 0.9229 , 0.9371, 0.9363, 0.9355]}),
        pd.DataFrame({'accuracy': [0.9465, 0.9565, 0.9573, 0.9229, 0.9229, 0.9229, 0.9440, 0.9446, 0.9451]}),
        pd.DataFrame({'accuracy': [0.9401, 0.9443, 0.9494, 0.9229, 0.9229, 0.9229, 0.9602, 0.9601, 0.9612]})]
    df = pd.concat(data_algorithm_k_hypo, keys=['k3', 'k5', 'k7']).reset_index()
    df.reset_index(inplace=True, drop=True)
    df = df.rename(columns={'level_0': 'k-value'})

    # Best algorithm at each data set

    # Best similarity metric
    data_algorithm_metric = [
        pd.DataFrame({'accuracy': [0.9591, 0.9946, 1.0, 0.9458, 0.9786, 0.9629, 0.9296, 0.9683, 0.9545, 0.9563, 0.9708,
                                   1.0, 0.9465, 0.9565, 0.9573, 0.9401, 0.9443, 0.9494]}),
        pd.DataFrame({'accuracy': [0.9602, 0.9902, 1.0, 0.9458, 0.9722, 0.9653, 0.9341, 0.9639, 0.9546, 0.9229, 0.9229,
                                   0.9229, 0.9229, 0.9229, 0.9229, 0.9229, 0.9229, 0.9229]}),
        pd.DataFrame({'accuracy': [0.9071, 0.9063, 0.9055, 0.9111, 0.9082, 0.9128, 0.9085, 0.9077, 0.9134, 0.9371, 0.9363,
                                   0.9355, 0.9440, 0.9446, 0.9451, 0.9602, 0.9601, 0.9612]})]
    df = pd.concat(data_algorithm_metric, keys=['euclidean', 'canberra', 'hvdm']).reset_index()
    df.reset_index(inplace=True, drop=True)
    df = df.rename(columns={'level_0': 'metric'})"""

    # Best voting policy
    data_algorithm_policy = [
        pd.DataFrame({'accuracy': [0.9591, 0.9602, 0.9071, 0.9458, 0.9458, 0.9111, 0.9296, 0.9341, 0.9085, 0.9563, 0.9229,
                                   0.9371, 0.9465, 0.9229, .9440, 0.9401, 0.9229, 0.9602]}),
        pd.DataFrame({'accuracy': [0.9946, 0.9902, 0.9063, 0.9786, 0.9722, 0.9082, 0.9683, 0.9639, 0.9077, 0.9708, 0.9229,
                                   0.9363, 0.9565, 0.9229, 0.9446, 0.9443, 0.9229, 0.9601]}),
        pd.DataFrame({'accuracy': [1.0, 1.0, 0.9055, 0.9629, 0.9653, 0.9128, 0.9545, 0.9546, 0.9134, 1.0, 0.9229, 0.9355,
                                   0.9573, 0.9229, 0.9451, 0.9494, 0.9229, 0.9612]})]
    df = pd.concat(data_algorithm_policy, keys=['mvs', 'modpl', 'borda']).reset_index()
    df.reset_index(inplace=True, drop=True)
    df = df.rename(columns={'level_0': 'policy'})

    # Analysis
    aov = pg.anova(dv='accuracy', between='policy', data=df,
                   detailed=True)
    pg.print_table(aov)

    posthocs = pg.pairwise_ttests(dv='accuracy', between='policy', data=df)
    pg.print_table(posthocs)


def main():
    algorithm = read_command_line()
    if algorithm == 'ibl':
        ibl_statistical_analysis()
    elif algorithm == 'kibl':
        kibl_statistical_analysis()


if __name__ == "__main__":
    main()
