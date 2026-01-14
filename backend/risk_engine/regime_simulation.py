from models.regime_model import RegimeModel





if __name__ == '__main__':
    regime_model = RegimeModel.load("main_regime_model")
    regime_info = regime_model.regime_info
    
    cov_matrix = [r.covariance_matrix for r in regime_info.regimes]

    means = dict([(f, r.mean_return) for f, r in factor_analysis[regime_id].factor_metrics.items()])
    print(cov_matrix)

