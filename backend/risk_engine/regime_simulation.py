from models.regime_model import RegimeModel


if __name__ == '__main__':
    regime_model = RegimeModel.load("main_regime_model")
    print(regime_model)