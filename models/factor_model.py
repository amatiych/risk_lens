
from dataclasses import dataclass
from pandas import DataFrame, read_csv
@dataclass
class FactorModel:
    model_name: str

    factors: DataFrame

    @classmethod
    def load(cls,model_name) -> 'FactorModel':
        factors = read_csv(f'data/factor_models/{model_name}.csv').set_index('Date')
        return FactorModel(model_name=model_name,factors=factors)

if __name__ == '__main__':
    factor_model = FactorModel.load('fama_french')
    print(factor_model.factors)