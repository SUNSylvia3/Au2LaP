from pymatgen.core import Composition
import numpy as np
import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementFraction
from matminer.featurizers.base import MultipleFeaturizer
import bz2
import pickle as cPickle

# 定义特征生成器
feature_calculators = MultipleFeaturizer([
    # cf.ElementProperty.from_preset(preset_name="magpie"),
    # cf.Stoichiometry(),
    # cf.ValenceOrbital(props=['frac']),
    # cf.IonProperty(fast=True),
    # cf.BandCenter(),
    ElementFraction(),
])

def generate_single(formula, ignore_errors=False):
    """
    直接接收化学式字符串生成特征
    """
    # 转换为 DataFrame 以适配现有 featurize_dataframe 函数
    fake_df = pd.DataFrame({"formula": [formula]})
    fake_df = StrToComposition().featurize_dataframe(fake_df, "formula", ignore_errors=ignore_errors)
    fake_df = fake_df.dropna()
    fake_df = feature_calculators.featurize_dataframe(fake_df, col_id='composition', ignore_errors=ignore_errors)
    fake_df["NComp"] = fake_df["composition"].apply(len)
    return fake_df.iloc[0]  # 返回特征结果的一行

def mlmdd_single(formula):
    """
    直接接收化学式字符串生成特征
    """
    comp = Composition(formula)
    redu = comp.get_reduced_formula_and_factor()[1]
    most = comp.num_atoms
    data = np.array(list(comp.as_dict().values()))

    max_value = max(data)
    min_value = min(data)
    mean_value = np.mean(data)
    var_value = np.var(data / most)

    return [most, max_value, min_value, mean_value, redu, var_value]

def get_features_from_formula(formula):
    """
    接收化学式字符串，生成完整特征
    """
    mlmd = mlmdd_single(formula)
    ext_mag = generate_single(formula)
    mlmd_df = pd.DataFrame([mlmd], columns=["most", "max", "min", "mean", "redu", "var"])
    result = pd.concat([ext_mag.reset_index(drop=True), mlmd_df], axis=1)
    return result

def compressed_pickle(title, data):
    """
    压缩保存数据
    """
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)

def decompress_pickle(file):
    """
    解压缩加载数据
    """
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

# 示例：直接接收一个化学式生成特征
if __name__ == "__main__":
    formula = "Br2Re6Se8"
    features = get_features_from_formula(formula)
    print(features)
