import pandas as pd
from scipy.io import savemat
import os

# 1. Ler o arquivo glass.data
# O arquivo não possui cabeçalho e utiliza vírgulas como separador
df = pd.read_csv('glass.data', header=None)

# 2. Separar rótulos e características segundo a especificação da UCI
# Coluna 0: ID (descartada)
# Colunas 1 a 9: Características (Features)
# Coluna 10: Rótulo da classe (Label)
features = df.iloc[:, 1:10].values
labels = df.iloc[:, 10].values.reshape(-1, 1)

# 3. Concatenar para colocar o Rótulo como a PRIMEIRA coluna[cite: 1]
# Formato resultante: [Label, Feature_1, Feature_2, ..., Feature_9]
label_data = pd.np.hstack((labels, features)) if hasattr(pd, 'np') else __import__('numpy').hstack((labels, features))

# 4. Criar a pasta Data se não existir e salvar em Data/glass.mat
os.makedirs('Data', exist_ok=True)
caminho_saida = os.path.join('Data', 'glass.mat')

savemat(caminho_saida, {'Label_data': label_data})

print(f"Sucesso! Arquivo 'glass.mat' gerado em: {caminho_saida}")
print(f"Dimensões da matriz Label_data: {label_data.shape} (214 amostras, 1 rótulo + 9 características)")