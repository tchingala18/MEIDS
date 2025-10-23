# -*- coding: utf-8 -*-
# Importando bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Para a matriz de confusão bonita
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# --- 1. Leitura dos dados ---
df = pd.read_excel(r"Pauta_MT3.xlsx")

# Selecionar colunas de disciplinas e estado
disciplinas = ['121 LP', '122 MAT', '123 FIS', '126 BIO', 'EMP', 'ED.FIS', '127 QUI', '128 ING', '130 FIL', '131 GEOL']
estado_col = 'Estado'

# Remover alunos com status "Desistente"
df = df[df['Estado'] != 'Desistente']

# Converter notas para numérico
for col in disciplinas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calcular média geral por aluno
df['media_geral'] = df[disciplinas].mean(axis=1)

# Converter Estado para binário: Aprovado = 1, Reprovado = 0
df['aprovado'] = df[estado_col].apply(lambda x: 1 if x == 'Aprovado' else 0)

# Remover linhas sem média
df_clean = df.dropna(subset=['media_geral', 'aprovado']).copy()

print(f"Total de alunos analisados: {len(df_clean)}")
print(f"Média de desempenho geral: {df_clean['media_geral'].mean():.2f}")

# --- 2. Modelo Random Forest ---
X = df_clean[['media_geral']]
y = df_clean['aprovado']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

print("\n=== RESULTADOS ===")
print(f"Taxa de êxito (acurácia): {accuracy:.2%}")
print(f"Taxa de erro: {error_rate:.2%}")
print(f"Média de desempenho geral: {df_clean['media_geral'].mean():.2f}")

# --- 3. GRÁFICOS INDIVIDUAIS ---

# ====================
# GRÁFICO 1: Dispersão
# ====================
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_clean['media_geral'], df_clean['aprovado'],
                      c=df_clean['media_geral'], cmap='viridis', alpha=0.7)
plt.axhline(0.5, color='red', linestyle='--', label='Limite (0.5)')
plt.colorbar(scatter, label='Média Geral')
plt.title('Gráfico de Dispersão: Média Geral vs Resultado Final')
plt.xlabel('Média Geral')
plt.ylabel('Estado (0 = Reprovado, 1 = Aprovado)')
plt.yticks([0, 1], ['Reprovado', 'Aprovado'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ====================
# GRÁFICO 2: Histograma
# ====================
plt.figure(figsize=(10, 6))
plt.hist(df_clean['media_geral'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
media_geral = df_clean['media_geral'].mean()
plt.axvline(media_geral, color='red', linestyle='--', linewidth=2, label=f'Média Geral: {media_geral:.2f}')
plt.title('Distribuição das Médias Gerais dos Alunos')
plt.xlabel('Média Geral')
plt.ylabel('Frequência')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ====================
# GRÁFICO 3: Boxplot
# ====================
plt.figure(figsize=(10, 6))
box_data = [df_clean[df_clean['aprovado'] == 0]['media_geral'],
            df_clean[df_clean['aprovado'] == 1]['media_geral']]
plt.boxplot(box_data, labels=['Reprovado', 'Aprovado'], patch_artist=True,
            medianprops=dict(color="black"),
            boxprops=dict(facecolor="lightcoral", alpha=0.7))

plt.title('Comparação de Médias por Estado (Aprovado vs Reprovado)')
plt.ylabel('Média Geral')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# --- 4. MATRIZ DE CONFUSÃO ---
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Reprovado', 'Aprovado'],
            yticklabels=['Reprovado', 'Aprovado'])
plt.title('Matriz de Confusão - Random Forest')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

# Exibir relatório completo
print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(y_test, y_pred, target_names=['Reprovado', 'Aprovado']))

# --- Resumo final ---
print("\nResumo Estatístico por Estado:")
print(df_clean.groupby('aprovado')['media_geral'].agg(['mean', 'count']))

# Guardar o relatório e o resumo num ficheiro Excel
relatorio = classification_report(y_test, y_pred, target_names=['Reprovado', 'Aprovado'], output_dict=True)
relatorio_df = pd.DataFrame(relatorio).transpose()
resumo_df = df_clean.groupby('aprovado')['media_geral'].agg(['mean', 'count'])

with pd.ExcelWriter('relatorio_resultados.xlsx') as writer:
    relatorio_df.to_excel(writer, sheet_name='Relatorio')
    resumo_df.to_excel(writer, sheet_name='Resumo')

