# 📊 Análise de Desempenho Acadêmico com Ciência de Dados

Projeto desenvolvido na cadeira de **Ciências de Dados**, voltado à análise preditiva do desempenho escolar dos alunos a partir de suas médias gerais. O objetivo é prever se um aluno será **aprovado** ou **reprovado** utilizando técnicas de aprendizado de máquina.

## 🎯 Objetivo
Prever o resultado final (Aprovado/Reprovado) de alunos com base na **média geral** das disciplinas, aplicando modelos de classificação e visualizações para interpretação dos dados.

## 🧰 Tecnologias Utilizadas
- **Python** como linguagem principal
- `pandas` – manipulação de dados
- `numpy` – operações numéricas
- `scikit-learn` – modelo de Random Forest e métricas
- `matplotlib` e `seaborn` – visualização de dados
- `Jupyter Notebook` (ou script Python)

## 📁 Estrutura do Projeto
├── main.py # Script principal de análise

├── PautaMT3.xlsx # Dados brutos (não versionado em produção)

├── README.md # Este arquivo

└── figs/ # Pasta opcional para salvar gráficos (recomendado)

> ⚠️ **Nota**: O arquivo `PautaMT3.xlsx` contém dados sensíveis (nomes, notas). Em ambientes reais, evite compartilhar dados pessoais.

## 🔍 Metodologia
1. **Leitura e Limpeza de Dados**
   - Leitura da planilha Excel
   - Remoção de alunos com status "Desistente"
   - Conversão de notas para formato numérico
   - Cálculo da média geral por aluno

2. **Preparação da Variável Alvo**
   - Codificação binária: `Aprovado = 1`, `Reprovado = 0`

3. **Modelo de Machine Learning**
   - Classificador: **Random Forest**
   - Divisão dos dados: 80% treino, 20% teste
   - Avaliação com acurácia, matriz de confusão e relatório de classificação

4. **Visualizações**
   - Gráfico de dispersão: Média vs Resultado
   - Histograma: Distribuição das médias
   - Boxplot: Comparação entre aprovados e reprovados
   - Matriz de confusão estilizada

## 📈 Resultados Principais
- **Taxa de acerto (acurácia)**: ~90%+ (valor exato depende da execução)
- A média geral é um forte indicador do resultado final
- Visualizações confirmam padrões claros entre desempenho e aprovação

## 🖼️ Exemplos de Gráficos

| Dispersão | Histograma |
|---------|----------|
| <img width="671" height="425" alt="Figure 2025-10-22 145723 (0)" src="https://github.com/user-attachments/assets/43a4e593-8cbd-4a9b-9044-9935197e0fd7" />|<img width="713" height="425" alt="Figure 2025-10-22 145723 (1)" src="https://github.com/user-attachments/assets/826778ab-157b-47c8-86d1-6f487c582a09" />|

| Boxplot | Matriz de Confusão |
|--------|-------------------|
| <img width="713" height="425" alt="Figure 2025-10-22 145723 (2)" src="https://github.com/user-attachments/assets/6ecf1bc2-17dd-43be-a463-2d19ab487c1a" />| <img width="528" height="425" alt="Figure 2025-10-22 145723 (3)" src="https://github.com/user-attachments/assets/48f61f88-9901-42f3-a81f-4c05b75c6e94" />|

> *(Imagens ilustrativas — geradas ao executar o código)*

---

## ▶️ Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/tchingala18/MEIDS.git
   
