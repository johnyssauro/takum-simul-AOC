# Simulação da Extensão Vetorial Zvtakum para RISC-V

Este repositório contém o simulador funcional da extensão vetorial **Zvtakum** (`takum_vector_sim_final.py`), desenvolvida para integrar o formato aritmético logarítmico **Takum** (T8, T16 e T32) à extensão vetorial do RISC-V (RVV 1.0).

O código realiza testes funcionais da ULA vetorial em $N=4$ pistas físicas (*hardware lanes*) e executa um benchmark comparativo de precisão contra os padrões IEEE 754 (Float32 e Float64), utilizando o padrão IEEE Float128 (34 dígitos decimais via `mpmath`) como gabarito de referência.

---

## 📋 Pré-requisitos e Bibliotecas

Para executar a simulação, é necessário ter o **Python 3.8+** instalado. O código utiliza apenas bibliotecas públicas e compatíveis com qualquer ambiente local (VS Code, PyCharm ou Terminal):

- **`numpy`**: Suporte para operações vetoriais e tipos de dados IEEE 754.
- **`matplotlib`**: Geração de gráficos de precisão e varredura de magnitudes.
- **`mpmath`**: Cálculo da referência exata em quadrupla precisão (Float128 com 34 dígitos decimais).

### Comando de Instalação

Abra o terminal e instale as dependências com o comando:

```bash
pip install numpy matplotlib mpmath
```

---

## 🚀 Como Executar o Código

Com as bibliotecas instaladas, clone ou baixe este repositório e execute o script principal pelo terminal ou pelo VS Code:
(Ou então simplesmente clique em Run no seu editor de código)

```bash
python takum_vector_sim_final.py
```

---

## 📊 O que a Execução Faz e Gera

Durante e após a execução do script:

1. **Rastreamento no Terminal**:
   - Executa 48 testes de validação aritmética e vetorial (com 100% de aprovação).
   - Exibe o rastreamento dos cálculos por ciclo de clock e por *lane* física.
   - Exibe o resumo dos erros relativos médios (%) para cargas de trabalho cosmológicas.

2. **Arquivos de Imagem Gerados**:
   Ao final, o script gera automaticamente duas figuras em alta resolução (300 DPI) na pasta do projeto:
   - **`takum_vs_ieee_precision_by_format.png`**: Gráfico comparativo de barras mostrando o erro relativo médio (%) dos formatos Takum vs. IEEE 754.
   - **`takum_vs_ieee_magnitude_sweep.png`**: Gráfico de varredura empírica ao longo de 105 ordens de magnitude ($10^{-52}$ a $10^{53}$).
