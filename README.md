Simulador Brasileirão 2026 (chance de permanência)
Python 3.10+.
```
pip install numpy scipy matplotlib
python carregar.py       
python analise_v2.py --n 1000000   
python graficos.py        
```
Teste rápido: `python analise_v2.py --n 50000 --nl 20000`.
Para atualizar após novas rodadas: acrescente os jogos em `jogos_2026.txt`, atualize `dados_tabela.py` e a lista RESTANTES em `simular_v2.py`.
(Opcional) `python backtest3.py` refaz a validação dos hiperparâmetros.
