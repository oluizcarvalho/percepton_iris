# Projeto 2 - Perceptron na classificação da base de dados Iris

O presente projeto compreende um perceptron de uma única camadas e de um único
neurônio capaz de classificar as espécies da base de dados Iris duas à duas. 

O programa obrigatoriamente deve receber quais as duas espécies que serão usadas para o 
treinamento do perceptron e, opcionalmente, pode receber o número de epochs, a taxa de aprendizado
ou a proporção da base que será utilizada para o treinamento.

Por padrão, o número de épocas está definido como 10, a taxa de aprendizado em 30% e a proporção para 
treinamento em 10%.

Modo de uso completo à seguir:

```bash
usage: main.py [-h] [--epocas [EPOCAS]] [--taxa [TAXA]] [--proporcao [PROPORCAO]] {setosa,versicolor,virginica} {setosa,versicolor,virginica}

Perceptron para classificação binária da base de dados Iris.

positional arguments:
  {setosa,versicolor,virginica}
                        Quais espécies de Iris (duas) devem ser usadas para treinar o Percéptron.

options:
  -h, --help            show this help message and exit
  --epocas [EPOCAS], -e [EPOCAS]
                        Número de épocas.
  --taxa [TAXA], -t [TAXA]
                        Taxa de aprendizado (eta). Deve ser inserido um valor entre 0 e 1.
  --proporcao [PROPORCAO], -p [PROPORCAO]
                        Proporção da base que deve ser usada para treinamento. Deve ser inserido um valor entre 0 e 1.
```

## Instalação
Execute `./install.sh` para instalar todas as dependências necessárias.

## Execução
Execute `./perceptron.sh`, passando os argumentos e opções necessárias.

## Saída
O programa imprime na saída padrão as informações gerais do treinamento (épocas, taxa, proporção e espécies)
seguidas pela acurácia dos testes referentes às duas espécies escolhidas.
Em seguida, testa a base da terceira espécie no modelo treinado e imprime quantos indivíduos
foram classificados em cada uma das classes.
