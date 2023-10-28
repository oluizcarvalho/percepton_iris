import pandas as pd
from ucimlrepo import fetch_ucirepo
import random
import math
import argparse

class perceptron:
    _pesos: [float]
    # primeira posição é o peso do bias
    # as 4 restantes são os pesos das entradas

    def __init__(self):
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        self._pesos = [random.uniform(-1, 1) for _ in range(5)]

    def treinar_perceptron(self, dados, classes, epochs, taxa_aprendizado, especies_disponiveis):

        erro_geral = 1
        for i in range(epochs):
            if erro_geral == 0:
                break
            else:
                erro_geral = 0

            for h in range(len(dados)):
                produto_escalar = self._juncao_aditiva(dados[h],1)
                result = self._func_sigmoide(produto_escalar)

                if result <= 0.5:
                    classe_obtida = 0
                else:
                    classe_obtida = 1

                erro = self._calcular_erro(classe_obtida, 0 if "Iris-" + especies_disponiveis[0] == classes[h] else 1)
                self._atualizar_pesos(taxa_aprendizado, erro, dados[h])
                erro_geral += erro ** 2

    def _juncao_aditiva(self, input, bias):
        somatorio = sum([input[i] * self._pesos[1 + i] for i in range(len(input))])
        return somatorio + (bias * self._pesos[0])

    def _func_sigmoide(self, valor):
        return 1.0 / (1 + math.exp(-valor))

    def _calcular_erro(self, classe_obtida, classe_esperada):
        return classe_esperada - classe_obtida

    def _atualizar_pesos(self, taxa_aprendizado, erro, entradas):
        for i in range(1,len(self._pesos)):
            self._pesos[i] += taxa_aprendizado * erro * entradas[i - 1]

        self._pesos[0] += taxa_aprendizado * erro

    def testar_perceptron(self, dados, classes, especies_disponiveis):

        corretos = 0
        for i in range(len(dados)):
            produto_escalar = self._juncao_aditiva(dados[i], 1)

            result = self._func_sigmoide(produto_escalar)

            classe_obtida = "Iris-"
            if result <= 0.5:
                classe_obtida += especies_disponiveis[0]
            else:
                classe_obtida += especies_disponiveis[1]

            if classe_obtida == classes[i]:
                corretos += 1

        print(f"\nTeste padrão.\nAcurácia: {(corretos / len(dados)) * 100}%.")

    def testar_terceira_classse(self, dados, especies_disponiveis):

        classe_um = 0
        classe_dois = 0
        for i in range(len(dados)):
            produto_escalar = self._juncao_aditiva(dados[i], 1)

            result = self._func_sigmoide(produto_escalar)

            if result <= 0.5:
                classe_um += 1
            else:
                classe_dois += 1


        print(f"\nTeste terceira classe.")
        print(f"{classe_um} de {len(dados)} classificados como {especies_disponiveis[0]}.")
        print(f"{classe_dois} de {len(dados)} classificados como {especies_disponiveis[1]}.")
def normalizar_dados(setosa, versicolor, virginica, nao_treinamento):

    menor = [math.inf for _ in range(4)]
    maior = [-math.inf for _ in range(4)]

    for h in range(4):
        for i in range(50):
            if nao_treinamento != 1:
                if setosa[i][h] < menor[h]:
                    menor[h] = setosa[i][h]

                if setosa[i][h] > maior[h]:
                    maior[h] = setosa[i][h]

            if nao_treinamento != 2:
                if versicolor[i][h] < menor[h]:
                    menor[h] = versicolor[i][h]

                if versicolor[i][h] > maior[h]:
                    maior[h] = versicolor[i][h]

            if nao_treinamento != 3:
                if virginica[i][h] < menor[h]:
                    menor[h] = virginica[i][h]

                if virginica[i][h] > maior[h]:
                    maior[h] = virginica[i][h]

    for i in range(50):
        for h in range(4):
            setosa[i][h] = (setosa[i][h] - menor[h]) / (maior[h] - menor[h])
            versicolor[i][h] = (versicolor[i][h] - menor[h]) / (maior[h] - menor[h])
            virginica[i][h] = (virginica[i][h] - menor[h]) / (maior[h] - menor[h])

def obter_dados(iris, especies, proporcao, nao_treinamento):
    final_index = int(50 * proporcao)

    entradas = iris.data.features
    classes = iris.data.targets

    setosa = []
    for i in range(50):
        setosa.append([entradas["sepal length"][i], entradas["sepal width"][i], entradas["petal length"][i],
                       entradas["petal width"][i]])

    versicolor = []
    for i in range(50,100):
        versicolor.append([entradas["sepal length"][i], entradas["sepal width"][i], entradas["petal length"][i],
                          entradas["petal width"][i]])

    virginica = []
    for i in range(100, 150):
        virginica.append([entradas["sepal length"][i], entradas["sepal width"][i], entradas["petal length"][i],
                         entradas["petal width"][i]])

    class_setosa = []
    for i in range(50):
        class_setosa.append(classes['class'][i])

    class_versicolor = []
    for i in range(50,100):
        class_versicolor.append(classes['class'][i])

    class_virginica = []
    for i in range(100,150):
        class_virginica.append(classes['class'][i])

    normalizar_dados(setosa, versicolor, virginica, nao_treinamento)

    dados_entrada = []
    dados_teste = []
    dados_teste_extendidos = []
    classes_entrada = []
    classes_teste = []
    classes_teste_extendidas = []


    if "setosa" in especies:
        dados_entrada.extend(setosa[0:final_index])
        dados_teste.extend(setosa[final_index:])
        classes_entrada.extend(class_setosa[0:final_index])
        classes_teste.extend(class_setosa[final_index:])
    else:
        dados_teste_extendidos.extend(setosa)
        classes_teste_extendidas.extend(class_setosa)

    if "versicolor" in especies:
        dados_entrada.extend(versicolor[0:final_index])
        dados_teste.extend(versicolor[final_index:])
        classes_entrada.extend(class_versicolor[0:final_index])
        classes_teste.extend(class_versicolor[final_index:])
    else:
        dados_teste_extendidos.extend(versicolor)
        classes_teste_extendidas.extend( class_versicolor)

    if "virginica" in especies:
        dados_entrada.extend(virginica[0:final_index])
        dados_teste.extend(virginica[final_index:])
        classes_entrada.extend(class_virginica[0:final_index])
        classes_teste.extend(class_virginica[final_index:])
    else:
        dados_teste_extendidos.extend(virginica)
        classes_teste_extendidas.extend(class_virginica)

    return (dados_entrada, dados_teste, dados_teste_extendidos, classes_entrada, classes_teste, classes_teste_extendidas)

def creating_arg_parser():
    disponiveis = ["setosa","versicolor","virginica"]

    # a instância de ArgumentParser irá conter todas as informações da interface de linha de comando
    parser = argparse.ArgumentParser(description='Perceptron para classificação binária da base de dados Iris.')
    # add_argument adiciona argumentos que podem ser inseridos na linha de comando
    parser.add_argument('especies', choices=disponiveis, nargs=2, help="Quais espécies de Iris (duas) devem ser usadas para treinar o Percéptron.")
    parser.add_argument('--epocas', '-e', nargs='?', default=10, type=int, help="Número de épocas.")
    parser.add_argument('--taxa', '-t', nargs='?', default=0.3, type=float,
                        help="Taxa de aprendizado (eta). Deve ser inserido um valor entre 0 e 1.")
    parser.add_argument('--proporcao', '-p', nargs='?', default=0.1, type=float,
                        help="Proporção da base que deve ser usada para treinamento. Deve ser inserido um valor entre 0 e 1.")

    return parser

def main():
    iris = fetch_ucirepo(id=53)
    command_line = creating_arg_parser().parse_args()
    if command_line.especies[0] == command_line.especies[1]:
        print("Espécies diferentes devem ser informadas.")
        exit(0)

    if int(command_line.proporcao * 100) % 2 == 1:
        command_line.proporcao += 0.01

    if "setosa" not in command_line.especies:
        nao_treinamento = 1
    elif "versicolor" not in command_line.especies:
        nao_treinamento = 2
    else:
        nao_treinamento = 3

    (dados_entrada, dados_teste, dados_teste_extendidos, classes_entrada, classes_teste, classes_teste_extendidas) = obter_dados(iris, command_line.especies, command_line.proporcao, nao_treinamento)

    p_iris = perceptron()
    p_iris.treinar_perceptron(dados_entrada, classes_entrada, command_line.epocas, command_line.taxa, command_line.especies)

    print(f"Especies de treino: {command_line.especies[0]} e {command_line.especies[1]}.")
    print(f"Taxa de aprendizado: {int(command_line.taxa * 100)}%.")
    print(f"Proporção de treinamento: {int(command_line.proporcao * 100)}%.")
    print(f"Epocas: {command_line.epocas}.")
    p_iris.testar_perceptron(dados_teste,classes_teste,command_line.especies)
    p_iris.testar_terceira_classse(dados_teste_extendidos, command_line.especies)


if __name__ == "__main__":
    main()