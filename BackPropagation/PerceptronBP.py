import random
import math as mt
class PerceptronBP:
    def __init__(self, entradas=2,epocas=2,tda=0.3,bias=1):
        self.entradas = entradas
        self.epocas = epocas
        self.tda = tda
        self.bias = bias
        self.erro = 0.0
        self.pesos = []
        self.amostras = []
        self.entrada = []
        self.saidas = []
        self.saida = 0.0
        self.saida = 0
        for i in range(entradas+1):
            self.pesos.append(random.uniform(0,1))
    
    def generateTable(self):
        self.amostras = []
        qtde = mt.pow(2, self.entradas)
        for i in range(int(qtde)):
            s = bin(i)[2:]
            amostra = []
            pos = 0
            for j in range(self.entradas):
                if(j > self.entradas - len(s)-1):
                    amostra.append(int(s[pos]))
                    pos+=1
                else:
                    amostra.append(0)
            self.amostras.append(amostra)
    
    def sig(self,soma):
        return(1/(1+pow(mt.e,-soma)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def calcSaida(self,amostra):
       # amostra.append(1)
        saida = self.somar(amostra)
        saida = self.sig(saida)
        self.saida = saida
    
    def calcErro(self, neuPos):
        #print(neuPos.erro)
        self.erro = self.sigmoid_derivative(self.saida*self.erro)
        
    
    def calcErroLast(self, saida):
        self.erro = self.sigmoid_derivative(saida-self.saida)
        
    
    def generateReturn(self, operator):
        match operator:
            case 1: #and
                for i in range(len(self.amostras)):
                    saida = self.amostras[i][0]
                    for j in range(len(self.amostras[i])):
                        saida = saida and self.amostras[i][j]                      
                    self.saidas.append(saida)
            case 2: #or
                for i in range(len(self.amostras)):
                    saida = self.amostras[i][0]
                    for j in range(len(self.amostras[i])):
                        saida = saida or self.amostras[i][j]
                    self.saidas.append(saida)
            case 3: # nor
                for i in range(len(self.amostras)):
                    saida = self.amostras[i][0]
                    for j in range(len(self.amostras[i])):
                        saida = saida or self.amostras[i][j]
                    self.saidas.append(not saida)
            case 4: #nand
                for i in range(len(self.amostras)):
                    saida = self.amostras[i][0]
                    for j in range(len(self.amostras[i])):
                        saida = saida and self.amostras[i][j]                      
                    self.saidas.append(not saida)
            case 5: #xor
                for i in range(len(self.amostras)):
                    saida = self.amostras[i][0]
                    for j in range(len(self.amostras[i])):
                        saida = saida ^ self.amostras[i][j]                      
                    self.saidas.append(saida)
                    
    def generateOR(self):
        self.amostras = []
        self.saidas = []
        qtde = mt.pow(2, self.entradas)
        for i in range(int(qtde)):
            saida = 0
            s = bin(i)[2:]
            amostra = []
            pos = 0
            for j in range(self.entradas):
                if(j > self.entradas - len(s)-1):
                    amostra.append(int(s[pos]))
                    saida = saida or int(s[pos])
                    pos+=1
                else:
                    amostra.append(0)
            self.amostras.append(amostra)
            self.saidas.append(int(saida))
        
    def generateAND(self):
        self.amostras = []
        self.saidas = []
        qtde = mt.pow(2, self.entradas)
        for i in range(int(qtde)):
            saida = 1
            s = bin(i)[2:]
            amostra = []
            pos = 0
            for j in range(self.entradas):
                if(j > self.entradas - len(s)-1):
                    amostra.append(int(s[pos]))
                    saida = saida and int(s[pos])
                    pos+=1
                else:
                    amostra.append(0)
                    saida = saida and 0
            self.amostras.append(amostra)
            self.saidas.append(int(saida))
    
    def trainBool(self, operator):
        self.generateTable()
        self.generateReturn(operator)
        self.train(self.amostras,self.saidas)
    
    def trainAND(self):
        self.generateAND()
        self.train(self.amostras, self.saidas)
        
    def trainOR(self):
        self.generateOR()
        self.train(self.amostras, self.saidas)
    
    def train(self, amostras, saidas):
        e = 0.0
        soma = 0.0
        saida = 0
        for i in range(len(amostras)):
            amostras[i].append(self.bias)
        while(e<self.epocas):
            for i in range(len(amostras)):
                soma = self.somar(amostras[i])
                saida = 1 if soma > 0 else 0            
                if(saida!=saidas[i]):
                    self.corrigir(amostras[i],saidas[i],saida)
            e+=(1/len(amostras))

    def somar(self, amostra):
        soma = 0.0
        self.entrada = amostra
        #amostra.append(1)
        #print(self.pesos)
        for i in range(len(self.pesos)):
            soma+=amostra[i]*self.pesos[i]
        return soma
    
    def corrigir(self, amostra):
        for i in range(len(self.pesos)):
            # print(i)
            # print(self.entrada)
            self.pesos[i] = self.pesos[i]+(self.tda*self.entrada[i]*(self.erro))
    
    def predict(self, amostra):
        amostra.append(self.bias)
        return 1 if self.somar(amostra)> 0 else 0
        
# y = PerceptronBP(entradas = 2, epocas = 5)
# x = [[0,0],[0,1],[1,0], [1,1]]
# z = [0,1,1,1]
# s = bin(1)[2:]
# y.trainBool(5)
# print(y.predict([0,0]))
# print(y.predict([0,1]))
# print(y.predict([1,0]))
# print(y.predict([1,1]))