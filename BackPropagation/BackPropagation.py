import random
import math as mt
import PerceptronBP as pc
class BackPropagation:
    def __init__(self, camadas = [1,1], entradas=2,epocas=2,tda=0.3,bias=1):
        self.entradas = entradas
        self.epocas = epocas
        self.tda = tda
        self.bias = bias
        self.saida = 0.0
        self.pesos = []
        self.amostras = []
        self.saidas = []
        self.qtde_neuronios = sum(camadas)
        print(self.qtde_neuronios)
        self.camadas = camadas
        self.neuronios = []
        j = 1
        for i in range(len(camadas)):
            for j in range(camadas[i]):
                self.neuronios.append(pc.PerceptronBP(entradas, epocas, tda, bias))
            entradas = camadas[i]
        #self.camadas.insert(0,0)
    def mse(self,camada):
        erro = 0.0
        pos = 0
        for i in range(camada):
            pos+=self.camadas[i]
        for i in range(self.camadas[camada]):
            erro+= pow(self.neuronios[pos].erro,2)
            pos+=1
        erro/=self.camadas[camada]
        return erro
        
    def corrigir(self,amostra, saida):
        cLast = len(self.camadas)-1
        pos = self.qtde_neuronios-1
        # corrected = 0
        # for i in range(self.camadas[cLast]):
        #     self.neuronios[pos].calcErroLast(saida)
        #     self.neuronios[pos].corrigir(self.neuronios[pos].entrada)
        #     self.neuronios[pos].entrada = []
        #     pos-=1
        #     corrected+=1
        # #for i in range(cLast-1, 0, -1):
        for i in range(len(self.camadas)-1, -1, -1):
            if(i == len(self.camadas)-1):
                for j in range(self.camadas[i]):
                    self.neuronios[pos].calcErroLast(saida)
                    self.neuronios[pos].corrigir(self.neuronios[pos].entrada)                        
                    pos-=1
                pos-=self.camadas[i-1]
                for j in range(self.camadas[j-1]):
                    self.neuronios[pos].erro = self.mse(i)
                    pos+=1
            else:
                for j in range(pos, pos-self.camadas[i], -1):
                    self.neuronios[j].calcErro(self.neuronios[1])
                    self.neuronios[pos].corrigir(self.neuronios[pos].entrada) 
                if(i>0):
                    print(i)
                    pos-=self.camadas[i-1]
                    for j in range(self.camadas[i-1]):
                        self.neuronios[pos].erro = self.mse(i)
                        pos+=1       
        #mse = self.mse(erro)
        # for catual in range(len(self.camadas)-1, -1, -1):
        #     for i in range(self.camadas[catual]):
        #         if(catual == len(self.camadas)-1):
        #             for i in range(self.camadas[catual]):
        #                 self.neuronios[pos].calcErroLast(saida)
        #                 self.neuronios[pos].corrigir(self.neuronios[pos].entrada)                        
        #                 pos-=1
        #         else:
        #             for i in range(self.camadas[catual]+self.camadas[catual-1]-1, self.camadas[catual-1]-1, -1):
        #                 self.neuronios[i].calcErro(self.neuronios[1])
        #                 self.neuronios[i].corrigir(self.neuronios[i].entrada)
        #                 pos-=1
        #                 #print(i, self.neuronios[i].pesos)
        # self.neuronios[0].corrigir(self.neuronios[0].entrada)            
    
    def train(self, amostras, saidas):
        e = 0.0
        for i in amostras:
            i.append(self.bias)
        self.amostras = amostras
        self.saidas = saidas
        saida = 0.0
        while(e<self.epocas):
            for i in range(len(amostras)):
                pos = 0
                for j in range(len(self.camadas)):
                    if(j==0):
                        for k in range(self.camadas[j]):
                            self.neuronios[k].calcSaida(amostras[i])
                            pos+=1
                    else:
                        for k in range(self.camadas[j]):
                            for x in range(self.camadas[j-1]):                           
                                self.neuronios[pos].entrada.append(self.neuronios[x].saida)
                            self.neuronios[pos].entrada.append(1)
                            self.neuronios[pos].calcSaida(self.neuronios[pos].entrada)
                            pos+=1
                e+=1/len(amostras)
                saida = self.neuronios[pos-1].saida
                if(saida!=saidas[i]):
                    self.corrigir(amostras[i], saidas[i])
    
    def predict(self, amostra):
        amostra.append(self.bias)
        saida = 0.0
        pos = 0
        for j in range(len(self.camadas)):
            if(j==0):
                for k in range(self.camadas[j]):
                    self.neuronios[k].calcSaida(amostra)
                    pos+=1
            else:
                for k in range(self.camadas[j]):
                    self.neuronios[pos].entrada = []
                    for x in range(self.camadas[j-1]):                            
                        self.neuronios[pos].entrada.append(self.neuronios[x].saida)
                    self.neuronios[pos].entrada.append(1)
                    self.neuronios[pos].calcSaida(self.neuronios[pos].entrada)
                    pos+=1
        saida = self.neuronios[pos-1].saida
        return saida
        

x = BackPropagation(epocas=1000,camadas = [1,1])
x.train([[0,0],[0,1],[1,0],[1,1]],[0,1,1,0])
# for i in range(len(x.neuronios)):
#     print(x.neuronios[i].entrada)
print(x.predict([0,0]))
print(x.predict([0,1]))
print(x.predict([1,0]))
print(x.predict([1,1]))