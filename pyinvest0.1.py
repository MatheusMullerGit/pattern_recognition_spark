import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import time
from functools import reduce
import gc
from datetime import datetime, timedelta
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, FloatType, StringType, ArrayType


os.environ["SPARK_HOME"]="C:/Users/Mathe/AppData/Local/Programs/Python/Python38-32/Lib/site-packages/pyspark/bin/.."
#os.environ["PYSPARK_PYTHON"]="C:/Users/Mathe/Google\ Drive/Repositorios/pattern_recognition_spark/pattern_recognition_spark/venv/Scripts/python"
#os.environ["PYSPARK_DRIVER_PYTHON"]="C:/Users/Mathe/Google\ Drive/Repositorios/pattern_recognition_spark/pattern_recognition_spark/venv/Scripts/python"

os.environ["HADOOP_HOME"]="C:/spark/hadoop"
os.environ["JAVA_HOME"]="C:/Program Files/Java/jdk1.8.0_291"
os.environ["SCALA_HOME"]="C:/spark/scala/bin"
#os.environ["SPARK_HOME"]="C:/spark/spark/bin"
#os.environ["PYTHON_PATH"]="C:/Users/Mathe/Google\ Drive/Repositorios/pattern_recognition_spark/pattern_recognition_spark/venv/Scripts/python.exe"
#os.environ["PYSPARK_PYTHON"]="C:/Users/Mathe/Google\ Drive/Repositorios/pattern_recognition_spark/pattern_recognition_spark/venv/Scripts/python.exe"

# create sparksession
spark = SparkSession\
    .builder\
    .appName("pyinvest0.1")\
    .config("spark.executor.memory", "4gb")\
    .config("spark.executor.cores","4")\
    .getOrCreate()

sc = spark.sparkContext

sns.set()
print('GC: ', gc.collect())

# FOLDER = './data/'
FOLDER = './stocks/'

stocks = os.listdir(FOLDER)
# print(stocks)

def percentChange(startPoint, currentPoint):
    try:
        x = ((float(currentPoint) - float(startPoint)) / abs(startPoint)) * 100.00
        if x == 0.0:
            return 0.000000001
        else:
            return x
    except:
        return 0.0001


class Singleton(object):
    ''' Classe que instância um único objeto para toda a sessão
    '''

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(
                                      cls, *args, **kwargs)
        return cls._instance
    
    def set_dataframe(self, stock):
        self.df = spark.read\
            .option("inferSchema", "true")\
            .option("header", "true")\
            .option("dateFormat", "yyyy-MM-dd")\
            .csv(FOLDER + stock)
        self.df = self.df.withColumn('Date', F.to_date(self.df['Date']))
        self.stock = stock.rstrip('.csv')

    def spark_shape(self):
        return (self.count(), len(self.columns))
    pyspark.sql.dataframe.DataFrame.shape = spark_shape
    
    
    def plot_chart(self):
        fig = px.line(self.df.toPandas(), x="Date", y="Adj_Close", title=self.stock)
        st.plotly_chart(fig)
    

    def __len__(self):
        return self.df.shape()[0]

    def start(self, pct):
        index = int(pct*len(self))
        it = 0
        self.accuracyArray = []
        lastLen = 0
        with st.spinner('Aguarde...'):
            start_time = time.time()
            while index < len(self):
                self.df = self.df.withColumn('avgLine', self.df['Adj_Close'][0:index])
                self.df = self.df.withColumn('dates', self.df['Date'][0:index])

                patternAr, performanceAr = self.pattern_storage(self.df.select('avgLine'), index)
                dates, patForRec = self.current_pattern(self.df.select('avgLine'), self.df.select('dates'))
                patFound, plotPatAr = self.pattern_recognition(patternAr, patForRec)

                if patFound:    
                    self.plot(self.df.select('Adj_Close'), index, dates, performanceAr, plotPatAr, patternAr, patForRec)

                end_time = time.time() - start_time
                st.text(f'Processamento levou {int(end_time)} segundos para {it} amostras.')
                it += 1
                index += 1
                if self.accuracyArray:
                    tam = len(self.accuracyArray)
                    accuracyAverage = sum(self.accuracyArray)/tam
                    if lastLen < tam:
                        st.text(f'Acurácia é de {int(accuracyAverage)}% depois de {tam} predições.')
                        lastLen = tam

    def plot(self, data, index, dates, performanceAr, plotPatAr, patternAr, patForRec):
        predArray = []

        plt.figure(figsize=(10, 6))

        lines = []
        color = []

        predictedOutcomesAr = []

        data = data.toPandas()
        dates = list(dates['dates'].iloc[:].apply(lambda x: datetime.strptime(x, "%Y-%M-%d")))

        max_date = max(dates)

        for eachPatt in plotPatAr:
            futurePoints = patternAr.index(eachPatt)

            if performanceAr[futurePoints] > patForRec[29]:
                pcolor = '#24bc00'
                predArray.append(1.000)
            else:
                pcolor = '#d40000'
                predArray.append(-1.000)

            plt.plot(dates, eachPatt)
            lines.append(eachPatt)
            predictedOutcomesAr.append(performanceAr[futurePoints])
            
            color.append(pcolor)

            plt.scatter(max_date+timedelta(days=5), performanceAr[futurePoints], c=pcolor, alpha=.3)

        realOutcomeRange_list = list(data[index+20:index+30]['Adj_Close'])
        rdd3 = sc.parallelize(realOutcomeRange_list)
        realOutcomeRange_reduced = rdd3.reduce(lambda x, y: x+y)
        realAvgOutcome = realOutcomeRange_reduced/len(realOutcomeRange_list)
        
        realMovement = percentChange(data['Adj_Close'][index], realAvgOutcome)

        rdd4 = sc.parallelize(predictedOutcomesAr)
        predictedAvgOutcome_reduced = rdd4.reduce(lambda x, y: x+y)
        predictedAvgOutcome = predictedAvgOutcome_reduced/len(predictedOutcomesAr)

        rdd5 = sc.parallelize(predArray)
        predictionAverage_reduced = rdd5.reduce(lambda x, y: x+y)
        predictionAverage = predictionAverage_reduced/len(predArray)

        print(predictionAverage)
        if predictionAverage < 0:
            st.text("Previsão de baixa")
            print('drop prediction')
            print(patForRec[29])
            print(realMovement)
            if realMovement < patForRec[29]:
                self.accuracyArray.append(100)
            else:
                self.accuracyArray.append(0)

        if predictionAverage > 0:
            st.text("Previsão de alta")
            print('rise prediction')
            print(patForRec[29])
            print(realMovement)
            if realMovement > patForRec[29]:
                self.accuracyArray.append(100)
            else:
                self.accuracyArray.append(0)
    
        plt.scatter(max_date+timedelta(days=10), realMovement, c='#54fff7', s=25)
        plt.scatter(max_date+timedelta(days=10), predictedAvgOutcome, c='b', s=25)

        plt.plot(dates, patForRec, '#54fff7', linewidth=3)

        plt.grid(True)
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.23)
        plt.title(f'Reconhecimento de padrões - {self.stock}')
        plt.show()
        st.pyplot()

    def pattern_storage(self, avgLine, index):
        """
        Esta função está ok!!!
        """
        patternAr = []
        performanceAr = []

        x = index-60

        y = 31
        while y < x:
            pattern = []
            outcomeRange_list = []
            
            for i in range(29, -1, -1):
                pattern.append(percentChange(float(avgLine.collect()[y - 30]['avgLine']), float(avgLine.collect()[y - i]['avgLine'])))
      
            for j in range(20, 30):
                outcomeRange_list.append(float(avgLine.collect()[y+j]['avgLine']))

            rdd1 = sc.parallelize(outcomeRange_list)

            currentPoint = float(avgLine.collect()[y]["avgLine"])

            try:
                outcomeRange_reduced = rdd1.reduce(lambda x, y: x+y)
                avgOutcome = outcomeRange_reduced/len(outcomeRange_list)
            except Exception as e:
                print(str(e))
                avgOutcome = 0

            futureOutcome = percentChange(currentPoint, avgOutcome)    
            patternAr.append(pattern)
            performanceAr.append(futureOutcome)

            print(f'Pattern storage {y} de {x}')
            
            y += 1

        return patternAr, performanceAr
    
    def current_pattern(self, avgLine, dates):
        """
        Esta função está ok!!!
        """
        avgLine_pandas= avgLine.toPandas()
        patForRec = avgLine_pandas['avgLine'].iloc[-30:].apply(lambda x: percentChange(float(avgLine_pandas['avgLine'].iloc[-31]), float(x))).values
        dates_pandas= dates.toPandas()
        
        return dates_pandas.iloc[-30:], patForRec
        
    def pattern_recognition(self, patternAr, patForRec):
        """
        Esta função está ok!!!
        """
        plotPatAr = []
        patFound = False

        for eachPattern in patternAr[0:-5]:
            l = []
            for i in range(0,30):
                sim = 100.00 - abs(percentChange(float(eachPattern[i]), float(patForRec[i])))
                if i < 15 and sim <= 50:
                    break

                l.append(sim)
    
            howSim = sum(l)/30.0

            if howSim > 70:
                patFound = True
                plotPatAr.append(eachPattern)
           
        return patFound, plotPatAr


def main():
    # st.image('images/logo.png', width=400)
    st.title('PyInvest v0.1 - Pattern Recognition')
    #https://github.com/MatheusMullerGit

    stock = st.selectbox('Qual ação?: ', stocks)

    c = Singleton()
    c.set_dataframe(stock)    

    c.plot_chart()

    st.text(f'O tamanho do dataframe é {len(c)}')

    st.header("Efetuar reconhecimento de padrões")
    pct = st.slider('Qual a porcentagem dos dados? ', 0.0, 1.0, 0.7, 0.05)

    if st.button('Iniciar'):
        c.start(pct)
        print('GC: ', gc.collect())


if __name__ == '__main__':
    main()
