import streamlit as st
import pandas as pd 
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

st.title('Aplikasi Pendeteksi Konsumsi CO2')

with st.expander('Dataset'):
    data = pd.read_csv('FuelConsumptionCo2.csv')
    st.write(data)

    st.success('Informasi Dataset')
    data1 = pd.DataFrame(data)
    buffer = io.StringIO()
    data1.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    st.success('Analisa Univariat')
    deskriptif = data.describe()
    st.write(deskriptif)

with st.expander('Visualisasi'):
    st.info('Visualisasi Per Column')
    
    fig,ax = plt.subplots()
    sns.histplot(data['ENGINESIZE'],color='blue')
    plt.xlabel('Ratio EngineSize')
    st.pyplot(fig)
    
    fig,ax = plt.subplots()
    sns.histplot(data['MAKE'],color='red')
    plt.xlabel('Jenis Pabrikan')
    plt.xticks(rotation=90)
    st.pyplot(fig)
    
    st.info('Korelasi Heatmap')
    fitur_angka = ['MODELYEAR',
                   'CYLINDERS',
                   'ENGINESIZE',
                   'FUELCONSUMPTION_CITY',
                   'FUELCONSUMPTION_HWY',
                   'FUELCONSUMPTION_COMB',
                   'FUELCONSUMPTION_COMB_MPG',
                   'CO2EMISSIONS']
    matriks_korelasi = data[fitur_angka].corr()
    
    fig,ax = plt.subplots()
    sns.heatmap(matriks_korelasi,annot=True, cmap='RdBu')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('Korelasi Antar Fitur Angka',fontsize=10)
    st.pyplot(fig)
    
    def plot_outlier(data,column):
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        sns.boxplot(data[column])
        plt.title(f'{column} - Box Plot')
        
        plt.subplot(1,2,2)
        sns.histplot(data[column],kde=True)
        plt.title(f'{column} - Histogram')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plot_outlier(data,'MODELYEAR'))
    st.pyplot(plot_outlier(data,'CYLINDERS'))
    st.pyplot(plot_outlier(data,'ENGINESIZE'))
    st.pyplot(plot_outlier(data,'FUELCONSUMPTION_CITY'))
    st.pyplot(plot_outlier(data,'FUELCONSUMPTION_HWY'))
    st.pyplot(plot_outlier(data,'FUELCONSUMPTION_COMB'))
    st.pyplot(plot_outlier(data,'FUELCONSUMPTION_COMB_MPG'))
    
    def remove_outlier(data,column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        
        IQR = Q3-Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        
        data = data[(data[column] >= lower) & (data[column] <= upper)]
        return data 
    
    data = remove_outlier(data,'MODELYEAR')
    data = remove_outlier(data,'CYLINDERS')
    data = remove_outlier(data,'ENGINESIZE')
    data = remove_outlier(data,'FUELCONSUMPTION_CITY')
    data = remove_outlier(data,'FUELCONSUMPTION_HWY')
    data = remove_outlier(data,'FUELCONSUMPTION_COMB')
    data = remove_outlier(data,'FUELCONSUMPTION_COMB_MPG')
    
    st.success('Data Setelah Outlier')
    st.pyplot(plot_outlier(data,'MODELYEAR'))
    st.pyplot(plot_outlier(data,'CYLINDERS'))
    st.pyplot(plot_outlier(data,'ENGINESIZE'))
    st.pyplot(plot_outlier(data,'FUELCONSUMPTION_CITY'))
    st.pyplot(plot_outlier(data,'FUELCONSUMPTION_HWY'))
    st.pyplot(plot_outlier(data,'FUELCONSUMPTION_COMB'))
    st.pyplot(plot_outlier(data,'FUELCONSUMPTION_COMB_MPG'))
    
    st.success('Data setelah outlier')
    st.write(f'Dataset : {data.shape}')

with st.expander('Modelling'):
    st.write('Splitting')
    data=data.drop(columns=['MAKE','MODEL',
                            'VEHICLECLASS','TRANSMISSION',
                            'FUELTYPE'])
    st.write(f'Dataset : {data.shape}')
    
    X_train,X_test,y_train,y_test = train_test_split(data.drop(['CO2EMISSIONS'],axis=1), 
                                                     data['CO2EMISSIONS'],
                                                     test_size=0.30)
    
    st.success('Apply Random Forest Regressor')
    rf_regressor = RandomForestRegressor(max_depth=2,random_state=0)
    rf_regressor.fit(X_train,y_train)
    #make prediction
    y_pred_rf = rf_regressor.predict(X_test)
    score = mean_absolute_error(y_test, y_pred_rf)
    st.write(score)
    

    
    
with st.sidebar:
    tahun = st.slider("Tahun Produksi", 2014, 2024, 2014)
    st.write("Tahun Produksi:", tahun)
    silinder = st.slider("Silinder", 3, 12, 4)
    st.write("Silinder:", silinder)
    engine = st.slider("Engine Size", 1.0, 9.0, 3.5)
    st.write("Engine Size:", engine)
    fc_city = st.slider("Konsumsi Bahan Bakar Dalam Kota", 4, 31, 15)
    st.write("Konsumsi Bahan Bakar Dalam Kota:", fc_city)
    fc_highway = st.slider("Konsumsi Bahan Bakar Luar Kota", 4, 22, 15)
    st.write("Konsumsi Bahan Bakar Luar Kota:", fc_highway)
    comb_h_c = st.slider("Kombinasi Bahan Bakar Dalam/Luar", 4, 26, 15)
    st.write("Kombinasi Bahan Bakar Dalam/Luar:", comb_h_c)
    mpg = st.slider("Kombinasi Bahan Bakar Dalam/Luar MPG", 10, 70, 30)
    st.write("Kombinasi Bahan Bakar Dalam/Luar MPG:", mpg)

with st.expander('Hasil Prediksi'):
    # st.success('Testing Data Baru')
    tahun = tahun
    silinder = silinder
    engine = engine
    fc_city = fc_city
    fc_highway = fc_highway
    comb_h_c = comb_h_c
    mpg = mpg
    data_baru = np.array([[tahun,silinder,engine,fc_city,fc_highway,comb_h_c,mpg]])
    prediksi = rf_regressor.predict(data_baru).reshape(1,-1)
    st.write(prediksi)