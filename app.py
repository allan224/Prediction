import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt
import pickle
model=pickle.load(open('model.pkl','rb'))

matplotlib.use('Agg')
from PIL import Image

st.title('Forest Fire Prediction/Analysis')
image=Image.open('forest.jpg')
st.image(image,use_column_width=True)

def predict_forest(oxygen,humidity,temperature):
    input=np.array([[oxygen,humidity,temperature]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    activities=['EDA','Visualisation','Prediction']
    option=st.sidebar.selectbox('Selection option:',activities)
    st.set_option('deprecation.showPyplotGlobalUse', False)
   
    if option=='EDA':
        st.subheader("Exploratory data analysis")
        data=st.file_uploader("Upload your dataset:",type=['csv','xlsx','txt','json'])
        
        if data is not None:
            st.success("Data successfully uploaded")
            df=pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox("Display shape"):
                st.write(df.shape)
            if st.checkbox("Display columns"):
                st.write(df.columns)
            if st.checkbox("Select multiple columns"):
                selected_column=st.multiselect('Select prefered columns:',df.columns)
                df1=df[selected_column]
                st.dataframe(df1)  
            if st.checkbox("Display summary"):
                st.write(df1.describe().T) 
            if st.checkbox("Display datatypes"):
                st.write(df.dtypes)
            if st.checkbox("Display Correlation of data various columns"):
                st.write(df.corr())
            
        
    




    elif option=='Visualisation':
        st.subheader("Visualisation")
        data=st.file_uploader("Upload your dataset:",type=['csv','xlsx','txt','json'])
        
        if data is not None:
            st.success("Data successfully uploaded")
            df=pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox('Select Multiple Columns to plot'):
                selected_column=st.multiselect('Select your preffered columns',df.columns)
                df1=df[selected_column]
                st.dataframe(df1)

            if st.checkbox('Display Heatmap'):
                st.write(sb.heatmap(df1.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
                st.pyplot()
            if st.checkbox('Display Pairplot'):
                st.write(sb.pairplot(df1,diag_kind='kde'))
                st.pyplot()
            if st.checkbox('Display Countplot'):
                st.write(df.Cover_Type.value_counts())
                st.write(sb.countplot(x='Cover_Type',data=df))
                st.pyplot()
            if st.checkbox('Display Histogram'):
                df1.hist(figsize=(13,11))
                plt.show()
                st.pyplot()

            if st.checkbox("Visualize Columns wrt Classes"):
                st.write("#### Select column to visualize: ")
                columns = df.columns.tolist()
                class_name = columns[-1]
                column_name = st.selectbox("",columns)
                st.write("#### Select type of plot: ")
                plot_type = st.selectbox("", ["kde","box", "violin","swarm"])
                if st.button("Generate"):
                    if plot_type == "kde":
                        st.write(sb.FacetGrid(df, hue=class_name, palette="husl", height=6).map(sb.kdeplot, column_name).add_legend())
                        st.pyplot()

                    if plot_type == "box":
                        st.write(sb.boxplot(x=class_name, y=column_name, palette="husl", data=df))
                        st.pyplot()

                    if plot_type == "violin":
                        st.write(sb.violinplot(x=class_name, y=column_name, palette="husl", data=df))

                        st.pyplot()
                    if plot_type == "swarm":
                        st.write(sb.swarmplot(x=class_name, y=column_name, data=df,color="y", alpha=0.9))
                        st.pyplot()

    elif option=='Prediction':
        st.subheader("Prediction")
        html_temp = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">Forest Fire Prediction ML App </h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        oxygen = st.text_input("Oxygen","")
        humidity = st.text_input("Humidity","")
        temperature = st.text_input("Temperature","")
        safe_html="""  
        <div style="background-color:#F4D03F;padding:10px >
        <h2 style="color:white;text-align:center;"> Your forest is safe</h2>
        </div>
        """
        danger_html="""  
        <div style="background-color:#F08080;padding:10px >
        <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
        </div>
        """

        if st.button("Predict"):
            output=predict_forest(oxygen,humidity,temperature)
            st.success('The probability of fire taking place is {}'.format(output))

            if output > 0.5:
                st.markdown(danger_html,unsafe_allow_html=True)
            else:
                st.markdown(safe_html,unsafe_allow_html=True)


if __name__=='__main__':
    main()