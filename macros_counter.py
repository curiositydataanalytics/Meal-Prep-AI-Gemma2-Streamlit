# Data manipulation
import numpy as np
import datetime as dt
import pandas as pd
import geopandas as gpd

# Database and file handling
import os
import json

# Data visualization
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
import pydeck as pdk

from langchain_community.llms import Ollama

path_cda = '\\CuriosityDataAnalytics'
path_wd = path_cda + '\\wd'
path_data = path_wd + '\\data'

# App config
#----------------------------------------------------------------------------------------------------------------------------------#
# Page config
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .element-container {
        margin-top: -2px;
        margin-bottom: -2px;
        margin-left: -2px;
        margin-right: -2px;
    }
    img[data-testid="stLogo"] {
                height: 6rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App title
st.title("Meal Prep AI")
st.divider()

llm = Ollama(model='gemma2')

if "pantry" not in st.session_state:
    st.session_state['pantry'] = None
if "mealplan" not in st.session_state:
    st.session_state['mealplan'] = None
if "recipe" not in st.session_state:
    st.session_state['recipe'] = None

with st.sidebar:
    st.logo(path_cda + '\\logo.png', size='large')

    food_list = st.data_editor(pd.DataFrame({'Food List' : ['']}), use_container_width=True, hide_index=True, num_rows="dynamic")

    if st.button('Generate Pantry'):
        with st.spinner('Loading pantry...'):
            prompt = f'''You have the following food items: {food_list['Food List'].unique().tolist()}.''' + '''
                        Please categorize these items into one of the five specified food groups based on common sense priority, ensuring that each item belongs to only one food group,
                        without any assumptions, explanations, or repetitions:

                        The food groups are:

                        1. Grains: e.g., rice, oats, bread, pasta.
                        2. Protein/Meat: e.g., chicken, fish, beans, eggs, tofu.
                        3. Fruits/Vegetables: e.g., apples, bananas, lettuce, tomatoes, carrots.
                        4. Dairy: e.g., milk, cheese, yogurt.
                        5. Sugars/Oils: e.g., butter, oils, chocolate, syrup, honey.
                        6. Miscellaneous: e.g., juice, coffee, spices.

                        If any items do not fit into the first five specified groups, categorize them under "miscellaneous."

                        Respond only with the following format, and strictly do not add any comments, explanations, or additional text:

                        Grains: Rice
                        Grains: Oatmeal
                        Grains: Bagel
                        Protein/Meat: Chicken
                        Protein/Meat: Steak
                        Fruits/Vegetables: Banana
                        Dairy: Milk
                        Miscellaneous: Cookie
                    '''

            response = llm.invoke(prompt)
            st.session_state['pantry'] = pd.DataFrame([line.split(": ") for line in response.strip().split("\n")], columns=['Food Group', 'Items'])

if st.session_state['pantry'] is None:
    st.warning('Add items to the food list.')
elif st.session_state['pantry'] is not None:

    pantry = st.session_state['pantry']

    # Create pantry
    st.header('My Pantry')
    pantry_gr = pantry.groupby('Food Group')['Items'].agg(list).reset_index()
    pantry_gr['Count'] = pantry_gr['Items'].apply(len)
    st.data_editor(
        pantry_gr.sort_values('Count', ascending=False),
        column_config={
            "Count": st.column_config.ProgressColumn(
                "Count",
                min_value=0,
                max_value=pantry_gr['Count'].max(),
                format="%d"
            ),
        },
        hide_index=True,
        use_container_width=True
    )

    # Create Meal Plan
    st.header('Meal Plan')
    if st.button('Create Meal Plan'):
        prompt = f'''You have the following food items: {food_list['Food List'].unique().tolist()}.''' + '''
                    Create a 5-day meal plan from Monday to Friday.
                    Each day should include three meals: breakfast, lunch, and dinner.
                    Ensure that each meal suggestion utilizes only the provided food items and is well-balanced.

                    Respond with the meal plan in the following format, without any additional comments, explanations, or text:

                    Monday | Breakfast | suggestion1
                    Monday | Lunch | suggestion2
                    Monday | Dinner | suggestion3
                    Tuesday | Breakfast | suggestion4
                    Tuesday | Lunch | suggestion5
                    Tuesday | Dinner | suggestion6
                    Wednesday | Breakfast | suggestion7
                    Wednesday | Lunch | suggestion8
                    Wednesday | Dinner | suggestion9
                    Thursday | Breakfast | suggestion10
                    Thursday | Lunch | suggestion11
                    Thursday | Dinner | suggestion12
                    Friday | Breakfast | suggestion13
                    Friday | Lunch | suggestion14
                    Friday | Dinner | suggestion15
                    '''
        response = llm.invoke(prompt)
        st.session_state['mealplan'] = pd.DataFrame([line.split(' | ') for line in response.strip().split('\n')], columns=['Day', 'Meal', 'Suggestion']).dropna()
    
    if st.session_state['mealplan'] is not None:
        mealplan = st.session_state['mealplan']
        df = mealplan.pivot(index='Meal', columns='Day', values='Suggestion').rename_axis(None, axis=1).reset_index()
        df = df[['Meal', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].copy()
        st.markdown(df.to_html(escape=False), unsafe_allow_html=True)

        st.header('Recipes')
        cols = st.columns((0.3,0.052,0.3), vertical_alignment='bottom')

        df = mealplan.copy()
        df['recipe'] = df.Day + ' - ' + df.Meal + ' - ' + df.Suggestion
        select_meal = cols[0].selectbox('', placeholder='Select Meal', options=df.recipe.unique())
        if cols[1].button('Create Recipe'):
            prompt = f'''Create a recipe for the following meal: {select_meal}.
                        Please include the following details:
                        1. A list of ingredients with quantities.
                        2. Step-by-step preparation instructions.

                        Only include the ingredients and instructions, without any additional comments, titles, or suggestions

                        Example format:
                        Ingredients:
                        - [Ingredient 1 with quantity]
                        - [Ingredient 2 with quantity]

                        Instructions:
                        1. [Step 1]
                        2. [Step 2]
                        '''
            response = llm.invoke(prompt)
            st.session_state['recipe'] = response

        if st.session_state['recipe'] is not None:
            subcols = st.columns((0.352,0.3))
            subcols[0].write(st.session_state['recipe'])

            if cols[2].button('Macros Breakdown'):
                prompt = f'''Generate a macros breakdown for the following recipe: {st.session_state['recipe']}.
                            Provide only:
                            1. Macros per Serving (calories, protein, carbohydrates, fiber, and fat).
                            2. Breakdown by Ingredient (including ingredient name, calories, protein, carbs, fiber, and fat).

                            Exclude any additional comments, titles, or suggestions.

                            Example format:
                            Macros per Serving:
                            - Calories: [value]
                            - Protein: [value]
                            - Carbohydrates: [value]
                            - Fiber: [value]
                            - Fat: [value]

                            Breakdown by Ingredient:
                            - [Ingredient 1]: [calories], [protein], [carbs], [fiber], [fat]
                            - [Ingredient 2]: [calories], [protein], [carbs], [fiber], [fat]

                            '''
                response = llm.invoke(prompt)
                subcols[1].write(response)          

        








#
#

