import streamlit as st
import pandas as pd
import numpy as np
from datetime import time
from datetime import datetime
import time

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))
st.table(dataframe)

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [-0.05, 109.35],
    columns=['lat', 'lon'])

st.map(map_data)

age = st.slider("How old are you?", 0, 130, 25)
st.write("I'm ", age, "years old")

values = st.slider("Select a range of values", 0.0, 100.0, (25.0, 75.0))
st.write("Values:", values)

# appointment = st.slider(
#     "Schedule your appointment:", value=(time(11, 30), time(12, 45))
# )
# st.write("You're scheduled for:", appointment)

start_time = st.slider(
    "When do you start?",
    value=datetime(2024, 1, 1, 9, 30),
    format="MM/DD/YY - hh:mm",
)
st.write("Start time:", start_time)

# st.button("Reset", type="primary")
# if st.button("Say hello"):
#     st.write("Why hello there")
# else:
#     st.write("Goodbye")

# if st.button("Aloha", type="primary"):
#     st.write("Ciao")
    

# left, middle, right = st.columns(3)
# if left.button("Plain button", use_container_width=True):
#     left.markdown("You clicked the plain button.")
# if middle.button("🔍 Emoji button", use_container_width=True):
#     middle.markdown("You clicked the emoji button.")
# if right.button("Material button", use_container_width=True):
#     right.markdown("You clicked the Material button.")

# option = st.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone"),
# )

# st.write("You selected:", option)

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

col1, col2 = st.columns(2)

with col1:
    st.checkbox("Disable selectbox widget")
    st.radio(
        "Set selectbox label visibility 👉",
        options=["visible", "hidden", "collapsed"],
    )

with col2:
    option = st.selectbox(
        "How would you like to be contacted?",
        ("Email", "Home phone", "Mobile phone"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'