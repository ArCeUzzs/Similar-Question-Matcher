import streamlit as st
import helper
import pickle

# Load the model
model = pickle.load(open('ImprovedNNmodel.pkl', 'rb'))

st.header('Similar Question Matcher')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

best_threshold = 0.38282012939453125


def streamlit_predict(model, user_input, threshold):
    # model.predict returns a 2D array with shape (1, 1) if there's one sample.
    y_prob = model.predict(user_input)[0][0]
    # Apply the best threshold to decide the class.
    y_pred = 1 if y_prob >= threshold else 0
    return y_pred


if st.button('Find'):
    # Create the query feature vector using your helper function
    query = helper.query_point_creator(q1, q2).reshape(1, -1)

    # Call the local streamlit_predict function directly (not as a method of model)
    result = streamlit_predict(model, query, best_threshold)

    if result:
        st.header('Same')
    else:
        st.header('Not Same')
