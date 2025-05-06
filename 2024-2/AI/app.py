
import streamlit as st
import pandas as pd

# Data setup
data = {
    'item_name': [
        'Chicken Bowl', 'Chicken Burrito', 'Chips and Guacamole', 'Steak Burrito', 'Canned Soft Drink',
        'Steak Bowl', 'Chips', 'Bottled Water', 'Chicken Soft Tacos', 'Chips and Fresh Tomato Salsa',
        'Chicken Salad Bowl', 'Canned Soda', 'Side of Chips', 'Veggie Burrito', 'Barbacoa Burrito',
        'Veggie Bowl', 'Carnitas Bowl', 'Barbacoa Bowl', 'Carnitas Burrito', 'Steak Soft Tacos'
    ],
    'count': [
        726, 553, 479, 368, 301, 211, 211, 162, 115, 110, 110, 104, 101, 95, 91, 85, 68, 66, 59, 55
    ]
}
df = pd.DataFrame(data)
df.set_index('item_name', inplace=True)

# Categorize items
def categorize(item):
    if any(x in item.lower() for x in ['drink', 'water', 'soda']):
        return 'Drink'
    return 'Food'

df['category'] = df.index.map(categorize)

# Get food and drink options
food_options = df[df['category'] == 'Food'].index.tolist()
drink_options = df[df['category'] == 'Drink'].index.tolist()

st.title("🍽️ Online Menu")

st.subheader("Today's Specials: Chicken Bowl, Steak Burrito")

# --- Food Selection ---
st.subheader("🍛 Select a Food Item")
selected_food = st.selectbox("Choose your food:", food_options)
food_quantity = st.number_input("Quantity of food", min_value=0, max_value=10, value=0, step=1)

# --- Drink Selection ---

st.subheader("🥤 Select a Drink Item")
selected_drink = st.selectbox("Choose your drink:", drink_options)
drink_quantity = st.number_input("Quantity of drink", min_value=0, max_value=10, value=0, step=1)

# --- Show Order ---
if food_quantity > 0 or drink_quantity > 0:
    st.subheader("🧾 Order Summary")
    summary = []
    if food_quantity > 0:
        summary.append({"Item": selected_food, "Quantity": food_quantity})
    if drink_quantity > 0:
        summary.append({"Item": selected_drink, "Quantity": drink_quantity})
    st.table(pd.DataFrame(summary))
else:
    st.info("Please select quantities for food or drink to see the order summary.")


