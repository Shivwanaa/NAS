import streamlit as st
import circuittype
import fosters1 
import foster2
import subprocess # Assuming you have foster1.py in the same directory
import cauer1


# Streamlit app
st.title("Circuit Generator Interface")

# Sidebar for options
st.sidebar.title("Select Circuit Type")
option = st.sidebar.selectbox("Choose a circuit generation method:", ( "Foster1", "Foster2"))
a=2
function_type = None
# function_type = st.selectbox("Choose the function type:", ["Z(s)", "Y(s)"])
if(option=="Foster1"):
 function_type = st.selectbox("Choose the function type:", ["Z(s)"], key="function_type_selectbox")
elif(option=="Foster2"):
   function_type = st.selectbox("Choose the function type:", ["Y(s)"], key="function_type_selectbox")

if(option=="Cauer1"):
 input_numexpr = st.text_input(f"Enter the {function_type} function (e.g., 's*(s + 1)/(s + 2)'):")
 input_denexpr = st.text_input(f"Enter  {function_type} function (e.g., 's*(s + 1)/(s + 2)'):")
if(option=="Cauer1"):
   cauer1.function_type=function_type
   cauer1.numerator_input=input_numexpr
   cauer1.denominator_input=input_denexpr

if(option!="Cauer1"):
 input_expr = st.text_input(f"Enter the {function_type} function (e.g., 's*(s + 1)/(s + 2)'):")
if function_type == "Z(s)" and option=="Foster1":
    print("no issue")
    a=0
# elif function_type == "Y(s)" and option=="Foster1":
#     a=1
elif function_type == "Z(s)" and option=="Foster2":
    print("no issue")
    a=0
# elif function_type == "Y(s)" and option=="Foster2":
#     a=1
print(a)
st.write(a)



# Display an input box for the chosen function type



def generate_circuit(option, input_expr):
    if option == "Foster1":
        fosters1.func_str = input_expr
      # Set the global variable in fosters1.py
        fosters1.inputtype=a
        print("plssssssss")
        print(fosters1.inputtype)

        # Run fosters1.py as a separate process and capture output
        result = subprocess.run(["python3", "myenv/templates/fosters1.py"], capture_output=True, text=True)
        return result.stdout  # Return the captured output to display in Streamlit
    elif option == "Cauer1":
        cauer1.function_type = function_type
      # Set the global variable in fosters1.py
        cauer1.numerator_input=input_numexpr
        cauer1.denominator_input=input_denexpr
        print("plssssssss")
        print(fosters1.inputtype)

        # Run fosters1.py as a separate process and capture output
        result = subprocess.run(["python3", "myenv/templates/cauer1.py"], capture_output=True, text=True)
        return result.stdout  # Return the captured output to display in Streamlit
    elif option=="Foster2":
        foster2.func_str = input_expr
      # Set the global variable in fosters1.py
        foster2.inputtype=a
        print("plssssssss")
        print(foster2.inputtype)
    

       
    # elif option=="Cauer1":
    #     cauer1.func_str = input_expr
    #     cauer1.inputtype=a
    #     result = subprocess.run(["python3", "myenv/templates/cauer1.py"], capture_output=True, text=True)
    #     return result.stdout  # Return the captured output to display in Streamlit
    else:
        return "Circuit generation method not implemented yet."


# Define a function to generate and display the circuit based on selection
# def generate_circuit(option, input_expr):
#     # if option == "Cauer1":
#     #     circuit = Cauer1.generate_circuit(input_expr)  # Call the function from cauer1.py
#     # elif option == "Cauer2":
#     #     circuit = Cauer2.generate_circuit(input_expr)  # Call the function from cauer2.py
#     # el
#     if option == "Foster1":
#         circuit = fosters1.generate_circuit(input_expr)  # Call the function from foster1.py
#     # elif option == "Foster2":
#     #     circuit = foster2.generate_circuit(input_expr)  # Call the function from foster2.py
#     return circuit
if(option=="Foster1" or option=="Foster2"):
 if st.button("Circuit type and graph"):
    circuit_type = circuittype.get_circuit_type(input_expr)

    
if st.button("Generate Circuit"):
 if(option=="Foster1"):
    fosters1.inputtype=a
    fosters1.func_str = input_expr
      # Set the global func_str in fosters1
    result = fosters1.startcode()
      # Call the function to process the expression
    st.write(result)  # Display the result in the Streamlit app
 elif(option=="Foster2"):
    foster2.inputtype=a
    foster2.func_str=input_expr
    result=foster2.startcode()
    st.write(result)
 elif(option=="Cauer1"):
   cauer1.function_type=function_type
   cauer1.numerator_input=input_numexpr
   cauer1.denominator_input=input_denexpr
   result=cauer1.startcode()
   st.write(result)

   
# if input_expr:
#     st.write(f"Generating {option} circuit...")
#     circuit_result = generate_circuit(option, input_expr)
#     st.write("Circuit Result:")
#     st.text(circuit_result) 
else:
    st.warning("Please enter an input function.")
