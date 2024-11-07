from sympy import symbols, expand, collect, degree
import sympy as sp
import schemdraw
import schemdraw.elements as elm
import streamlit as st
from io import BytesIO
# import cairosvg
function_type = None
numerator_input = None
denominator_input = None
# def draw_schematic(circuit_components, function_type):
#     img_buffer = BytesIO()
#     with schemdraw.Drawing() as d:
#         first_element_position = None 
#         in_parallel = (function_type == "Y(s)") 
#         in_series= (function_type == "Z(s)") 
#         prev_element_position = d.here  

#         for i, component in enumerate(circuit_components):
#             # Split and trim component type and value
#             comp_type, value = component.split("with value")
#             value = value.strip()
#             print(value)
#             if "Resistor" in comp_type:
#                 element = elm.Resistor().label(value)
#             elif "Inductor" in comp_type:
#                 element = elm.Inductor().label(value)
#             elif "Capacitor" in comp_type:
#                 element = elm.Capacitor().label(value)
#             else:
#                 continue  
#             if first_element_position is None:
#                 first_element_position = d.here  

#             # Alternate between parallel and series connections
#             if in_parallel:
#                 if i > 0:
#                     if i==len(circuit_components)-1:
#                         d.move(-3, 0)
#                         d += elm.Line().down().length(1.7)  # Offset downwards
#                         d += element 
#                         d += elm.Line().left().length(3)
#                     else:
                   
#                         d.push()
#                         d.move(-3, 0)
#                         d += elm.Line().down().length(1.7)  
#                         d += element 
#                         d += elm.Line().left().length(3) 
#                         d.pop() 
#                 else:
#                     d.push()
#                     d += elm.Line().down().length(1.7) 
#                     d += element.down() 
#                     d.pop()  
#                     if i==0:
#                         d.push()
#                         d += elm.Line().left().length(3)
#                         d.move(3, -3.29)
#                         d += elm.Line().left().length(3)
#                         d.pop()
#                     d.move(3, 0)                    
#             else:         
#                 d.move(-5, 0)
#                 d += element
#                 if i==len(circuit_components)-1:                     
#                         d += elm.Line().down().length(3.29) 
#                         d += elm.Line().left().length(3)
        
#             in_parallel = not in_parallel
#             prev_element_position = d.here
#         # d.draw()
#         x_min, x_max, y_min, y_max = d.ax.get_xlim()[0], d.ax.get_xlim()[1], d.ax.get_ylim()[0], d.ax.get_ylim()[1]
#         if not (x_min and x_max and y_min and y_max):
#             d.ax.set_xlim(-10, 10)
#             d.ax.set_ylim(-10, 10)
#         st.write("cirucit elements")
#         d.save(img_buffer)
#     img_buffer.seek(0)

#     # Display the schematic in Streamlit

#     st.image(img_buffer)

import schemdraw
import schemdraw.elements as elm
from io import BytesIO
import streamlit as st

def draw_schematic(circuit_components, function_type):
    # Create an in-memory buffer
    img_buffer = BytesIO()

    # Create the schematic drawing
    with schemdraw.Drawing() as d:
        first_element_position = None
        in_parallel = (function_type == "Y(s)")
        in_series = (function_type == "Z(s)")
        prev_element_position = d.here

        # Track maximum and minimum coordinates for axis limits
        x_min, x_max, y_min, y_max = None, None, None, None

        for i, component in enumerate(circuit_components):
            # Split and trim component type and value
            comp_type, value = component.split("with value")
            value = value.strip()

            if "Resistor" in comp_type:
                element = elm.Resistor().label(value)
            elif "Inductor" in comp_type:
                element = elm.Inductor().label(value)
            elif "Capacitor" in comp_type:
                element = elm.Capacitor().label(value)
            else:
                continue

            # Store the position of the first element
            if first_element_position is None:
                first_element_position = d.here

            # Alternate between parallel and series connections
            if in_parallel:
                if i > 0:
                    if i == len(circuit_components) - 1:
                        d.move(-3, 0)
                        d += elm.Line().down().length(1.7)
                        d += element
                        d += elm.Line().left().length(3)
                    else:
                        d.push()
                        d.move(-3, 0)
                        d += elm.Line().down().length(1.7)
                        d += element
                        d += elm.Line().left().length(3)
                        d.pop()
                else:
                    d.push()
                    d += elm.Line().down().length(1.7)
                    d += element.down()
                    d.pop()
                    if i == 0:
                        d.push()
                        d += elm.Line().left().length(3)
                        d.move(3, -3.29)
                        d += elm.Line().left().length(3)
                        d.pop()
                    d.move(3, 0)
            else:
                d.move(-5, 0)
                d += element
                if i == len(circuit_components) - 1:
                    d += elm.Line().down().length(3.29)
                    d += elm.Line().left().length(3)

            in_parallel = not in_parallel
            prev_element_position = d.here

            # Track the min/max coordinates to set axis limits
            x, y = d.here
            if x_min is None or x < x_min:
                x_min = x
            if x_max is None or x > x_max:
                x_max = x
            if y_min is None or y < y_min:
                y_min = y
            if y_max is None or y > y_max:
                y_max = y

        # If axis limits were not set, assign defaults
        if x_min is None or y_min is None:
            x_min, x_max, y_min, y_max = -10, 10, -10, 10  # Default axis limits

        # Set axis limits before showing the figure
        d.ax.set_xlim(x_min - 1, x_max + 1)  # Add a little margin to avoid clipping
        d.ax.set_ylim(y_min - 1, y_max + 1)

        # Save the schematic to the buffer
        d.draw()  # Ensure drawing is complete before saving
        img_buffer.seek(0)
        d.save(img_buffer)  # Save to buffer

    # Seek back to the beginning of the buffer before displaying it
    img_buffer.seek(0)

    # Display the schematic in Streamlit
    st.image(img_buffer)

s = symbols('s')



def arrange_rational_function(numerator, denominator):
    expanded_num = collect(expand(numerator), s, evaluate=True)
    expanded_den = collect(expand(denominator), s, evaluate=True)
    return expanded_num, expanded_den

# def get_user_inputinusableform():
#     numerator = eval(numerator_input)
#     denominator = eval(denominator_input)
    
#     return function_type, numerator, denominator

def ensure_higher_degree(numerator, denominator):
    if degree(numerator) < degree(denominator):
        return False  # Indicate that swapping will occur
    return True  # Indicate that no swapping is needed

# Get user-defined function type, numerator, and denominator
# function_type, numerator, denominator = get_user_input()

# Check if the numerator has a greater degree than the denominator
# needs_swapping = not ensure_higher_degree(numerator, denominator)

# arranged_num, arranged_den = arrange_rational_function(numerator, denominator)

# Print the results
# print("\nNumerator in ascending order of powers of s:", arranged_num)
# print("Denominator in ascending order of powers of s:", arranged_den)
# print("Function type has been changed to:", function_type)

# n = sp.apart(f'{arranged_num/arranged_den}')
# print("Partial fraction form of the function:", n)
# num, den = sp.fraction(n)  
# leading_t = num.as_ordered_terms()[-1] 
# leading_c = leading_t.as_coeff_mul(s)[0]
# print("Leading coefficient:", leading_c)

# if leading_c<0:
#     arranged_num = arranged_den 
#     needs_swapping=True 

# # Swap if necessary and update the function type
# if needs_swapping:
#     numerator, denominator = denominator, numerator  # Swap if necessary
#     if function_type == "y(s)":
#         function_type = "Z(s)"  # Change to impedance
#     elif function_type == "z(s)":
#         function_type = "Y(s)"  # Change to admittance
# quotient_terms=[]
# while True:
#  try:
#     quotient, _ = sp.div(numerator, denominator)
#     leading_term = quotient.as_ordered_terms()[0] if quotient.has(s) else quotient
#     product = sp.expand(leading_term* denominator)
    
#     remainder = sp.expand(numerator - product)
#     quotient_terms.append(leading_term)
#     if remainder.is_zero:
#         break
#     numerator = denominator  
#     denominator = remainder 
#  except Exception as e:
#         print(f"Error during division: {e}")
#         integ=numerator/denominator
#         quotient_terms.append(integ)
#         break
# print(quotient_terms)



def startcode():
    s = sp.symbols('s')
    # st.write(function_type)
    function_type = st.selectbox("Choose the function type:", ["Z(s)", "Y(s)"])
    numerator=eval(numerator_input)
    denominator=eval(denominator_input)
    needs_swapping = not ensure_higher_degree(numerator, denominator)
    arranged_num, arranged_den = arrange_rational_function(numerator, denominator)
    n = sp.apart(f'{arranged_num/arranged_den}')
    print("Partial fraction form of the function:", n)
    num, den = sp.fraction(n)  
    leading_t = num.as_ordered_terms()[-1] 
    # leading_c = leading_t.as_coeff_mul(s)[0]
    if leading_t.has(s):
     leading_c = leading_t.as_coeff_mul(s)[0]
    else:
     leading_c = leading_t  # Assume it's a constant term if it doesn't contain 's'

    print("Leading coefficient:", leading_c)

    if leading_c<0:
        arranged_num = arranged_den 
        needs_swapping=True 

    # Swap if necessary and update the function type
    if needs_swapping:
        numerator, denominator = denominator, numerator  # Swap if necessary
        if function_type == "Y(s)":
            function_type = "Z(s)"  # Change to impedance
        elif function_type == "Z(s)":
            function_type = "Y(s)"  # Change to admittance
    quotient_terms=[]
    while True:
     try:
        quotient, _ = sp.div(numerator, denominator)
        leading_term = quotient.as_ordered_terms()[0] if quotient.has(s) else quotient
        product = sp.expand(leading_term* denominator)
        
        remainder = sp.expand(numerator - product)
        quotient_terms.append(leading_term)
        if remainder.is_zero:
            break
        numerator = denominator  
        denominator = remainder 
     except Exception as e:
            print(f"Error during division: {e}")
            integ=numerator/denominator
            quotient_terms.append(integ)
            break
    print(quotient_terms)
    circuit_components = []  # Move this outside to accumulate all components

    for i, term in enumerate(quotient_terms):
        s = sp.symbols('s')
        numerator, denominator = term.as_numer_denom()
        num_degree = sp.Poly(numerator, s).degree()
        den_degree = sp.Poly(denominator, s).degree()
        
        # Print term details for debugging
        print(f"Processing term {i}: {term}")
        print(f"Numerator degree: {num_degree}, Denominator degree: {den_degree}")
        
        component = ""  # Initialize component

        if function_type == "Y(s)":
            if i % 2 == 0:  # Even index logic
                if num_degree == den_degree == 0:
                    co = term
                    component = f"Resistor with value {1/co} Ω"
                elif num_degree == 0 and den_degree == 1:
                    x = denominator.coeff(s, 1)
                    y = denominator / x
                    if (y - s) == 0:
                        co = numerator / x
                        component = f"Inductor with value {1/co} H"
                elif num_degree == 1 and den_degree == 0:
                    co = term.coeff(s, 1)
                    component = f"Capacitor with value {co} F"
            else:  # Odd index logic
                if num_degree == 0 and den_degree == 0:
                    co = term
                    component = f"Resistor with value {co} Ω"
                elif num_degree == 0 and den_degree == 1:
                    x = denominator.coeff(s, 1)
                    y = denominator / x
                    if (y - s) == 0:
                        co = numerator / x
                        component = f"Capacitor with value {1/co} F"
                elif num_degree == 1 and den_degree == 0:
                    co = term.coeff(s, 1)
                    component = f"Inductor with value {co} H"
        else:  # Logic for "Z(s)"
            if i % 2 == 0:  # Even index logic
                if num_degree == 0 and den_degree == 0:
                    co = term
                    component = f"Resistor with value {co} Ω"
                elif num_degree == 0 and den_degree == 1:
                    x = denominator.coeff(s, 1)
                    y = denominator / x
                    if (y - s) == 0:
                        co = numerator / x
                        component = f"Capacitor with value {1/co} F"
                elif num_degree == 1 and den_degree == 0:
                    co = term.coeff(s, 1)
                    component = f"Inductor with value {co} H"
            else:  # Odd index logic
                if num_degree == 0 and den_degree == 0:
                    co = term
                    component = f"Resistor with value {1/co} Ω"
                elif num_degree == 0 and den_degree == 1:
                    x = denominator.coeff(s, 1)
                    y = denominator / x
                    if (y - s) == 0:
                        co = numerator / x
                        component = f"Inductor with value {1/co} H"
                elif num_degree == 1 and den_degree == 0:
                    co = term.coeff(s, 1)
                    component = f"Capacitor with value {co} F"
        circuit_components.append(component)

    # Print all components after processing all terms
    print("\nCircuit components:")
    print(circuit_components)
    draw_schematic(circuit_components, function_type)


if __name__ == "__main__":
    startcode()
