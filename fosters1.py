import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sympy as sp
import re
import schemdraw
import schemdraw.elements as elm
import io
func_str=None
inputtype=None
import streamlit as st

# def generate_circuit():
#     global func_str
#     try:
#         s = sp.symbols('s')
#         function = sp.sympify(func_str)
#         numerator, denominator = function.as_numer_denom()

#         # Placeholder for actual circuit generation logic
#         result = f"Processed function with numerator: {numerator}, denominator: {denominator}"
#         print(result)  # Printing the result to capture it in app.py
#     except sp.SympifyError as e:
#         print(f"Error parsing expression: {e}")

def generate_circuit():
    global func_str
    try:
        s = sp.symbols('s')
        function = sp.sympify(func_str)
        numerator, denominator = function.as_numer_denom()

        # Placeholder for actual circuit generation logic
        print("numeratorshivukrith")
        result = f"Processed function with numerator: {numerator}, denominator: {denominator}"
        return result  # Return the result instead of printing it
    except sp.SympifyError as e:
        return f"Error parsing expression: {e}"  # Return error message

def decomposeRL(transfer_function):
    print("decompose")
    s = sp.symbols('s')
    if(inputtype==1):
        transfer_function=1/transfer_function
    transfer_function=transfer_function/s
    numerator, denominator = transfer_function.as_numer_denom()
    num_degree = numerator.as_poly(s).degree() if numerator.has(s) else 0
    den_degree = denominator.as_poly(s).degree() if denominator.has(s) else 0
    
    if(num_degree>=den_degree):
        quotient, remainder = polynomial_division(numerator, denominator)
        remainder_fraction = remainder / denominator
        decomposed_remainder = sp.apart(remainder_fraction)
        print(quotient + decomposed_remainder)
        if(st.button("SOLVED")):
            st.write(quotient+decomposed_remainder)
        return quotient + decomposed_remainder
        
    else:
        if(st.button("SOLVED")):
            st.write(sp.apart(f'{numerator/denominator}'))

        return sp.apart(f'{numerator/denominator}')

def map_partial_fractions_to_circuitsRL(dam):
    s = sp.symbols('s')
    terms = sp.Add.make_args(dam)
    circuit_componentsRL = []
    for term in terms:
        numerator, denominator = term.as_numer_denom()
        num_degree = sp.Poly(numerator, s).degree()
        den_degree = sp.Poly(denominator, s).degree()
        x=denominator.coeff(s,1)
        if num_degree==0 and den_degree==1 and (denominator/x)-s==0:
            co=numerator/x
            component = f"Resistor with value {co}Ω"
        elif num_degree==0 and den_degree==1 and (denominator/x)-s!=0:           
            y=denominator/x-s
            z=numerator/x
            l=z/y
            component = f"Resistor with value {z}Ω in parallel with Inductor with value {l}H"
        elif num_degree==0 and den_degree==0:
            x=term
            component = f"Inductor with value {x}H"

        circuit_componentsRL.append(component)

    return circuit_componentsRL
def draw_circuit(circuit_components):
    with schemdraw.Drawing() as d:
        for i, component in enumerate(circuit_components):
            if "in parallel with" in component:
                # Leave a gap before starting the parallel section
                d += elm.Line().right().length(1)
                
                # Split the components to draw them in parallel (one above, one below)
                sub_components = component.split(" in parallel with ")
                print(sub_components)
                d.push()  # Save the current position for the start of the parallel branches
                
                # Draw the branch going up (first component)
                d += elm.Line().up().length(1)  # Move up for the first branch
                  # Save position for the first branch
                if "Inductor" in sub_components[0]:
                    value = sub_components[0].split("value ")[1]
                    d += elm.Inductor().right().label(value, loc='bottom')
                    d += elm.Line().down().length(1)
                elif "Capacitor" in sub_components[0]:
                    value = sub_components[0].split("value ")[1]
                    d += elm.Capacitor().right().label(value, loc='bottom')
                    d += elm.Line().down().length(1)
                elif "Resistor" in sub_components[0]:
                    value = sub_components[0].split("value ")[1]
                    d += elm.Resistor().right().label(value, loc='bottom')
                    d += elm.Line().down().length(1)
                d.pop()
                d += elm.Line().down().length(1)  # Bring the first branch back down
                d.push()
                d.pop()
                  # Save position for the second branch
                if "Inductor" in sub_components[1]:
                    value = sub_components[1].split("value ")[1]
                    d += elm.Inductor().right().label(value, loc='bottom')
                    d += elm.Line().up().length(1)
                elif "Capacitor" in sub_components[1]:
                    value = sub_components[1].split("value ")[1]
                    d += elm.Capacitor().right().label(value, loc='bottom')
                    d += elm.Line().up().length(1)
                elif "Resistor" in sub_components[1]:
                    value = sub_components[1].split("value ")[1]
                    d += elm.Resistor().right().label(value, loc='bottom')
                    d += elm.Line().up().length(1)
               
                d += elm.Dot()  # Connect branches back together
                d += elm.Line().right().length(3)  # Extend to the right after closing the parallel

            else:
                # Draw standard series components
                if "Inductor" in component:
                    value = component.split("value ")[1]
                    d += elm.Inductor().label(value, loc='bottom')
                elif "Capacitor" in component:
                    value = component.split("value ")[1]
                    d += elm.Capacitor().label(value, loc='bottom')
                elif "Resistor" in component:
                    value = component.split("value ")[1]
                    d += elm.Resistor().label(value, loc='bottom')
                else:
                    print(f"Unrecognized component format: {component}")
        
        # Ensure the circuit is completed by drawing a wire to the start
        d += elm.Line().down().length(2)  # Extend to the right before closing
        d += elm.Line().left().tox(0)  # Connect back to the starting x-position
        


    img_buffer = io.BytesIO()
    # d.save(img_buffer, fmt='png')
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0) 
    st.image(img_buffer)    

# Function to validate the input transfer function
def validate_input(func_str):
    # Check for negative coefficients/terms
  print("validating")
  print(inputtype)
  if(inputtype==0):
   if re.search(r'-\s*\d*\.?\d*\s*', func_str):  # Match negative numbers
        st.write("Invalid input")
        return False
  return True
  
# Function to extract coefficients from an expression using sympy
def parse_function(func_str):
    # Parse the function using sympy
    s = sp.symbols('s')
    print("Y(s)")
    print(inputtype)
    if '/' not in func_str:
        num_expr=func_str
        den_expr='1'

    else: 
     num_expr, den_expr = func_str.split('/')

    num_poly = sp.expand(sp.sympify(num_expr.strip()))
    
    den_poly = sp.expand(sp.sympify(den_expr.strip()))



    
    # Handle cases where the polynomial is a constant (degree 0)
    num_poly = sp.Poly(num_poly, s) if num_poly.has(s) else sp.Poly(num_poly, s, domain='QQ')
    den_poly = sp.Poly(den_poly, s) if den_poly.has(s) else sp.Poly(den_poly, s, domain='QQ')

    # Get the coefficients as lists
    num_coeffs = num_poly.all_coeffs()
    den_coeffs = den_poly.all_coeffs()

    # Convert to floats for signal.TransferFunction compatibility
    num_coeffs = [float(c) for c in num_coeffs]
    den_coeffs = [float(c) for c in den_coeffs]

    return num_coeffs, den_coeffs, num_poly, den_poly

def polynomial_division(numerator, denominator):
    """Perform polynomial division and return the quotient and remainder."""
    s = sp.symbols('s')
    quotient, remainder = sp.div(numerator, denominator)
    return quotient, remainder

def partial_fraction_decomposition(function):
    """Decompose the function into partial fractions."""
    s = sp.symbols('s')
    numerator, denominator = function.as_numer_denom()

    # Check if the polynomials are constant and set degrees accordingly
    num_degree = numerator.as_poly(s).degree() if numerator.has(s) else 0
    den_degree = denominator.as_poly(s).degree() if denominator.has(s) else 0

    print("Degree of numerator:", num_degree)
    print("Degree of denominator:", den_degree)

    if num_degree > den_degree:
        quotient, remainder = polynomial_division(numerator, denominator)
        remainder_fraction = remainder / denominator
        decomposed_remainder = sp.apart(remainder_fraction)
        print("Decomposed Partial Fraction:")
        return quotient + decomposed_remainder
    else:
        return sp.apart(function)

def map_partial_fractions_to_circuits(partial_fractions):
    s = sp.symbols('s')
    terms = sp.Add.make_args(partial_fractions)
    circuit_components = []
    

    for term in terms:
        numerator, denominator = term.as_numer_denom()
        num_degree = sp.Poly(numerator, s).degree()
        den_degree = sp.Poly(denominator, s).degree()
       
        if num_degree==1 and den_degree==0:
            co=term.coeff(s,1)
            component = f"Inductor with value {co}H"
        elif num_degree==0 and den_degree==0:
            co=term
            component = f"Resistor with value {co}Ω"
        elif num_degree==0 and den_degree==1:
            x=denominator.coeff(s,1)
            y=denominator/x
            if (y-s)==0:
                co=numerator/denominator.coeff(s,1)
                component = f"Capacitor with value {1/co}F"
            else:
                b=(denominator-s*x)/x
                co=numerator/((denominator-s*x)/b)
                component = f"Capacitor with value {1/co}F in parallel with Resistor with value {co/b}Ω"
        elif num_degree==1 and den_degree==1:
            co=numerator.coeff(s,1)
            b=denominator-s
            component = f"Resistor with value {co}Ω in parallel with Inductor with value {co/b}H"
        elif num_degree==1 and den_degree==2:
            co=numerator.coeff(s,1)/denominator.coeff(s,2)
            b=(denominator-(denominator.coeff(s,2)*((s**2))))/denominator.coeff(s,2)
            component = f"Inductor with value {co/b}H in parallel with Capacitor with value {1/co}F"
        else:
            component = f"Unrecognized form: {term}"

        circuit_components.append(component)

    return circuit_components
# Function to check if poles and zeros are valid (on real or imaginary axes only)
def check_validity(zeros, poles):
    for z in zeros:
        if not (np.isclose(np.imag(z), 0) or np.isclose(np.real(z), 0)):
            return False
    for p in poles:
        if not (np.isclose(np.imag(p), 0) or np.isclose(np.real(p), 0)):
            return False
    return True

# Function to check if poles and zeros are alternating and identify the circuit type
def check_alternating_and_identify(poles, zeros):
    # Combine poles and zeros and sort them along the real and imaginary axes
    combined_points = [(z, 'zero') for z in zeros] + [(p, 'pole') for p in poles]
    combined_points.sort(key=lambda x: (np.real(x[0]), np.imag(x[0])))
    # Check for alternating pattern and identify the closest point to the origin
    last_type = None
    is_alternating = True
    closest_point = min(combined_points, key=lambda x: np.abs(x[0]))

    for point, point_type in combined_points:
        if last_type is None:
            last_type = point_type
        else:
            if last_type == point_type:
                is_alternating = False
                break
            last_type = point_type

    if is_alternating:
        # Determine axis type
        if np.all(np.isclose(np.imag([p[0] for p in combined_points]), 0)):
            axis_type = "real"
        elif np.all(np.isclose(np.real([p[0] for p in combined_points]), 0)):
            axis_type = "imaginary"
        else:
            axis_type = "mixed"

        # Identify circuit type based on the closest point to the origin
        if axis_type == "imaginary":
            st.write("LC circuit (alternating along the imaginary axis)")
            d="LC"
            print("LC circuit (alternating along the imaginary axis)")
        elif axis_type == "real":
            if closest_point[1] == 'pole':
                d="RC"
                # print("RC circuit (alternating along the real axis, closest point is a pole)")
                st.write("RC circuit (alternating along the real axis, closest point is a pole)")
            elif closest_point[1] == 'zero':
                d="RL"
                st.write("RL circuit (alternating along the real axis, closest point is a zero)")
                print("RL circuit (alternating along the real axis, closest point is a zero)")
        else:
            d="INVALID"
            st.write("Mixed axis detected, cannot classify as a standard LC, RC, or RL circuit.")
            print("Mixed axis detected, cannot classify as a standard LC, RC, or RL circuit.")
    else:
        d="INVALID"
        # st.write("Invalid Case: Poles and zeros are not alternating along the axis.")
        print("Invalid Case: Poles and zeros are not alternating along the axis.")
    return d

# Take user input for the transfer function
# func_str = input("Enter the transfer function in the format 'numerator/denominator' (e.g., '(s*3 + s) / ((s + 3)*3 * (s + 1))'): ")

# Validate input for negative terms
        
# if not validate_input(func_str):
#     print("Invalid input: Transfer function should not contain negative terms.")
# else:
#     # Parse the function to get numerator and denominator coefficients and expressions
#     numerator, denominator, num_poly, den_poly = parse_function(func_str)

#     # Create a transfer function system
#     system = signal.TransferFunction(numerator, denominator)

#     # Find the poles and zeros
#     zeros = system.zeros
#     poles = system.poles

#     def format_values(values):
#         return [complex(0, round(z.imag, 10)) if np.isclose(z.real, 0, atol=1e-10) else z for z in values]

#     # Format the poles and zeros
#     formatted_zeros = format_values(zeros)
#     formatted_poles = format_values(poles)

#     # Plotting the poles and zeros
#     plt.figure()
#     plt.scatter(np.real(formatted_zeros), np.imag(formatted_zeros), s=50, color='red', label='Zeroes', marker='o')
#     plt.scatter(np.real(formatted_poles), np.imag(formatted_poles), s=50, color='blue', label='Poles', marker='x')

#     plt.axhline(0, color='black', linewidth=0.5)
#     plt.axvline(0, color='black', linewidth=0.5)
#     plt.xlabel('Real Part')
#     plt.ylabel('Imaginary Part')
#     plt.title('Poles and Zeroes Plot')
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     # plt.legend()
#     # plt.show()

#     # Print partial fraction decomposition
#     s = sp.symbols('s')
#     transfer_function = num_poly/den_poly
#     partial_fractions = partial_fraction_decomposition(transfer_function)
#     print("\nPartial Fraction Decomposition:")
#     print(partial_fractions)

#     # Check the validity of poles and zeros
#     if not check_validity(zeros, poles):
#         print("Invalid Case: Poles and zeros must be on the real or imaginary axes.")
#     else:
#         # Check if poles and zeros are alternating and identify the circuit type
#         check_alternating_and_identify(poles, zeros)

#     circuit_components = map_partial_fractions_to_circuits(partial_fractions)
#     print("\nCircuit Component Mapping:")
#     for component in circuit_components:
#         print(component)

#     if not check_validity(formatted_zeros, formatted_poles):
#         print("Invalid Case: Poles and zeros must be on the real or imaginary axes.")
#     else:
#         check_alternating_and_identify(formatted_poles, formatted_zeros)

#     print("\nThese individual components are connected together in series.")
    

#     draw_circuit(circuit_components)

# def startcode():
#   print("shivwanaa")
#   generate_circuit()
  
#   if not validate_input(func_str) :
#     print("Invalid input: Transfer function should not contain negative terms.")
#   else:
#     # Parse the function to get numerator and denominator coefficients and expressions
    
#     # if(inputtype==1):
#     #     func_str=1/func_str
#     numerator, denominator, num_poly, den_poly = parse_function(func_str)

#     # Create a transfer function system
#     system = signal.TransferFunction(numerator, denominator)

#     # Find the poles and zeros
#     zeros = system.zeros
#     poles = system.poles

#     def format_values(values):
#         return [complex(0, round(z.imag, 10)) if np.isclose(z.real, 0, atol=1e-10) else z for z in values]

#     # Format the poles and zeros
#     formatted_zeros = format_values(zeros)
#     formatted_poles = format_values(poles)

#     # Plotting the poles and zeros
#     plt.figure()
#     plt.scatter(np.real(formatted_zeros), np.imag(formatted_zeros), s=50, color='red', label='Zeroes', marker='o')
#     plt.scatter(np.real(formatted_poles), np.imag(formatted_poles), s=50, color='blue', label='Poles', marker='x')

#     plt.axhline(0, color='black', linewidth=0.5)
#     plt.axvline(0, color='black', linewidth=0.5)
#     plt.xlabel('Real Part')
#     plt.ylabel('Imaginary Part')
#     plt.title('Poles and Zeroes Plot')
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.savefig('poles_and_zeros_plot.png', dpi=300)
#     # pyplot(plt)st.
#     # plt.legend()
#     # plt.show()

# # Save the plot as an image

#     # Print partial fraction decomposition
    
#     transfer_function = num_poly/den_poly
#     partial_fractions = partial_fraction_decomposition(transfer_function)
#     print("\nPartial Fraction Decomposition:")
#     print(partial_fractions)
    

#     # Check the validity of poles and zeros
#     if not check_validity(zeros, poles):
#         print("Invalid Case: Poles and zeros must be on the real or imaginary axes.")
#     else:
#         # Check if poles and zeros are alternating and identify the circuit type
#         cir=check_alternating_and_identify(poles, zeros)
#     if (cir=="RL"):
        
#         dam=decomposeRL(transfer_function)
#         circuitRLcomponents=map_partial_fractions_to_circuitsRL(dam)
#         draw_circuit(circuitRLcomponents)
#     else:
#      circuit_components = map_partial_fractions_to_circuits(partial_fractions)
#      print("\nCircuit Component Mapping:")
#      for component in circuit_components:
#         print(component)

#      if not check_validity(formatted_zeros, formatted_poles):
#         print("Invalid Case: Poles and zeros must be on the real or imaginary axes.")
#      else:
#         check_alternating_and_identify(formatted_poles, formatted_zeros)

#      print("\nThese individual components are connected together in series.")
    

#      draw_circuit(circuit_components)





def startcode():
 generate_circuit()
 if not validate_input(func_str):
    print("Invalid input: Transfer function should not contain negative terms.")
 else:
    # Parse the function to get numerator and denominator coefficients and expressions
    numerator, denominator, num_poly, den_poly = parse_function(func_str)

    # Create a transfer function system
    system = signal.TransferFunction(numerator, denominator)

    # Find the poles and zeros
    zeros = system.zeros
    poles = system.poles

    def format_values(values):
        return [complex(0, round(z.imag, 10)) if np.isclose(z.real, 0, atol=1e-10) else z for z in values]

    # Format the poles and zeros
    formatted_zeros = format_values(zeros)
    formatted_poles = format_values(poles)

    # Plotting the poles and zeros
    plt.figure()
    plt.scatter(np.real(formatted_zeros), np.imag(formatted_zeros), s=50, color='red', label='Zeroes', marker='o')
    plt.scatter(np.real(formatted_poles), np.imag(formatted_poles), s=50, color='blue', label='Poles', marker='x')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Poles and Zeroes Plot')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend()
    # plt.show()
    

    # Print partial fraction decomposition
    s = sp.symbols('s')
    transfer_function = num_poly/den_poly
    partial_fractions = partial_fraction_decomposition(transfer_function)
    print("\nPartial Fraction Decomposition:")
    print(partial_fractions)

    # Check the validity of poles and zeros
    if not check_validity(zeros, poles):
        print("Invalid Case: Poles and zeros must be on the real or imaginary axes.")
    else:
        # Check if poles and zeros are alternating and identify the circuit type
        cir=check_alternating_and_identify(poles, zeros)
        print(cir)

        ###RLLLL
        if (cir=="RL"):
         dam=decomposeRL(transfer_function)
         circuitRLcomponents=map_partial_fractions_to_circuitsRL(dam)
         draw_circuit(circuitRLcomponents)

        elif(cir=="RC" or cir=="LC" ):

         circuit_components = map_partial_fractions_to_circuits(partial_fractions)
         print("\nCircuit Component Mapping:")
         for component in circuit_components:
            print(component)

         if not check_validity(formatted_zeros, formatted_poles):
            print("Invalid Case: Poles and zeros must be on the real or imaginary axes.")
         else:
            check_alternating_and_identify(formatted_poles, formatted_zeros)

         print("\nThese individual components are connected together in series.")
        

         draw_circuit(circuit_components)
        elif(cir=="INVALID"):
         st.write("Circuit cannot be generated for such a case")
if __name__ == "__main__":
    startcode()

        
